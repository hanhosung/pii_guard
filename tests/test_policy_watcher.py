"""
Tests for PII-Guard policy hot-reload watcher (Sub-AC 5b).

Scope
-----
This module targets the *watcher* functionality of :class:`PolicyLoader`:
the background thread that detects filesystem changes and atomically swaps
the live config object, including:

* Change detection within a reasonable time-bound (AC: "within a timeout").
* Debounce: rapid successive writes coalesce into a single reload that fires
  only after the file has been stable for the settling window.
* Error guard: a parse-invalid write retains the last-good config; the watcher
  continues and picks up the next valid write.
* File-lifecycle events: deletion reverts to SECURE_DEFAULTS; recreation picks
  up the new content.
* Atomic swap: the ``config`` property is never in a partial state during reload.
* No silent pass: even under error conditions the config degrades to the
  most-restrictive fallback rather than going open.

Test structure
--------------
TestWatcherBasicReload       — core "change detected within timeout" behaviour
TestWatcherDebounce          — debounce semantics (settling window, coalescing)
TestWatcherErrorGuard        — parse / schema errors retain last-good config
TestWatcherFileLifecycle     — deletion and recreation events
TestWatcherAtomicSwap        — concurrent reads see a consistent config object
TestWatcherStopResume        — start / stop / restart lifecycle
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from pii_guard.policy import (
    SECURE_DEFAULTS,
    PolicyLoader,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write(path: Path, content: str, mtime_offset: float = 0.05) -> None:
    """
    Write *content* to *path* and advance the mtime by *mtime_offset* seconds
    so that ``stat()`` always sees a new timestamp regardless of OS clock
    resolution.  This avoids spurious "not changed" returns on fast filesystems
    where two writes land in the same 1-second bucket.
    """
    path.write_text(content, encoding="utf-8")
    t = time.time() + mtime_offset
    os.utime(str(path), (t, t))


def _wait_for(
    condition,
    timeout: float = 3.0,
    poll: float = 0.02,
    msg: str = "condition not met within timeout",
) -> None:
    """Busy-poll *condition()* until it returns truthy or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(poll)
    raise AssertionError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# TestWatcherBasicReload
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherBasicReload:
    """The watcher must detect file mutations and update the config in-process."""

    def test_change_detected_within_timeout(self, tmp_path):
        """
        Mutate the YAML and assert the in-process config reflects the new value
        within 3 seconds.  This is the primary AC-5b acceptance test.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            _write(p, "fail_mode: closed\n")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=3.0,
                msg="Config was not updated to 'closed' within 3 s of file mutation",
            )
        finally:
            loader.stop_watcher()

    def test_multiple_fields_updated_atomically(self, tmp_path):
        """
        All changed fields appear together after reload (no partial-state
        window observed from outside the lock).
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\nmemory_budget_mb: 256\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            _write(p, "fail_mode: closed\nmemory_budget_mb: 512\nrehydrate: false\n")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=3.0,
                msg="fail_mode not updated",
            )
            # By the time fail_mode is "closed", the other fields must also
            # reflect the same reload (atomic swap).
            assert loader.config.memory_budget_mb == 512
            assert loader.config.rehydrate is False
        finally:
            loader.stop_watcher()

    def test_watcher_source_tag_updated_on_reload(self, tmp_path):
        """After reload, config.source still references the policy file path."""
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            _write(p, "fail_mode: closed\n")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=3.0,
            )
            assert str(p) in loader.config.source
        finally:
            loader.stop_watcher()

    def test_category_override_picked_up_by_watcher(self, tmp_path):
        """Per-category changes inside the watcher window are applied."""
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            _write(p, "fail_mode: open\ncategories:\n  EMAIL:\n    action: allow\n")
            _wait_for(
                lambda: loader.config.categories.get("EMAIL") is not None,
                timeout=3.0,
                msg="EMAIL category override not loaded by watcher",
            )
            assert loader.config.categories["EMAIL"].action == "allow"
        finally:
            loader.stop_watcher()

    def test_watcher_no_false_update_when_file_unchanged(self, tmp_path):
        """
        The ``reload_if_changed()`` mtime guard must prevent spurious reloads.
        Write a file, let the watcher run for a bit, then check that the
        ``loaded_at`` timestamp doesn't advance without a file change.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        original_loaded_at = loader.config.loaded_at

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            time.sleep(0.4)  # Let the watcher run several ticks with no change
            assert loader.config.loaded_at == original_loaded_at, (
                "Config was reloaded even though the file did not change"
            )
        finally:
            loader.stop_watcher()


# ─────────────────────────────────────────────────────────────────────────────
# TestWatcherDebounce
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherDebounce:
    """
    Debounce semantics: rapid successive writes must coalesce into a single
    reload that fires only after the settling window expires.
    """

    def test_debounce_delays_reload_beyond_settling_window(self, tmp_path):
        """
        After a file write, the reload must NOT fire before the debounce window
        has passed.  We check within half the debounce window that the config
        has not yet updated, then wait for the full window to elapse.
        """
        debounce = 0.3
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.02, debounce=debounce)
        try:
            assert loader.config.fail_mode == "open"
            # Write a change and immediately sample.
            _write(p, "fail_mode: closed\n")
            # Within half the debounce window the reload should NOT have fired.
            time.sleep(debounce / 2)
            assert loader.config.fail_mode == "open", (
                "Reload fired before the debounce window expired "
                f"(waited {debounce/2:.2f}s < debounce={debounce}s)"
            )
            # After the full debounce window the reload should have fired.
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=debounce * 4,
                msg=f"Config not updated after {debounce * 4:.2f}s",
            )
        finally:
            loader.stop_watcher()

    def test_debounce_coalesces_rapid_writes(self, tmp_path):
        """
        Multiple rapid writes within the debounce window must produce exactly
        one reload reflecting the **last** write — no intermediate value should
        ever be visible through ``loader.config``.
        """
        debounce = 0.3
        p = tmp_path / "policy.yaml"
        _write(p, "memory_budget_mb: 256\n")
        loader = PolicyLoader(str(p))
        assert loader.config.memory_budget_mb == 256

        loader.start_watcher(interval=0.02, debounce=debounce)
        try:
            # Rapid writes — each resets the debounce clock on new mtime.
            _write(p, "memory_budget_mb: 512\n", mtime_offset=0.05)
            time.sleep(0.05)
            _write(p, "memory_budget_mb: 768\n", mtime_offset=0.10)
            time.sleep(0.05)
            _write(p, "memory_budget_mb: 1024\n", mtime_offset=0.15)

            # Give the debounce time to settle on the last write.
            _wait_for(
                lambda: loader.config.memory_budget_mb in (1024,),
                timeout=debounce * 5,
                msg="Final memory_budget_mb not applied after rapid writes",
            )
            # Must land on the final value (coalesced, not intermediate).
            assert loader.config.memory_budget_mb == 1024
        finally:
            loader.stop_watcher()

    def test_debounce_zero_reloads_immediately(self, tmp_path):
        """
        With debounce=0 the reload fires on the next poll tick after any mtime
        change — debounce is effectively disabled.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.02, debounce=0)
        try:
            _write(p, "fail_mode: closed\n")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=1.0,
                msg="Config not updated within 1 s with debounce=0",
            )
        finally:
            loader.stop_watcher()

    def test_debounce_resets_on_new_write_during_window(self, tmp_path):
        """
        A second write during an active debounce window resets the clock so
        the reload is delayed by another full debounce period from the second
        write.  This means the first write's value is never applied.
        """
        debounce = 0.3
        p = tmp_path / "policy.yaml"
        _write(p, "memory_budget_mb: 256\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.02, debounce=debounce)
        try:
            # First write — starts debounce clock.
            _write(p, "memory_budget_mb: 512\n", mtime_offset=0.05)
            # Wait slightly less than debounce, then write again.
            time.sleep(debounce * 0.4)
            # Second write resets the debounce clock.
            _write(p, "memory_budget_mb: 1024\n", mtime_offset=0.10)

            # After total settle, only the second value should be visible.
            _wait_for(
                lambda: loader.config.memory_budget_mb == 1024,
                timeout=debounce * 5,
                msg="Second write's value not applied",
            )
            # First intermediate value (512) was never the stable config.
            assert loader.config.memory_budget_mb == 1024
        finally:
            loader.stop_watcher()


# ─────────────────────────────────────────────────────────────────────────────
# TestWatcherErrorGuard
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherErrorGuard:
    """
    Error guard: if the watcher sees an invalid file it must retain the
    last-valid config and keep watching for the next valid write.
    """

    def test_yaml_parse_error_retains_last_good_config(self, tmp_path):
        """
        An invalid YAML write must not corrupt the in-process config; the next
        valid write must still be picked up correctly.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            # Inject syntactically invalid YAML.
            _write(p, "fail_mode: {unclosed_brace\n")
            time.sleep(0.5)   # Enough time for the watcher to attempt a reload.

            # Last-good config must be retained — not reverted to defaults.
            assert loader.config.fail_mode == "open", (
                "YAML parse error must not corrupt the last-good config"
            )

            # Recovery: write a valid file; the watcher must pick it up.
            _write(p, "fail_mode: closed\n")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=3.0,
                msg="Valid write not picked up after earlier parse error",
            )
        finally:
            loader.stop_watcher()

    def test_schema_error_retains_last_good_config(self, tmp_path):
        """
        A schema-invalid write (bad enum value) retains the last-good config.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            _write(p, "fail_mode: totally_wrong_value\n")
            time.sleep(0.5)

            assert loader.config.fail_mode == "open", (
                "Schema error must not corrupt last-good config"
            )
        finally:
            loader.stop_watcher()

    def test_partial_write_recovered_by_subsequent_valid_write(self, tmp_path):
        """
        Simulates an editor truncating a file mid-write (leaves empty content),
        followed by a complete write.  The watcher must end up with the final
        valid state.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            # Truncate (empty file) — treated as empty policy, defaults apply.
            _write(p, "")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=3.0,
                msg="Empty file should revert to closed default",
            )

            # Complete write with a specific value.
            _write(p, "fail_mode: open\nmemory_budget_mb: 512\n")
            _wait_for(
                lambda: loader.config.memory_budget_mb == 512,
                timeout=3.0,
                msg="Final complete write not picked up after empty-file recovery",
            )
        finally:
            loader.stop_watcher()

    def test_error_guard_does_not_stop_watcher(self, tmp_path):
        """
        After a parse error the watcher thread must stay alive and continue
        monitoring.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            # Bad write.
            _write(p, "---\nbad: [unclosed\n")
            time.sleep(0.5)

            # Thread must still be alive.
            assert loader._watcher_thread is not None
            assert loader._watcher_thread.is_alive(), (
                "Watcher thread died after a parse error"
            )

            # And must still detect subsequent changes.
            _write(p, "fail_mode: closed\n")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=3.0,
                msg="Watcher stopped detecting changes after error",
            )
        finally:
            loader.stop_watcher()


# ─────────────────────────────────────────────────────────────────────────────
# TestWatcherFileLifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherFileLifecycle:
    """Watcher must handle file deletion and recreation correctly."""

    def test_file_deletion_reverts_to_secure_defaults(self, tmp_path):
        """
        When the policy file is deleted, the watcher must revert to
        SECURE_DEFAULTS (fail-closed) within the watch window.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\non_content_failure: warn_allow\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            p.unlink()
            _wait_for(
                lambda: loader.config is SECURE_DEFAULTS,
                timeout=3.0,
                msg="Config did not revert to SECURE_DEFAULTS after file deletion",
            )
            assert loader.config.fail_mode == "closed"
            assert loader.config.on_content_failure == "block"
        finally:
            loader.stop_watcher()

    def test_file_recreation_after_deletion_picked_up(self, tmp_path):
        """
        After a file is deleted (reverts to defaults) and then recreated with
        new content, the watcher must pick up the new content.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            # Delete and wait for revert.
            p.unlink()
            _wait_for(
                lambda: loader.config is SECURE_DEFAULTS,
                timeout=3.0,
                msg="Did not revert to SECURE_DEFAULTS after deletion",
            )

            # Recreate with different content.
            _write(p, "fail_mode: open\nmemory_budget_mb: 512\n")
            _wait_for(
                lambda: loader.config.memory_budget_mb == 512,
                timeout=3.0,
                msg="Recreated file content not picked up by watcher",
            )
            assert loader.config.fail_mode == "open"
        finally:
            loader.stop_watcher()

    def test_watcher_starts_with_missing_file_and_detects_creation(self, tmp_path):
        """
        A watcher started when no policy file exists must detect the file
        appearing and load it.
        """
        p = tmp_path / "policy.yaml"
        # File does not exist yet.
        loader = PolicyLoader(str(p))
        assert loader.config is SECURE_DEFAULTS

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            # File created after watcher is running.
            _write(p, "fail_mode: open\n")
            _wait_for(
                lambda: loader.config.fail_mode == "open",
                timeout=3.0,
                msg="Watcher did not detect newly created policy file",
            )
        finally:
            loader.stop_watcher()


# ─────────────────────────────────────────────────────────────────────────────
# TestWatcherAtomicSwap
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherAtomicSwap:
    """
    Concurrent readers must never observe a partially-constructed config object.
    The swap must appear instantaneous (under the RLock).
    """

    def test_concurrent_readers_see_consistent_config(self, tmp_path):
        """
        Spin up N reader threads that continuously read ``loader.config``.
        While they run, trigger a reload via file mutation.  No reader should
        ever see a config where the fields are inconsistent (e.g. fail_mode
        from the new config but memory_budget_mb from the old).
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\nmemory_budget_mb: 256\n")
        loader = PolicyLoader(str(p))

        inconsistencies: List[str] = []
        stop_readers = threading.Event()

        def _reader() -> None:
            while not stop_readers.is_set():
                cfg = loader.config
                # The only valid combinations are old or new — never mixed.
                fm = cfg.fail_mode
                mb = cfg.memory_budget_mb
                # Old state: open + 256.  New state: closed + 512.
                if fm == "open" and mb != 256:
                    inconsistencies.append(f"open/{mb}")
                elif fm == "closed" and mb != 512:
                    inconsistencies.append(f"closed/{mb}")
                # Any other fail_mode is also unexpected.
                elif fm not in ("open", "closed"):
                    inconsistencies.append(f"unknown fail_mode={fm!r}")

        threads = [threading.Thread(target=_reader, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()

        loader.start_watcher(interval=0.02, debounce=0.1)
        try:
            _write(p, "fail_mode: closed\nmemory_budget_mb: 512\n")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=3.0,
                msg="Reload not detected by watcher",
            )
        finally:
            stop_readers.set()
            loader.stop_watcher()

        for t in threads:
            t.join(timeout=2)

        assert inconsistencies == [], (
            f"Concurrent readers saw inconsistent config states: {inconsistencies}"
        )

    def test_config_property_never_returns_none(self, tmp_path):
        """
        ``loader.config`` must always return a non-None PolicyConfig, even
        during a reload.
        """
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        none_seen = threading.Event()

        def _reader() -> None:
            for _ in range(500):
                if loader.config is None:  # type: ignore[comparison-overlap]
                    none_seen.set()
                time.sleep(0.001)

        loader.start_watcher(interval=0.02, debounce=0.05)
        try:
            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            _write(p, "fail_mode: closed\n")
            t.join(timeout=2)
        finally:
            loader.stop_watcher()

        assert not none_seen.is_set(), "loader.config returned None during reload"


# ─────────────────────────────────────────────────────────────────────────────
# TestWatcherStopResume
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherStopResume:
    """Watcher start/stop/restart lifecycle."""

    def test_stop_terminates_thread(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        loader.start_watcher(interval=0.05, debounce=0.1)
        assert loader._watcher_thread is not None
        assert loader._watcher_thread.is_alive()

        loader.stop_watcher()

        # Thread should be terminated and reference cleared.
        assert loader._watcher_thread is None

    def test_restart_after_stop_detects_changes(self, tmp_path):
        """Stop and then restart the watcher; changes must still be detected."""
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        loader.stop_watcher()

        # Restart.
        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            _write(p, "fail_mode: closed\n")
            _wait_for(
                lambda: loader.config.fail_mode == "closed",
                timeout=3.0,
                msg="Restarted watcher did not detect change",
            )
        finally:
            loader.stop_watcher()

    def test_double_start_uses_same_thread(self, tmp_path):
        """Calling start_watcher twice must not create a second thread."""
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))

        loader.start_watcher(interval=0.05, debounce=0.1)
        first = loader._watcher_thread
        loader.start_watcher(interval=0.05, debounce=0.1)  # no-op
        assert loader._watcher_thread is first
        loader.stop_watcher()

    def test_stop_without_start_is_safe(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        loader.stop_watcher()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# TestWatcherGetFileMtime
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherGetFileMtime:
    """
    Unit tests for the ``_get_file_mtime()`` internal helper that drives
    change detection in the watcher loop.
    """

    def test_returns_mtime_for_existing_file(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        mtime = loader._get_file_mtime()
        assert mtime is not None
        assert isinstance(mtime, float)
        assert mtime > 0

    def test_returns_none_for_missing_file(self, tmp_path):
        loader = PolicyLoader(str(tmp_path / "nonexistent.yaml"))
        assert loader._get_file_mtime() is None

    def test_returns_none_for_no_path(self):
        loader = PolicyLoader(None)
        assert loader._get_file_mtime() is None

    def test_returns_updated_mtime_after_write(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n", mtime_offset=0.0)
        loader = PolicyLoader(str(p))
        mtime_before = loader._get_file_mtime()

        _write(p, "fail_mode: closed\n", mtime_offset=1.0)
        mtime_after = loader._get_file_mtime()

        assert mtime_after is not None
        assert mtime_after > mtime_before  # type: ignore[operator]

    def test_returns_none_after_deletion(self, tmp_path):
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader._get_file_mtime() is not None

        p.unlink()
        assert loader._get_file_mtime() is None


# ─────────────────────────────────────────────────────────────────────────────
# TestWatcherSecureByDefault (no-silent-pass regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestWatcherSecureByDefault:
    """
    Regression suite: the watcher must never transition the live config to
    an open/unprotected state except by explicit user action (valid YAML with
    permissive values).

    Specifically:
    - File deletion → SECURE_DEFAULTS (fail-closed), never open.
    - Parse error → last-good retained, never silent open.
    - Start with no file → SECURE_DEFAULTS, never open.
    """

    def test_deleted_file_never_opens_gateway(self, tmp_path):
        """Deleting the policy file must never result in fail_mode == 'open'."""
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: open\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "open"

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            p.unlink()
            _wait_for(
                lambda: loader.config is SECURE_DEFAULTS,
                timeout=3.0,
                msg="Did not revert to SECURE_DEFAULTS after deletion",
            )
            # Must be fail-closed, not open.
            assert loader.config.fail_mode == "closed"
            assert loader.config.on_content_failure == "block"
        finally:
            loader.stop_watcher()

    def test_parse_error_never_opens_gateway(self, tmp_path):
        """A parse-error write must not transition a fail-closed config to open."""
        p = tmp_path / "policy.yaml"
        _write(p, "fail_mode: closed\n")
        loader = PolicyLoader(str(p))
        assert loader.config.fail_mode == "closed"

        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            # Inject a broken write.
            _write(p, "fail_mode: open\nbad: [unclosed\n")
            time.sleep(0.5)
            # Must remain closed — last-good.
            assert loader.config.fail_mode == "closed"
        finally:
            loader.stop_watcher()

    def test_watcher_with_no_initial_file_is_fail_closed(self, tmp_path):
        """When started with no file, the watcher's initial state is SECURE_DEFAULTS."""
        loader = PolicyLoader(str(tmp_path / "missing.yaml"))
        loader.start_watcher(interval=0.05, debounce=0.1)
        try:
            cfg = loader.config
            assert cfg.fail_mode == "closed"
            assert cfg.on_content_failure == "block"
            assert cfg.unscannable_action == "block"
        finally:
            loader.stop_watcher()
