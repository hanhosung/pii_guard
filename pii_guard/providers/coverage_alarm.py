"""
Coverage Alarm Emission and Action Enforcement — Sub-AC 3b.

Consumes the delta objects produced by the schema-coverage detection layer
(:mod:`pii_guard.providers.schema_coverage`) and:

  (a) Emits a structured :class:`CoverageAlarmEvent` for **each non-empty
      delta** (i.e. every :class:`FieldDelta` with ``extra_keys`` non-empty
      and every :class:`VersionDelta` with ``is_unknown=True``).

  (b) Applies the configured ``unknown_field_action`` policy:

      ``"block"`` (strict mode)
          Each emitted alarm sets ``should_block=True``.  The caller must
          not forward the request.

      ``"warn_allow"`` (permissive mode)
          Each emitted alarm has ``should_block=False``.  The request may
          proceed; the alarm is logged for audit purposes.

This module is **purely a consumer** of pre-built delta objects — it has
zero dependency on the live schema-comparison logic in
:mod:`schema_coverage`.  It can be tested by injecting synthetic deltas.

Design invariants
-----------------
  - An **empty** delta (``extra_keys`` is empty / ``is_unknown=False``)
    produces **no** alarm.  Only genuinely novel fields/versions trigger
    alarms, preventing false-positive noise.
  - ``should_block`` on the :class:`CoverageAlarmResult` is the logical OR
    of every individual alarm's ``should_block`` flag — a single blocking
    alarm is sufficient to block the whole request.
  - No PII or raw payload content is ever stored in the alarm.  Only the
    structural metadata (path, extra_keys, version string) is recorded, and
    none of those fields contain user-data.

Usage
-----
    from pii_guard.providers.schema_coverage import diff_claude_fields, diff_api_version
    from pii_guard.providers.coverage_alarm import apply_coverage_alarm_policy

    field_deltas = diff_claude_fields(payload)
    version_delta = diff_api_version(api_version, "claude") if api_version else None

    all_deltas = list(field_deltas)
    if version_delta:
        all_deltas.append(version_delta)

    result = apply_coverage_alarm_policy(all_deltas, unknown_field_action="block")
    if result.should_block:
        # Return 400; do not forward payload
        ...
    for alarm in result.alarms:
        ledger.record_coverage_alarm(alarm)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from .schema_coverage import FieldDelta, VersionDelta

log = logging.getLogger(__name__)

# Union type for all delta objects consumed by this module.
AnyDelta = Union[FieldDelta, VersionDelta]


# ─────────────────────────────────────────────────────────────────────────────
# Public result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoverageAlarmEvent:
    """
    A structured audit event emitted for a single non-empty delta.

    Every field here is purely structural metadata — no user data, no raw
    PII/secret literal is stored.

    Attributes
    ----------
    alarm_type:
        ``"unknown_fields"`` for a :class:`FieldDelta`-originated alarm, or
        ``"unknown_version"`` for a :class:`VersionDelta`-originated alarm.
    provider:
        ``"claude"``, ``"openai"``, or ``"gemini"``.
    path:
        Dot-/bracket-notation path inside the payload where the unknown
        keys were found (``""`` = root).  Set to the ``location`` field of
        the originating :class:`VersionDelta` for version alarms.
    extra_keys:
        Frozenset of unknown field names (non-empty for field alarms;
        empty for version alarms).
    declared_version:
        The unrecognized version string (version alarms only; ``None`` for
        field alarms).
    is_future:
        ``True`` when the declared version appears newer than every
        known-good version (version alarms only; ``False`` otherwise).
    should_block:
        Whether the ``unknown_field_action`` policy mandates blocking this
        request.  ``True`` in strict mode (``"block"``), ``False`` in
        permissive mode (``"warn_allow"``).
    source_delta:
        The original delta object that generated this alarm.  Stored as a
        reference for callers that need to inspect raw delta metadata.
        Not serialized to logs/ledger by default.
    """

    alarm_type: str                      # "unknown_fields" | "unknown_version"
    provider: str
    path: str
    extra_keys: frozenset                # set of str; empty for version alarms
    declared_version: Optional[str]      # None for field alarms
    is_future: bool                      # False for field alarms
    should_block: bool                   # determined by unknown_field_action
    source_delta: Any = field(
        default=None, repr=False, compare=False
    )

    def as_log_dict(self) -> dict:
        """
        Return a ledger-safe dict — no raw user data.

        The ``source_delta`` field is intentionally excluded.
        """
        d: dict = {
            "alarm_type": self.alarm_type,
            "provider": self.provider,
            "path": self.path,
            "should_block": self.should_block,
        }
        if self.extra_keys:
            # Sort for deterministic output; these are field names, not PII.
            d["extra_keys"] = sorted(self.extra_keys)
        if self.declared_version is not None:
            d["declared_version"] = self.declared_version
            d["is_future"] = self.is_future
        return d


@dataclass
class CoverageAlarmResult:
    """
    Aggregate result of processing all deltas for one request.

    Attributes
    ----------
    alarms:
        All :class:`CoverageAlarmEvent` objects emitted (one per non-empty
        delta).  May be empty when there are no coverage issues.
    should_block:
        Logical OR of every alarm's ``should_block``.  When ``True``, the
        caller must not forward the request.
    unknown_field_action:
        The policy mode that was applied — ``"block"`` or ``"warn_allow"``.
    """

    alarms: List[CoverageAlarmEvent]
    should_block: bool
    unknown_field_action: str


# ─────────────────────────────────────────────────────────────────────────────
# Core alarm-emission helpers
# ─────────────────────────────────────────────────────────────────────────────

def _emit_field_alarm(
    delta: FieldDelta,
    *,
    unknown_field_action: str,
) -> CoverageAlarmEvent:
    """
    Emit a :class:`CoverageAlarmEvent` for a :class:`FieldDelta`.

    ``should_block`` is ``True`` iff ``unknown_field_action == "block"``.
    """
    blocking = unknown_field_action == "block"
    return CoverageAlarmEvent(
        alarm_type="unknown_fields",
        provider=delta.provider,
        path=delta.path,
        extra_keys=delta.extra_keys,
        declared_version=None,
        is_future=False,
        should_block=blocking,
        source_delta=delta,
    )


def _emit_version_alarm(
    delta: VersionDelta,
    *,
    unknown_field_action: str,
) -> CoverageAlarmEvent:
    """
    Emit a :class:`CoverageAlarmEvent` for a :class:`VersionDelta`.

    ``should_block`` is ``True`` iff ``unknown_field_action == "block"``.
    """
    blocking = unknown_field_action == "block"
    return CoverageAlarmEvent(
        alarm_type="unknown_version",
        provider=delta.provider,
        path=delta.location,
        extra_keys=frozenset(),
        declared_version=delta.declared_version,
        is_future=delta.is_future,
        should_block=blocking,
        source_delta=delta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def emit_coverage_alarms(
    deltas: List[AnyDelta],
    *,
    unknown_field_action: str = "block",
) -> List[CoverageAlarmEvent]:
    """
    Emit :class:`CoverageAlarmEvent` objects for every non-empty delta.

    This is the low-level emission function.  For most callers the
    higher-level :func:`apply_coverage_alarm_policy` is more convenient.

    Parameters
    ----------
    deltas:
        List of :class:`FieldDelta` and/or :class:`VersionDelta` objects.
        Objects whose delta is "clean" (``extra_keys`` empty or
        ``is_unknown=False``) are silently skipped.
    unknown_field_action:
        ``"block"`` — emitted alarms will have ``should_block=True``.
        ``"warn_allow"`` — emitted alarms will have ``should_block=False``.

    Returns
    -------
    list[CoverageAlarmEvent]
        One event per non-empty input delta.  Empty when all deltas are
        clean.
    """
    alarms: List[CoverageAlarmEvent] = []

    for delta in deltas:
        if isinstance(delta, FieldDelta):
            # Only non-empty extra_keys represent a real coverage issue.
            if delta.extra_keys:
                alarm = _emit_field_alarm(delta, unknown_field_action=unknown_field_action)
                alarms.append(alarm)
                _log_alarm(alarm)

        elif isinstance(delta, VersionDelta):
            # Only truly unknown versions produce an alarm.
            if delta.is_unknown:
                alarm = _emit_version_alarm(delta, unknown_field_action=unknown_field_action)
                alarms.append(alarm)
                _log_alarm(alarm)

        # Unknown delta types are silently skipped to remain forward-compatible.

    return alarms


def apply_coverage_alarm_policy(
    deltas: List[AnyDelta],
    *,
    unknown_field_action: str = "block",
) -> CoverageAlarmResult:
    """
    Apply the ``unknown_field_action`` policy to a list of schema deltas.

    Consumes the delta objects produced by the detection layer, emits a
    structured :class:`CoverageAlarmEvent` for each non-empty delta, and
    determines whether the overall request should be blocked or allowed.

    Parameters
    ----------
    deltas:
        List of :class:`FieldDelta` and/or :class:`VersionDelta` objects.
        Typically the output of
        :func:`~pii_guard.providers.schema_coverage.diff_claude_fields`,
        :func:`~pii_guard.providers.schema_coverage.diff_openai_fields`,
        :func:`~pii_guard.providers.schema_coverage.diff_gemini_fields`,
        and/or
        :func:`~pii_guard.providers.schema_coverage.diff_api_version`.

        **No live schema-comparison logic is invoked here** — the caller
        is responsible for producing the deltas.  Pass pre-built objects
        for unit testing.
    unknown_field_action:
        Policy mode:

        ``"block"`` (default / strict mode)
            Any non-empty delta → ``should_block=True`` on the result.
            The caller must return an error to the client (HTTP 400 / 403)
            and must not forward the payload to the upstream LLM.

        ``"warn_allow"`` (permissive mode)
            Alarms are emitted and logged, but ``should_block`` remains
            ``False``.  The request proceeds.

    Returns
    -------
    CoverageAlarmResult
        ``.alarms`` contains one event per non-empty delta.
        ``.should_block`` is the aggregate blocking decision.
    """
    alarms = emit_coverage_alarms(deltas, unknown_field_action=unknown_field_action)
    should_block = any(a.should_block for a in alarms)

    return CoverageAlarmResult(
        alarms=alarms,
        should_block=should_block,
        unknown_field_action=unknown_field_action,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal logging helper
# ─────────────────────────────────────────────────────────────────────────────

def _log_alarm(alarm: CoverageAlarmEvent) -> None:
    """
    Log a coverage alarm at the appropriate level.

    Uses ``WARNING`` level in blocking mode (the request will be refused)
    and ``INFO`` level in permissive (warn_allow) mode.

    No raw payload content is logged — only structural metadata.
    """
    msg_parts = [
        f"PII-Guard coverage alarm: provider={alarm.provider!r}",
        f"path={alarm.path!r}",
        f"alarm_type={alarm.alarm_type!r}",
    ]
    if alarm.extra_keys:
        msg_parts.append(f"extra_keys={sorted(alarm.extra_keys)!r}")
    if alarm.declared_version is not None:
        msg_parts.append(f"declared_version={alarm.declared_version!r}")
        msg_parts.append(f"is_future={alarm.is_future}")
    msg_parts.append(f"action={'BLOCK' if alarm.should_block else 'WARN_ALLOW'}")

    message = " | ".join(msg_parts)
    if alarm.should_block:
        log.warning(message)
    else:
        log.info(message)
