"""
PII-Guard policy config loader (Sub-AC 5a).

Design
------
*PolicyConfig* is the typed, in-process policy object — a dataclass with
secure built-in defaults for every knob.  Callers can use it without any
on-disk file.

*PolicyLoader* wraps a YAML file path, validates the schema on load, and
provides atomic hot-reload semantics:

  * **File missing / deleted** → falls back to ``SECURE_DEFAULTS`` (never to
    unprotected).
  * **File present but schema-invalid** → retains the last-valid config and
    logs the error (never silently switches to open / unprotected).
  * **File changes** → detected by ``mtime`` polling; ``reload_if_changed()``
    can be called directly or driven by a background watcher thread
    (``start_watcher()`` / ``stop_watcher()``).

Pin-list change guard
---------------------
Pin-list entries are stored as opaque hashes (never raw values).  When the
pin-list changes relative to the last approved snapshot, the new entries are
rejected unless ``pin_list_approved: true`` is also set in the YAML.  This
requires out-of-band user action (editing a file outside the agent's write
permission) — the agent cannot silently promote its own values.

YAML schema (all fields optional — secure defaults apply for omitted fields)::

    version: "1"

    # Global failure knobs
    fail_mode: closed                    # closed | open
    on_content_failure: block            # block | warn_allow
    on_infra_failure: degrade_to_stage1  # degrade_to_stage1 | block
    stage2_fail_action: mask_known_only  # block | mask_known_only | open
    unscannable_action: block            # block | warn_allow | ocr

    # Round-trip rehydration for agent-facing content
    rehydrate: true                      # bool

    # Proxy memory budget
    memory_budget_mb: 1024               # int ≥ 128

    # Per-category action/confidence overrides
    categories:
      EMAIL:
        action: tokenize_roundtrip       # allow|mask|block|tokenize_roundtrip
        mask_style: tokenize             # tokenize|partial|format_preserving
        min_confidence: 0.90            # float [0,1]
        stage2_fail_action: mask_known_only  # optional per-category override
      API_KEY:
        action: block

    # Project-scoped allowlist (regex patterns — matched values skip detection)
    allowlist:
      - pattern: "test@example\\.com"
        label: "CI fixture email"
      - "another-literal-pattern"        # shorthand string form

    # Pin-list (stored as hashes, not raw values)
    # Changes here require pin_list_approved: true in the same file
    pin_list:
      - hash: "sha256:abcdef..."
        category: EMAIL
        action: allow
        label: "internal relay address"
    pin_list_approved: false             # set true after reviewing any change

    # Per-channel policy overrides
    channel_overrides:
      cli:
        unscannable_action: warn_allow
      ouroboros:
        stage2_fail_action: block
        fail_mode: closed

    # Proximity (context-gated detection, R17) — all keys optional, omit to keep defaults
    proximity:
      enabled: true
      window_chars: 25
      account_triggers: [국민, 신한, 우리, 하나, 농협, 토스뱅크, 카카오뱅크, 입금, 이체, 계좌]
      biz_triggers: [사업자]
      password_keywords: [비밀번호, 비번, 암호]
      ner_filter_enabled: true           # negative proximity: suppress NER false positives
      ner_extra_stopwords: [사내약어]    # extra common-noun deny-list entries
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .proximity import ProximityConfig

log = logging.getLogger(__name__)

# Deferred import of PinListMutationGuard to avoid circular imports.
# The guard is wired in at runtime inside _try_load() via _get_pin_list_guard().
_PIN_LIST_GUARD = None  # type: Optional[Any]


def _get_pin_list_guard():
    """Lazily import and return the default PinListMutationGuard singleton."""
    global _PIN_LIST_GUARD
    if _PIN_LIST_GUARD is None:
        from .pinlist_guard import DEFAULT_GUARD
        _PIN_LIST_GUARD = DEFAULT_GUARD
    return _PIN_LIST_GUARD


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation constants
# ─────────────────────────────────────────────────────────────────────────────

_VALID_FAIL_MODES = frozenset({"closed", "open"})
_VALID_CONTENT_FAILURE_ACTIONS = frozenset({"block", "warn_allow"})
_VALID_INFRA_FAILURE_ACTIONS = frozenset({"degrade_to_stage1", "block"})
_VALID_STAGE2_FAIL_ACTIONS = frozenset({"block", "mask_known_only", "open"})
_VALID_UNSCANNABLE_ACTIONS = frozenset({"block", "warn_allow", "ocr"})
_VALID_ACTIONS = frozenset({"allow", "mask", "block", "tokenize_roundtrip"})
_VALID_MASK_STYLES = frozenset({"tokenize", "partial", "format_preserving"})


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CategoryPolicy:
    """
    Per-category policy override loaded from the YAML ``categories`` block.

    Any field left as ``None`` means "use the built-in default" from
    ``categories.py``.
    """
    action: Optional[str] = None          # allow | mask | block | tokenize_roundtrip
    mask_style: Optional[str] = None      # tokenize | partial | format_preserving
    min_confidence: Optional[float] = None  # float in [0, 1]
    stage2_fail_action: Optional[str] = None  # block | mask_known_only | open


@dataclass
class AllowlistEntry:
    """A regex pattern (with an optional label) that bypasses PII detection."""
    pattern: str
    label: str = ""
    compiled: Optional[re.Pattern] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.compiled is None:
            # Eagerly compile so callers don't re-compile on every scan
            self.compiled = re.compile(self.pattern)


@dataclass
class PinListEntry:
    """
    A pinned override stored as a keyed hash — never the raw original value.

    The ``hash`` field is an opaque string supplied by the user in the policy
    YAML (e.g. ``sha256:<hex>``).  PII-Guard does NOT store or verify what
    the hash represents — it is used only for change-detection: if the set of
    hashes changes between policy reloads, the change is flagged as requiring
    explicit user approval (``pin_list_approved: true``).
    """
    hash: str       # opaque hash string for the pinned value
    category: str   # which PII category this pin applies to
    action: str     # allow | mask | block | tokenize_roundtrip
    label: str = "" # human-readable description (never contains raw PII)


@dataclass
class ChannelOverride:
    """Per-channel policy overrides (only set fields are applied)."""
    unscannable_action: Optional[str] = None
    stage2_fail_action: Optional[str] = None
    on_content_failure: Optional[str] = None
    fail_mode: Optional[str] = None


@dataclass
class PolicyConfig:
    """
    Typed, in-process policy config object.

    All fields carry secure defaults.  This object is treated as immutable
    after construction — hot-reload creates a new instance; the ``PolicyLoader``
    swaps the reference atomically under a lock.

    Secure defaults guarantee:
      * ``fail_mode = "closed"``      — content failures are blocked
      * ``on_content_failure = "block"`` — unscannable → blocked
      * ``unscannable_action = "block"`` — non-text → blocked
      * No allowlist, no pin-list overrides (nothing whitelisted by default)
    """

    # Source metadata
    source: str = "<built-in defaults>"
    loaded_at: float = field(default_factory=time.monotonic)

    # ── Global failure knobs ─────────────────────────────────────────────────
    fail_mode: str = "closed"
    on_content_failure: str = "block"
    on_infra_failure: str = "degrade_to_stage1"
    stage2_fail_action: str = "mask_known_only"
    unscannable_action: str = "block"

    # ── Rehydration ──────────────────────────────────────────────────────────
    rehydrate: bool = True

    # ── Resource knob ────────────────────────────────────────────────────────
    memory_budget_mb: int = 1024

    # ── Per-category overrides ───────────────────────────────────────────────
    categories: Dict[str, CategoryPolicy] = field(default_factory=dict)

    # ── Project-scoped allowlist ─────────────────────────────────────────────
    allowlist: List[AllowlistEntry] = field(default_factory=list)

    # ── Pin-list ─────────────────────────────────────────────────────────────
    pin_list: List[PinListEntry] = field(default_factory=list)
    # Must be set ``true`` by the user after any pin-list modification.
    # Built-in defaults have an empty pin-list, so approval is vacuously true.
    pin_list_approved: bool = True

    # ── Channel overrides ────────────────────────────────────────────────────
    channel_overrides: Dict[str, ChannelOverride] = field(default_factory=dict)

    # ── Proximity (context-gated detection, R17) ─────────────────────────────
    proximity: ProximityConfig = field(default_factory=ProximityConfig)

    # ── Stage2 NER 백엔드 (R18) ──────────────────────────────────────────────
    # "gliner"(기본) | "spacy". 환경변수 PIIGUARD_NER_BACKEND가 있으면 그쪽이 우선.
    ner_backend: str = "gliner"

    # ── Convenience helpers ──────────────────────────────────────────────────

    def get_category_policy(self, category: str) -> Optional[CategoryPolicy]:
        """Return the per-category override for *category*, or ``None``."""
        return self.categories.get(category)

    def allowlist_patterns(self) -> List[re.Pattern]:
        """Return compiled patterns from the project allowlist."""
        return [e.compiled for e in self.allowlist if e.compiled is not None]

    def channel_setting(self, channel: str, setting: str):
        """
        Return the channel-specific override for *setting* (or ``None``).

        Channels are matched by exact name (e.g. ``"cli"``, ``"ouroboros"``).
        """
        override = self.channel_overrides.get(channel)
        if override is None:
            return None
        return getattr(override, setting, None)


# ─────────────────────────────────────────────────────────────────────────────
# Secure built-in defaults singleton
# ─────────────────────────────────────────────────────────────────────────────

#: Immutable reference to the secure built-in defaults.
#: Used when the policy file is absent or fails validation.
SECURE_DEFAULTS: PolicyConfig = PolicyConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash_pin_list(pin_list: List[PinListEntry]) -> str:
    """Compute a stable content-hash of the pin-list for change-detection."""
    items = sorted((e.hash, e.category, e.action) for e in pin_list)
    return hashlib.sha256(repr(items).encode()).hexdigest()


def _validate_enum(
    data: dict,
    key: str,
    valid_values: frozenset,
) -> Optional[str]:
    """
    Extract and validate a string enum field.

    Returns the value if present and valid.
    Raises ``ValueError`` on a bad value.
    Returns ``None`` if the key is absent.
    """
    if key not in data:
        return None
    v = data[key]
    if not isinstance(v, str) or v not in valid_values:
        raise ValueError(
            f"'{key}' must be one of {sorted(valid_values)}, got {v!r}"
        )
    return v


def _parse_category_policy(
    cat_name: str,
    cat_data: dict,
) -> CategoryPolicy:
    """Parse and validate one entry from the ``categories`` block."""
    if not isinstance(cat_data, dict):
        raise ValueError(
            f"categories.{cat_name} must be a YAML mapping, got {type(cat_data).__name__}"
        )

    action = cat_data.get("action")
    if action is not None and action not in _VALID_ACTIONS:
        raise ValueError(
            f"categories.{cat_name}.action must be one of {sorted(_VALID_ACTIONS)}, "
            f"got {action!r}"
        )

    mask_style = cat_data.get("mask_style")
    if mask_style is not None and mask_style not in _VALID_MASK_STYLES:
        raise ValueError(
            f"categories.{cat_name}.mask_style must be one of "
            f"{sorted(_VALID_MASK_STYLES)}, got {mask_style!r}"
        )

    min_confidence = cat_data.get("min_confidence")
    if min_confidence is not None:
        if not isinstance(min_confidence, (int, float)) or not (0.0 <= float(min_confidence) <= 1.0):
            raise ValueError(
                f"categories.{cat_name}.min_confidence must be a float in [0, 1], "
                f"got {min_confidence!r}"
            )
        min_confidence = float(min_confidence)

    stage2_fail_action = cat_data.get("stage2_fail_action")
    if stage2_fail_action is not None and stage2_fail_action not in _VALID_STAGE2_FAIL_ACTIONS:
        raise ValueError(
            f"categories.{cat_name}.stage2_fail_action must be one of "
            f"{sorted(_VALID_STAGE2_FAIL_ACTIONS)}, got {stage2_fail_action!r}"
        )

    return CategoryPolicy(
        action=action,
        mask_style=mask_style,
        min_confidence=min_confidence,
        stage2_fail_action=stage2_fail_action,
    )


def _parse_allowlist(al_raw: list) -> List[AllowlistEntry]:
    """Parse and validate the ``allowlist`` block."""
    entries: List[AllowlistEntry] = []
    for i, entry in enumerate(al_raw):
        if isinstance(entry, str):
            # Shorthand: bare regex pattern string
            try:
                entries.append(AllowlistEntry(pattern=entry))
            except re.error as exc:
                raise ValueError(
                    f"allowlist[{i}] invalid regex {entry!r}: {exc}"
                ) from exc
        elif isinstance(entry, dict):
            pattern = entry.get("pattern")
            if not isinstance(pattern, str):
                raise ValueError(f"allowlist[{i}].pattern must be a string")
            label = str(entry.get("label", ""))
            try:
                entries.append(AllowlistEntry(pattern=pattern, label=label))
            except re.error as exc:
                raise ValueError(
                    f"allowlist[{i}] invalid regex {pattern!r}: {exc}"
                ) from exc
        else:
            raise ValueError(
                f"allowlist[{i}] must be a string or mapping, "
                f"got {type(entry).__name__}"
            )
    return entries


def _parse_pin_list(pl_raw: list) -> List[PinListEntry]:
    """Parse and validate the ``pin_list`` block."""
    entries: List[PinListEntry] = []
    for i, entry in enumerate(pl_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"pin_list[{i}] must be a YAML mapping, got {type(entry).__name__}"
            )
        h = entry.get("hash")
        cat = entry.get("category")
        action = entry.get("action")
        if not isinstance(h, str) or not h.strip():
            raise ValueError(f"pin_list[{i}].hash must be a non-empty string")
        if not isinstance(cat, str) or not cat.strip():
            raise ValueError(f"pin_list[{i}].category must be a non-empty string")
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f"pin_list[{i}].action must be one of {sorted(_VALID_ACTIONS)}, "
                f"got {action!r}"
            )
        label = str(entry.get("label", ""))
        entries.append(PinListEntry(hash=h.strip(), category=cat.strip(), action=action, label=label))
    return entries


def _parse_proximity(raw) -> ProximityConfig:
    """
    Parse the ``proximity:`` block into a :class:`ProximityConfig`.

    Any omitted key keeps the secure built-in default. Example::

        proximity:
          enabled: true
          window_chars: 25
          account_triggers: [국민, 신한, 입금, 계좌, 토스뱅크]
          biz_triggers: [사업자]
          password_keywords: [비밀번호, 비번, 암호]
          ner_filter_enabled: true
          ner_extra_stopwords: [별칭, 사내용어]
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"'proximity' must be a YAML mapping, got {type(raw).__name__}"
        )
    d = ProximityConfig()  # defaults

    def _bool(key, cur):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise ValueError(f"proximity.{key} must be true/false")
            return raw[key]
        return cur

    def _strs(key, cur):
        if key in raw:
            v = raw[key]
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f"proximity.{key} must be a list of strings")
            return tuple(v)
        return cur

    window = d.window_chars
    if "window_chars" in raw:
        w = raw["window_chars"]
        if not isinstance(w, int) or isinstance(w, bool) or not (1 <= w <= 200):
            raise ValueError("proximity.window_chars must be an int in [1, 200]")
        window = w

    return ProximityConfig(
        enabled=_bool("enabled", d.enabled),
        window_chars=window,
        account_triggers=_strs("account_triggers", d.account_triggers),
        biz_triggers=_strs("biz_triggers", d.biz_triggers),
        password_keywords=_strs("password_keywords", d.password_keywords),
        ner_filter_enabled=_bool("ner_filter_enabled", d.ner_filter_enabled),
        ner_extra_stopwords=_strs("ner_extra_stopwords", d.ner_extra_stopwords),
    )


#: Stage2 NER 백엔드로 허용되는 값(R18)
_VALID_NER_BACKENDS = {"gliner", "spacy"}


def _parse_stage2_backend(raw) -> str:
    """
    ``stage2:`` 블록에서 ``ner_backend``를 파싱해 반환.

    예시 YAML::

        stage2:
          ner_backend: gliner   # gliner(기본) | spacy(경량 폴백)

    - ``stage2`` 자체는 매핑이어야 하고, ``ner_backend`` 키가 없으면 기본값 "gliner".
    - 알 수 없는 값이면 ValueError(침묵 폴백 금지 — P3).
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"'stage2' must be a YAML mapping, got {type(raw).__name__}"
        )
    backend = raw.get("ner_backend", "gliner")
    backend = str(backend).strip().lower()
    if backend not in _VALID_NER_BACKENDS:
        raise ValueError(
            f"stage2.ner_backend must be one of {sorted(_VALID_NER_BACKENDS)}, "
            f"got {raw.get('ner_backend')!r}"
        )
    return backend


def _parse_channel_overrides(co_raw: dict) -> Dict[str, ChannelOverride]:
    """Parse and validate the ``channel_overrides`` block."""
    overrides: Dict[str, ChannelOverride] = {}
    for channel_name, raw in co_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"channel_overrides.{channel_name} must be a YAML mapping, "
                f"got {type(raw).__name__}"
            )
        co = ChannelOverride()
        for key, valid, attr in [
            ("unscannable_action", _VALID_UNSCANNABLE_ACTIONS, "unscannable_action"),
            ("stage2_fail_action", _VALID_STAGE2_FAIL_ACTIONS, "stage2_fail_action"),
            ("on_content_failure", _VALID_CONTENT_FAILURE_ACTIONS, "on_content_failure"),
            ("fail_mode", _VALID_FAIL_MODES, "fail_mode"),
        ]:
            v = raw.get(key)
            if v is not None:
                if v not in valid:
                    raise ValueError(
                        f"channel_overrides.{channel_name}.{key} must be one of "
                        f"{sorted(valid)}, got {v!r}"
                    )
                setattr(co, attr, v)
        overrides[channel_name] = co
    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Core parse + validate
# ─────────────────────────────────────────────────────────────────────────────

def _parse_and_validate(
    raw_yaml: str,
    source: str,
) -> Tuple[PolicyConfig, List[str]]:
    """
    Parse *raw_yaml* and return ``(PolicyConfig, warnings)``.

    Raises ``ValueError`` for structural/schema errors.
    Warnings are returned for non-fatal issues (e.g. unknown top-level keys).
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for policy loading. "
            "Install with: pip install pyyaml"
        ) from exc

    try:
        data = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc

    if data is None:
        # Empty file → treat as explicit empty policy (secure defaults apply)
        data = {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Policy file must be a YAML mapping at the top level, "
            f"got {type(data).__name__}"
        )

    warnings: List[str] = []
    config = PolicyConfig(source=source, loaded_at=time.monotonic())

    # ── Global failure knobs ─────────────────────────────────────────────────
    v = _validate_enum(data, "fail_mode", _VALID_FAIL_MODES)
    if v is not None:
        config.fail_mode = v

    v = _validate_enum(data, "on_content_failure", _VALID_CONTENT_FAILURE_ACTIONS)
    if v is not None:
        config.on_content_failure = v

    v = _validate_enum(data, "on_infra_failure", _VALID_INFRA_FAILURE_ACTIONS)
    if v is not None:
        config.on_infra_failure = v

    v = _validate_enum(data, "stage2_fail_action", _VALID_STAGE2_FAIL_ACTIONS)
    if v is not None:
        config.stage2_fail_action = v

    v = _validate_enum(data, "unscannable_action", _VALID_UNSCANNABLE_ACTIONS)
    if v is not None:
        config.unscannable_action = v

    # ── Rehydrate ────────────────────────────────────────────────────────────
    if "rehydrate" in data:
        rv = data["rehydrate"]
        if not isinstance(rv, bool):
            raise ValueError(
                f"'rehydrate' must be a boolean (true/false), got {type(rv).__name__!r}"
            )
        config.rehydrate = rv

    # ── Memory budget ────────────────────────────────────────────────────────
    if "memory_budget_mb" in data:
        mb = data["memory_budget_mb"]
        if not isinstance(mb, int) or isinstance(mb, bool) or mb < 128:
            raise ValueError(
                f"'memory_budget_mb' must be an integer ≥ 128, got {mb!r}"
            )
        config.memory_budget_mb = mb

    # ── Proximity (context-gated detection, R17) ─────────────────────────────
    if "proximity" in data:
        config.proximity = _parse_proximity(data["proximity"])

    # ── Stage2 NER 백엔드 선택 (R18) ─────────────────────────────────────────
    if "stage2" in data:
        config.ner_backend = _parse_stage2_backend(data["stage2"])

    # ── Per-category overrides ───────────────────────────────────────────────
    if "categories" in data:
        cats_raw = data["categories"]
        if not isinstance(cats_raw, dict):
            raise ValueError(
                f"'categories' must be a YAML mapping, got {type(cats_raw).__name__}"
            )
        for cat_name, cat_data in cats_raw.items():
            config.categories[str(cat_name)] = _parse_category_policy(
                str(cat_name), cat_data
            )

    # ── Allowlist ────────────────────────────────────────────────────────────
    if "allowlist" in data:
        al_raw = data["allowlist"]
        if not isinstance(al_raw, list):
            raise ValueError(
                f"'allowlist' must be a YAML list, got {type(al_raw).__name__}"
            )
        config.allowlist = _parse_allowlist(al_raw)

    # ── Pin-list ─────────────────────────────────────────────────────────────
    if "pin_list" in data:
        pl_raw = data["pin_list"]
        if not isinstance(pl_raw, list):
            raise ValueError(
                f"'pin_list' must be a YAML list, got {type(pl_raw).__name__}"
            )
        config.pin_list = _parse_pin_list(pl_raw)

    if "pin_list_approved" in data:
        pla = data["pin_list_approved"]
        if not isinstance(pla, bool):
            raise ValueError(
                f"'pin_list_approved' must be a boolean, got {type(pla).__name__!r}"
            )
        config.pin_list_approved = pla

    # ── Channel overrides ────────────────────────────────────────────────────
    if "channel_overrides" in data:
        co_raw = data["channel_overrides"]
        if not isinstance(co_raw, dict):
            raise ValueError(
                f"'channel_overrides' must be a YAML mapping, "
                f"got {type(co_raw).__name__}"
            )
        config.channel_overrides = _parse_channel_overrides(co_raw)

    # ── Warn on unknown top-level keys ───────────────────────────────────────
    _KNOWN_KEYS = frozenset({
        "version",
        "fail_mode", "on_content_failure", "on_infra_failure",
        "stage2_fail_action", "unscannable_action",
        "rehydrate", "memory_budget_mb",
        "categories", "allowlist", "pin_list", "pin_list_approved",
        "channel_overrides",
    })
    unknown = set(data.keys()) - _KNOWN_KEYS
    for uk in sorted(unknown):
        warnings.append(f"unknown top-level key '{uk}' — ignored")

    return config, warnings


# ─────────────────────────────────────────────────────────────────────────────
# PolicyLoader
# ─────────────────────────────────────────────────────────────────────────────

class PolicyLoader:
    """
    Loads, validates, and hot-reloads the PII-Guard policy YAML.

    Thread-safety
    -------------
    ``config`` property reads and ``reload_if_changed()`` calls are protected
    by an internal ``RLock``.  Multiple concurrent readers are safe.

    Hot-reload semantics
    --------------------
    * File deleted → revert to ``SECURE_DEFAULTS`` immediately.
    * File present, YAML invalid / schema error → retain last-valid config.
    * File present, valid → swap atomically.

    Pin-list approval
    -----------------
    When the pin-list changes vs. the previously loaded snapshot, the new
    entries are rejected unless ``pin_list_approved: true`` is set in the same
    file.  The old pin-list from the last-valid config is retained, and a
    warning is logged.  This ensures pin-list modifications require explicit
    user action outside the agent's write scope.

    Parameters
    ----------
    policy_path:
        Path to the policy YAML file.  If ``None`` or the file does not exist
        at construction time, ``SECURE_DEFAULTS`` are used.
    """

    def __init__(self, policy_path: Optional[str] = None) -> None:
        self._path: Optional[Path] = (
            Path(policy_path) if policy_path else None
        )
        self._lock = threading.RLock()
        self._current: PolicyConfig = SECURE_DEFAULTS
        self._last_mtime: Optional[float] = None
        self._last_pin_list_hash: str = _hash_pin_list([])

        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Initial load (only if a path was given)
        if self._path is not None:
            self._try_load()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def config(self) -> PolicyConfig:
        """
        Return the current (possibly hot-reloaded) :class:`PolicyConfig`.

        Thread-safe; never returns ``None``.
        """
        with self._lock:
            return self._current

    def reload_if_changed(self) -> bool:
        """
        Check whether the policy file has changed (by ``mtime``) and reload.

        Returns
        -------
        bool
            ``True`` if the config was successfully reloaded (including
            reverting to ``SECURE_DEFAULTS`` on deletion).
            ``False`` if no change was detected or reload failed (last-valid
            config retained on failure).
        """
        if self._path is None:
            return False

        # ── Check for file deletion ──────────────────────────────────────────
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            with self._lock:
                if self._current is not SECURE_DEFAULTS:
                    log.warning(
                        "PII-Guard: policy file %s was deleted — "
                        "reverting to secure built-in defaults",
                        self._path,
                    )
                    self._current = SECURE_DEFAULTS
                    self._last_mtime = None
                    return True
            return False
        except OSError as exc:
            log.warning("PII-Guard: cannot stat policy file %s: %s", self._path, exc)
            return False

        # ── Check whether mtime changed ──────────────────────────────────────
        with self._lock:
            if mtime == self._last_mtime:
                return False

        return self._try_load()

    def start_watcher(
        self,
        interval: float = 1.0,
        debounce: float = 0.2,
    ) -> None:
        """
        Start a background daemon thread that watches the policy file for changes
        and atomically swaps the live config object without process restart.

        Change detection
        ----------------
        The watcher polls the file's ``mtime`` every *interval* seconds (or more
        frequently when a change is pending — see below).  When a change is
        detected, a debounce timer starts.  The reload fires only after the file
        has been **stable** for *debounce* seconds with no further mtime changes.
        This prevents partial-file reads during editor save patterns (write-to-tmp
        + rename, or truncate-and-rewrite) and coalesces rapid successive writes
        into a single reload.

        Error / failure guard
        ---------------------
        If the reloaded file is YAML-invalid or fails schema validation, the
        **last-valid config is retained** and the error is logged.  The watcher
        continues running and picks up the next valid write.

        Parameters
        ----------
        interval:
            Polling interval in seconds (default 1.0).  The watcher polls for
            mtime changes at ``min(interval, max(0.05, debounce / 3))`` granularity
            when debounce is enabled so that the settling window is measured
            accurately.
        debounce:
            Settling time in seconds after the **last** detected change before the
            reload is triggered (default 0.2).  Set to ``0`` to disable debouncing
            and reload immediately on any detected mtime change (matches the legacy
            behaviour).

        Notes
        -----
        Calling this method when the watcher is already running is a no-op.
        Call :meth:`stop_watcher` to terminate the thread.
        """
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            return
        self._stop_event.clear()

        # Inner poll tick: fine-grained when debounce is active so the settling
        # window fires on time; falls back to `interval` when debounce is off.
        _tick: float = (
            interval
            if debounce <= 0
            else min(interval, max(0.05, debounce / 3))
        )

        def _loop() -> None:
            _pending: bool = False          # True = change detected, debounce running
            _pending_since: float = 0.0    # monotonic time of this debounce window
            _observed_mtime: Optional[float] = None  # mtime that started _pending

            while not self._stop_event.wait(_tick):
                try:
                    current_mtime = self._get_file_mtime()

                    with self._lock:
                        known_mtime = self._last_mtime

                    # Detect whether the file has changed vs. what the loader knows.
                    # None == None means "still absent" → no change.
                    mtime_changed = current_mtime != known_mtime

                    if mtime_changed:
                        if not _pending:
                            # First detection: start the debounce clock.
                            _pending = True
                            _pending_since = time.monotonic()
                            _observed_mtime = current_mtime
                        elif current_mtime != _observed_mtime:
                            # The file changed *again* while we were debouncing
                            # (e.g. a second editor save).  Reset the clock so
                            # the full settling window runs from this new write.
                            _pending_since = time.monotonic()
                            _observed_mtime = current_mtime
                        # else: same mtime as when _pending was set — debounce
                        # clock keeps ticking; do NOT reset it.

                    if _pending:
                        elapsed = time.monotonic() - _pending_since
                        if debounce <= 0 or elapsed >= debounce:
                            self.reload_if_changed()
                            _pending = False
                            _observed_mtime = None
                except Exception as exc:  # pragma: no cover
                    log.warning(
                        "PII-Guard policy watcher: unexpected error: %s", exc
                    )

        self._watcher_thread = threading.Thread(
            target=_loop,
            daemon=True,
            name="pii-guard-policy-watcher",
        )
        self._watcher_thread.start()

    def stop_watcher(self) -> None:
        """Stop the background file-watcher thread (blocks until it exits)."""
        self._stop_event.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=5)
            self._watcher_thread = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_file_mtime(self) -> Optional[float]:
        """
        Return the current mtime of the policy file, or ``None`` if the file
        is absent or unreadable.

        Used by the watcher loop to separate *change detection* from *reload*
        so that debouncing can interpose between the two.
        """
        if self._path is None:
            return None
        try:
            return self._path.stat().st_mtime
        except (FileNotFoundError, OSError):
            return None

    def _try_load(self) -> bool:
        """
        Read, parse, and validate the policy file, then swap the current
        config atomically.

        On file-not-found → reverts to ``SECURE_DEFAULTS``.
        On read / parse / validation error → retains last-valid config.

        Returns ``True`` if the config was updated, ``False`` otherwise.
        """
        assert self._path is not None

        # ── Read ─────────────────────────────────────────────────────────────
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            with self._lock:
                if self._current is not SECURE_DEFAULTS:
                    log.warning(
                        "PII-Guard: policy file %s not found — "
                        "using secure built-in defaults",
                        self._path,
                    )
                    self._current = SECURE_DEFAULTS
                    self._last_mtime = None
            return True
        except OSError as exc:
            log.error(
                "PII-Guard: cannot read policy file %s: %s — "
                "retaining last-valid config",
                self._path, exc,
            )
            return False

        # ── Parse + validate ─────────────────────────────────────────────────
        try:
            new_config, validation_warnings = _parse_and_validate(raw, str(self._path))
        except (ValueError, ImportError) as exc:
            log.error(
                "PII-Guard: policy file %s is invalid — "
                "retaining last-valid config. Error: %s",
                self._path, exc,
            )
            return False

        for w in validation_warnings:
            log.warning("PII-Guard policy (%s): %s", self._path, w)

        # ── Pin-list change guard (Sub-AC 5d-i) ──────────────────────────────
        # Use PinListMutationGuard to classify and evaluate the change.
        # File-system changes are always OUT_OF_BAND; they are allowed only
        # when the user has set pin_list_approved: true in the same file.
        new_pin_hash = _hash_pin_list(new_config.pin_list)
        with self._lock:
            old_pin_hash = self._last_pin_list_hash
            if old_pin_hash and new_pin_hash != old_pin_hash:
                # Pin-list changed relative to the last approved snapshot.
                # Classify as OUT_OF_BAND (file-system edit) and check approval.
                from .pinlist_guard import MutationSource
                guard = _get_pin_list_guard()
                mutation_result = guard.check(
                    source=MutationSource.OUT_OF_BAND,
                    approved=new_config.pin_list_approved,
                )
                if not mutation_result.allowed:
                    log.warning(
                        "PII-Guard: pin_list in %s blocked by mutation guard "
                        "(%s: %s). The old pin-list is retained.",
                        self._path,
                        mutation_result.error_type,
                        mutation_result.error_message,
                    )
                    # Retain old pin-list; update everything else
                    new_config.pin_list = list(self._current.pin_list)
                    new_config.pin_list_approved = False
                    # Recompute hash with the retained pin-list
                    new_pin_hash = old_pin_hash
                else:
                    log.info(
                        "PII-Guard: pin_list in %s updated (user-approved, "
                        "source=out_of_band).",
                        self._path,
                    )

            # Resolve mtime after a successful parse (may differ from the
            # initial stat if the file was replaced while we were reading)
            try:
                mtime = self._path.stat().st_mtime
            except OSError:
                mtime = None

            self._current = new_config
            self._last_mtime = mtime
            self._last_pin_list_hash = new_pin_hash

        log.info("PII-Guard: policy loaded from %s", self._path)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience API
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(path: Optional[str] = None) -> PolicyConfig:
    """
    Load the policy from *path* (or return ``SECURE_DEFAULTS`` if absent).

    This is a convenience one-shot function for callers that don't need
    hot-reload semantics.  For live reload, use :class:`PolicyLoader` directly.

    Parameters
    ----------
    path:
        Path to the policy YAML file, or ``None`` to get ``SECURE_DEFAULTS``.

    Returns
    -------
    PolicyConfig
        Never raises — on error returns ``SECURE_DEFAULTS``.
    """
    if path is None:
        return SECURE_DEFAULTS
    loader = PolicyLoader(policy_path=path)
    return loader.config
