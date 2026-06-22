"""
PII-Guard — local-first PII/secret detection and redaction engine.

Quick start::

    from pii_guard import Engine

    engine = Engine()
    result = engine.scan("Contact bob@example.com or call 010-1234-5678")
    print(result.redacted_text)
    # "Contact [EMAIL_1] or call [PHONE_1]"

Process-launch wrapper (Sub-AC 6a)::

    from pii_guard import ProcessLauncher

    launcher = ProcessLauncher()          # reads PIIGUARD_HOST / PIIGUARD_PORT
    launcher.run(["codex", "--prompt", "summarise this"])
    # child process sees ANTHROPIC_BASE_URL / OPENAI_BASE_URL / GEMINI_BASE_URL
    # all pointing to the local PII-Guard proxy
"""
from .boundary import BoundaryItem, BoundaryReport, EnforcementTier, get_protection_boundary, print_boundary_report
from .decision import FailureDecision, PolicyDecision, PolicyDecisionEngine
from .engine import Engine
from .launcher import ProcessLauncher, build_proxy_env, ALL_PROXY_ENV_VARS, PROVIDER_ENV_VARS
from .ledger import Ledger, LedgerEventType
from .masker import maskPayload
from .models import Action, CategoryClass, Detection, DetectionStage, MaskStyle, RedactionResult
from .policy import (
    AllowlistEntry,
    CategoryPolicy,
    ChannelOverride,
    PinListEntry,
    PolicyConfig,
    PolicyLoader,
    SECURE_DEFAULTS,
    load_policy,
)
from .proxy import PIIGuardProxy
from .response_rehydrator import RehydrationResult, ResponsePostProcessor
from .session_map import SessionMap
from .streaming_buffer import StreamingLookAheadBuffer
from .vault import RequestVault, apply_mask_style, mask_payload_with_vault
from .tripwire import TripwireHit, TripwireResult, sweep_raw_body
from .updater import (
    ManifestEntry,
    UpdateManifest,
    UpdateRejectedError,
    UpdateSigner,
    UpdateVerifier,
    generate_update_key,
    make_manifest,
)
from .pinlist_guard import (
    AGENT_MUTATION_BLOCKED,
    CONTROL_PIN_LIST_PATH,
    MutationResult,
    MutationSource,
    PIN_LIST_NOT_APPROVED,
    PinListMutationGuard,
    classify_source,
)
from .pinlist_approval import (
    ApprovalResult,
    GATE_COMMITTED,
    GATE_IDLE,
    GATE_REJECTED,
    GATE_STAGED,
    PinListApprovalGate,
    run_interactive_approval,
)

__all__ = [
    # Pin-list mutation guard (Sub-AC 5d-i)
    "AGENT_MUTATION_BLOCKED",
    "CONTROL_PIN_LIST_PATH",
    "MutationResult",
    "MutationSource",
    "PIN_LIST_NOT_APPROVED",
    "PinListMutationGuard",
    "classify_source",
    # Pin-list approval flow (Sub-AC 5d-ii)
    "ApprovalResult",
    "GATE_COMMITTED",
    "GATE_IDLE",
    "GATE_REJECTED",
    "GATE_STAGED",
    "PinListApprovalGate",
    "run_interactive_approval",
    # Core detection
    "Engine",
    "SessionMap",
    "Action",
    "CategoryClass",
    "Detection",
    "DetectionStage",
    "MaskStyle",
    "RedactionResult",
    # Pure masking function (Sub-AC 2b-i)
    "maskPayload",
    # Intercepting proxy (Sub-AC 2b-ii)
    "PIIGuardProxy",
    # Response rehydration post-processor (Sub-AC 2c)
    "ResponsePostProcessor",
    "RehydrationResult",
    # Streaming SSE look-ahead buffer (Sub-AC 9.1)
    "StreamingLookAheadBuffer",
    # Audit Ledger (Sub-AC 4.1)
    "Ledger",
    "LedgerEventType",
    # Process launcher (Sub-AC 6a)
    "ProcessLauncher",
    "build_proxy_env",
    "ALL_PROXY_ENV_VARS",
    "PROVIDER_ENV_VARS",
    # Protection-boundary declaration (Sub-AC 6c)
    "BoundaryItem",
    "BoundaryReport",
    "EnforcementTier",
    "get_protection_boundary",
    "print_boundary_report",
    # Policy config loader (Sub-AC 5a)
    "AllowlistEntry",
    "CategoryPolicy",
    "ChannelOverride",
    "PinListEntry",
    "PolicyConfig",
    "PolicyLoader",
    "SECURE_DEFAULTS",
    "load_policy",
    # Policy decision engine (Sub-AC 5c-i)
    "FailureDecision",
    "PolicyDecision",
    "PolicyDecisionEngine",
    # Request-scoped masking vault (Sub-AC 5c-ii)
    "RequestVault",
    "apply_mask_style",
    "mask_payload_with_vault",
    # Full-body tripwire sweep (Sub-AC 8.2)
    "TripwireHit",
    "TripwireResult",
    "sweep_raw_body",
    # Signed-channel update mechanism (Sub-AC 7.1)
    "ManifestEntry",
    "UpdateManifest",
    "UpdateRejectedError",
    "UpdateSigner",
    "UpdateVerifier",
    "generate_update_key",
    "make_manifest",
]

__version__ = "0.1.0"
