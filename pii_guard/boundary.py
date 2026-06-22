"""
PII-Guard protection-boundary declaration module.

Sub-AC 6c — Protection-boundary declaration
-------------------------------------------
Returns a structured report that enumerates, with no false assurance:

  - What IS defended: which process classes and traffic paths flow through
    the PII-Guard proxy and are therefore subject to PII/secret detection.

  - What is NOT defended: process classes, ports, protocols, and actor
    postures that are outside the current protection boundary and will
    NOT be intercepted.

  - Bypass paths: concrete ways the protection can be circumvented in the
    current enforcement tier.

  - Threat-actor model: the assumed actor posture (trusted_but_compromisable)
    and what is explicitly out of scope.

Two enforcement tiers are reported:

    ``cooperative_gateway``  (default-on)
        Base-URL env-var injection via ProcessLauncher.  Only processes that
        honour ANTHROPIC_BASE_URL / OPENAI_BASE_URL / GEMINI_BASE_URL are
        routed through the proxy.  A process that ignores env vars, has a
        hard-coded base URL, or is spawned outside the ouroboros launcher
        context is NOT protected.

    ``egress_lockdown``  (opt-in, requires root + macOS pf(4))
        pf(4) firewall anchor blocks direct outbound TCP to known LLM-provider
        CIDR ranges on ports 80 and 443.  Processes that ignore env vars are
        still forced through the proxy because the OS drops their direct
        connections.  Root-level actors and non-standard ports remain outside
        the boundary.

The module exposes:

    get_protection_boundary(enforcement_tier, proxy_url) -> BoundaryReport
        Return the structured report.

    BoundaryReport
        Dataclass with .defended / .undefended / .bypass_paths / .as_dict()

    print_boundary_report(report, stream)
        Human-readable rendering (used by the CLI ``boundary`` command).

CLI surface (added to cli.py)::

    piiguard boundary [--mode cooperative_gateway|egress_lockdown]
                      [--proxy-url http://127.0.0.1:4444]
                      [--json]
"""
from __future__ import annotations

import datetime
import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, TextIO


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class EnforcementTier(str, Enum):
    """Active enforcement tier for the PII-Guard proxy."""

    COOPERATIVE_GATEWAY = "cooperative_gateway"
    """
    Default-on tier: proxy env-var injection only.  Processes that honour the
    injected base-URL env vars are intercepted; all others are unprotected.
    """

    EGRESS_LOCKDOWN = "egress_lockdown"
    """
    Opt-in tier: pf(4) firewall anchor drops direct TCP to LLM provider CIDRs.
    Requires root / sudo.  Non-standard ports and root actors remain outside
    the boundary.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundaryItem:
    """
    A single defended or undefended scope element.

    Attributes
    ----------
    description:
        One-line summary of what is (or isn't) covered.
    detail:
        Longer explanation of the mechanism or limitation.
    category:
        Broad grouping: ``"process"``, ``"port"``, ``"protocol"``,
        ``"actor"``, or ``"coverage"``.
    """

    description: str
    detail: str
    category: str


@dataclass
class BoundaryReport:
    """
    Structured protection-boundary report.

    This is the primary output of :func:`get_protection_boundary`.

    Attributes
    ----------
    enforcement_tier:
        Active tier (``"cooperative_gateway"`` or ``"egress_lockdown"``).
    proxy_url:
        Base URL of the running PII-Guard proxy (e.g. ``http://127.0.0.1:4444``).
    defended:
        Scope elements that ARE intercepted and scrubbed.
    undefended:
        Scope elements that are NOT intercepted — explicitly declared to
        avoid false assurance.
    bypass_paths:
        Concrete ways the protection can be circumvented in the current tier.
    threat_actor_model:
        Assumed actor posture and what is explicitly out of scope.
    assurance_statement:
        Non-false-assurance declaration — what PII-Guard guarantees and what
        it does NOT guarantee in the current configuration.
    generated_at:
        ISO-8601 UTC timestamp when the report was generated.
    """

    enforcement_tier: str
    proxy_url: str
    defended: List[BoundaryItem] = field(default_factory=list)
    undefended: List[BoundaryItem] = field(default_factory=list)
    bypass_paths: List[str] = field(default_factory=list)
    threat_actor_model: str = ""
    assurance_statement: str = ""
    generated_at: str = ""

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict of the report."""
        return asdict(self)

    def as_json(self, indent: int = 2) -> str:
        """Return the report as a formatted JSON string."""
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

#: Items that are DEFENDED in the cooperative-gateway tier (env-var injection).
_DEFENDED_COOPERATIVE: List[BoundaryItem] = [
    BoundaryItem(
        description="ouroboros-spawned processes via ProcessLauncher",
        detail=(
            "Processes launched through pii_guard.ProcessLauncher.run() / "
            "popen() / call() inherit ANTHROPIC_BASE_URL, OPENAI_BASE_URL, "
            "OPENAI_API_BASE, GEMINI_BASE_URL, and GOOGLE_GENAI_BASE_URL all "
            "pointing to the local PII-Guard proxy.  Their outbound LLM "
            "requests pass through the proxy and are subject to full PII/secret "
            "detection before reaching the upstream LLM endpoint."
        ),
        category="process",
    ),
    BoundaryItem(
        description="LLM CLI tools that honour SDK base-URL env vars",
        detail=(
            "Any tool that reads ANTHROPIC_BASE_URL (Anthropic Python SDK ≥0.20), "
            "OPENAI_BASE_URL (OpenAI SDK v1+), or OPENAI_API_BASE (SDK <v1 / "
            "LangChain / LiteLLM) will route its traffic through the proxy when "
            "launched inside the env_patch() context manager."
        ),
        category="process",
    ),
    BoundaryItem(
        description="ouroboros workflow agent requests",
        detail=(
            "Ouroboros workflow agents spawned by the framework inherit the "
            "patched environment and therefore send all LLM API calls through "
            "the proxy.  System prompts, tool_use arguments, tool_result content, "
            "and document blocks are all scrubbed before forwarding."
        ),
        category="process",
    ),
    BoundaryItem(
        description="Claude, OpenAI, and Gemini request payloads",
        detail=(
            "Structured parsers for the Anthropic Messages API, OpenAI "
            "chat-completions API, and Google Gemini generateContent API "
            "cover all known PII-bearing fields: message content, system "
            "prompts, tool_use_input, tool_result, and document blocks."
        ),
        category="protocol",
    ),
]

#: Items that are NOT defended in the cooperative-gateway tier.
_UNDEFENDED_COOPERATIVE: List[BoundaryItem] = [
    BoundaryItem(
        description="processes spawned outside ouroboros or without ProcessLauncher",
        detail=(
            "Any subprocess that is NOT launched via ProcessLauncher or inside "
            "the env_patch() context does NOT inherit the proxy env vars.  It "
            "will connect directly to the upstream LLM endpoint, bypassing the "
            "proxy entirely.  No PII detection occurs for that traffic."
        ),
        category="process",
    ),
    BoundaryItem(
        description="LLM clients with hard-coded base URLs",
        detail=(
            "Clients that ignore SDK base-URL env vars and instead embed a "
            "hard-coded 'https://api.anthropic.com' (or similar) in their code "
            "will bypass the proxy even inside a ProcessLauncher context.  The "
            "cooperative-gateway tier cannot override hard-coded connection targets."
        ),
        category="process",
    ),
    BoundaryItem(
        description="non-standard ports (not 80 or 443)",
        detail=(
            "Traffic on ports other than 80 and 443 is not intercepted.  If an "
            "LLM provider or custom endpoint uses a non-standard port, the proxy "
            "will not see that traffic in the cooperative-gateway tier."
        ),
        category="port",
    ),
    BoundaryItem(
        description="LLM providers not covered by the three default provider formats",
        detail=(
            "Only Anthropic Messages API, OpenAI chat-completions, and Google "
            "Gemini generateContent are actively parsed and scrubbed.  Traffic "
            "to other LLM providers passes through the proxy unmodified if it "
            "reaches an unrecognised path (unknown_field_action applies, default: "
            "block in strict mode)."
        ),
        category="protocol",
    ),
    BoundaryItem(
        description="root-level and kernel-level actors",
        detail=(
            "The threat model explicitly excludes agents with root / sudo / kernel "
            "privileges.  Such an actor can read the proxy's memory, overwrite env "
            "vars, install kernel modules, or use raw sockets to bypass any "
            "userspace protection.  Containment of root-level agents requires a "
            "hardware VM or sandbox boundary which is outside the scope of this tool."
        ),
        category="actor",
    ),
    BoundaryItem(
        description="system-level TLS traffic (no MITM certificate installed)",
        detail=(
            "PII-Guard does NOT install a local root CA or perform TLS "
            "man-in-the-middle inspection.  Interception relies entirely on "
            "base-URL redirection to the local HTTP proxy.  Traffic that does "
            "not go through the local proxy is not inspected."
        ),
        category="protocol",
    ),
    BoundaryItem(
        description="inbound LLM responses on terminal output (terminal_restore=OFF)",
        detail=(
            "By default, terminal_restore is disabled.  The terminal therefore "
            "displays [CATEGORY_N] placeholder tokens rather than restored real "
            "values.  Agents receive rehydrated responses (real values restored "
            "locally before being returned), but the user's terminal does NOT "
            "show the original PII."
        ),
        category="coverage",
    ),
]

#: Additional items that become DEFENDED when egress_lockdown is active.
_DEFENDED_EGRESS_LOCKDOWN_EXTRA: List[BoundaryItem] = [
    BoundaryItem(
        description="direct outbound TCP to known LLM provider CIDR ranges (ports 80/443)",
        detail=(
            "A pf(4) firewall anchor blocks outbound TCP connections from any "
            "process on the host to the known Anthropic, OpenAI, and Google LLM "
            "provider IP ranges on ports 80 and 443.  Even processes that ignore "
            "env vars cannot establish a direct TLS session to provider APIs — "
            "their SYN packets are silently dropped by the OS."
        ),
        category="port",
    ),
    BoundaryItem(
        description="LLM CLI tools that bypass SDK env-var hooks",
        detail=(
            "Tools that hard-code 'https://api.anthropic.com' or similar and "
            "therefore ignore env-var redirection are still blocked at the "
            "network layer by the pf(4) anchor.  They cannot reach the real "
            "endpoint directly; they must route via the local proxy or fail."
        ),
        category="process",
    ),
]

#: Items that remain UNDEFENDED even with egress_lockdown active.
_UNDEFENDED_EGRESS_LOCKDOWN_ONLY: List[BoundaryItem] = [
    BoundaryItem(
        description="non-standard ports (not 80 or 443)",
        detail=(
            "The pf(4) anchor blocks ports 80 and 443 only.  Traffic to "
            "LLM APIs on other ports (e.g. 8443, 8080) is NOT blocked by the "
            "firewall anchor and is NOT intercepted by the proxy."
        ),
        category="port",
    ),
    BoundaryItem(
        description="LLM provider IP ranges not in the static CIDR table",
        detail=(
            "The blocked CIDR table is a best-effort static list.  If a provider "
            "uses new IP ranges, CDN edge nodes outside the listed CIDRs, or "
            "non-standard IP allocations, direct connections to those IPs will "
            "NOT be blocked by the anchor.  The table must be refreshed manually."
        ),
        category="coverage",
    ),
    BoundaryItem(
        description="root-level and kernel-level actors",
        detail=(
            "pf(4) itself runs in the kernel and can be flushed by any process "
            "with root privileges.  A compromised agent that obtains root can "
            "disable the pf anchor and then connect directly to provider APIs.  "
            "This is explicitly outside the threat model."
        ),
        category="actor",
    ),
    BoundaryItem(
        description="the PII-Guard proxy process itself",
        detail=(
            "The proxy must be permitted to reach upstream provider endpoints.  "
            "On macOS, loopback traffic (127.0.0.1) is exempt from pf(4) by "
            "default, so the proxy's forwarding connections are unaffected.  "
            "However, if the proxy is compromised, it can forward unmasked content."
        ),
        category="process",
    ),
    BoundaryItem(
        description="system-level TLS traffic (no MITM certificate installed)",
        detail=(
            "pf(4) egress-lockdown forces traffic through the proxy by blocking "
            "direct connections, but the proxy still terminates TLS at the "
            "provider endpoint using standard HTTPS.  No local CA is installed; "
            "the interception boundary is the HTTP layer, not TLS inspection."
        ),
        category="protocol",
    ),
]

#: Bypass paths for cooperative_gateway tier.
_BYPASS_PATHS_COOPERATIVE: List[str] = [
    "Hard-coded base URL in client code (ignores SDK env-var injection)",
    "Process spawned outside ouroboros ProcessLauncher without env_patch()",
    "Non-standard LLM API port (not 80 or 443)",
    "Root-level actor overwriting env vars or using raw sockets",
    "Kernel-level actor (out of scope; requires VM/sandbox boundary)",
    "LLM provider not covered by Claude/OpenAI/Gemini parsers sending to unrecognised path",
    "HTTP clients that use their own DNS resolution and ignore SDK conventions",
]

#: Bypass paths for egress_lockdown tier (subset — harder to bypass).
_BYPASS_PATHS_EGRESS_LOCKDOWN: List[str] = [
    "LLM API endpoint on non-standard port (not 80 or 443)",
    "LLM provider IP outside static CIDR table (CDN drift / new allocation)",
    "Root-level actor flushing the pf(4) anchor (sudo pfctl -F all)",
    "Kernel-level actor loading a custom kernel extension (out of scope; requires VM/sandbox)",
    "The PII-Guard proxy process itself (loopback traffic is pf-exempt on macOS)",
    "IPv6 traffic if the pf anchor only covers IPv4 ranges",
]

#: Threat actor model description (applies to both tiers).
_THREAT_ACTOR_MODEL: str = (
    "Assumed actor posture: trusted_but_compromisable.  PII-Guard protects "
    "against a normally-behaving agent or LLM CLI that has been injected with "
    "a malicious prompt or otherwise compromised at the application level.  It "
    "does NOT protect against:\n"
    "  • Agents with root / sudo / kernel privileges\n"
    "  • Attackers with physical access to the machine\n"
    "  • Compromised hardware (firmware, hypervisor, or kernel)\n"
    "  • Side-channel or memory-inspection attacks\n"
    "These threat scenarios require a hardware VM, secure enclave, or OS-level "
    "sandboxing boundary which is outside the scope of this tool."
)

#: Non-false-assurance statements per tier.
_ASSURANCE_COOPERATIVE: str = (
    "PII-Guard (cooperative_gateway tier) guarantees PII/secret detection for "
    "outbound LLM API requests that pass through the local proxy.  It does NOT "
    "guarantee protection for processes launched outside the ouroboros framework, "
    "LLM clients with hard-coded base URLs, or any process that does not inherit "
    "the proxy env-var injection.  The protection boundary is limited to "
    "cooperative cooperation with the agent SDK's base-URL configuration."
)

_ASSURANCE_EGRESS_LOCKDOWN: str = (
    "PII-Guard (egress_lockdown tier) adds a pf(4) OS-level network block that "
    "prevents direct outbound TCP to known LLM provider CIDR ranges on ports "
    "80 and 443.  This catches processes that ignore env-var injection.  However, "
    "it does NOT guarantee protection against non-standard ports, IP ranges "
    "outside the static CIDR table, root-level actors, or the proxy process "
    "itself.  The protection boundary is limited to the OS network layer for "
    "traffic to known provider IPs on standard ports."
)


def get_protection_boundary(
    enforcement_tier: str = EnforcementTier.COOPERATIVE_GATEWAY,
    proxy_url: str = "http://127.0.0.1:4444",
    generated_at: Optional[str] = None,
) -> BoundaryReport:
    """
    Return a structured protection-boundary report for the given enforcement tier.

    Parameters
    ----------
    enforcement_tier:
        ``"cooperative_gateway"`` (default) or ``"egress_lockdown"``.
        Determines which defended/undefended items and bypass paths are included.
    proxy_url:
        Base URL of the local PII-Guard proxy (for display purposes).
    generated_at:
        ISO-8601 UTC timestamp string.  If *None*, the current UTC time is used.

    Returns
    -------
    BoundaryReport
        Structured report with no false-assurance claims.

    Raises
    ------
    ValueError
        If *enforcement_tier* is not a recognised value.
    """
    tier = enforcement_tier.lower().strip() if isinstance(enforcement_tier, str) else enforcement_tier

    if tier not in (EnforcementTier.COOPERATIVE_GATEWAY, EnforcementTier.EGRESS_LOCKDOWN,
                    "cooperative_gateway", "egress_lockdown"):
        raise ValueError(
            f"Unknown enforcement_tier: {enforcement_tier!r}. "
            f"Expected 'cooperative_gateway' or 'egress_lockdown'."
        )

    ts = generated_at or datetime.datetime.utcnow().isoformat() + "Z"

    if tier in (EnforcementTier.COOPERATIVE_GATEWAY, "cooperative_gateway"):
        defended = list(_DEFENDED_COOPERATIVE)
        undefended = list(_UNDEFENDED_COOPERATIVE)
        bypass_paths = list(_BYPASS_PATHS_COOPERATIVE)
        assurance = _ASSURANCE_COOPERATIVE
        tier_str = "cooperative_gateway"
    else:
        # egress_lockdown: all cooperative defended items PLUS lockdown extras
        defended = list(_DEFENDED_COOPERATIVE) + list(_DEFENDED_EGRESS_LOCKDOWN_EXTRA)
        undefended = list(_UNDEFENDED_EGRESS_LOCKDOWN_ONLY)
        bypass_paths = list(_BYPASS_PATHS_EGRESS_LOCKDOWN)
        assurance = _ASSURANCE_EGRESS_LOCKDOWN
        tier_str = "egress_lockdown"

    return BoundaryReport(
        enforcement_tier=tier_str,
        proxy_url=proxy_url,
        defended=defended,
        undefended=undefended,
        bypass_paths=bypass_paths,
        threat_actor_model=_THREAT_ACTOR_MODEL,
        assurance_statement=assurance,
        generated_at=ts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable renderer
# ─────────────────────────────────────────────────────────────────────────────

def print_boundary_report(
    report: BoundaryReport,
    stream: TextIO = sys.stdout,
    *,
    verbose: bool = False,
) -> None:
    """
    Render *report* as a human-readable text summary to *stream*.

    Parameters
    ----------
    report:
        The :class:`BoundaryReport` to render.
    stream:
        Output file object (default: ``sys.stdout``).
    verbose:
        If ``True``, include the ``detail`` field for each item.
        If ``False`` (default), show only the one-line ``description``.
    """
    def _hr(char: str = "─", width: int = 72) -> None:
        stream.write(char * width + "\n")

    def _section(title: str) -> None:
        _hr()
        stream.write(f"  {title}\n")
        _hr()

    _hr("═")
    stream.write("  PII-Guard — Protection Boundary Report\n")
    _hr("═")
    stream.write(f"  Enforcement tier : {report.enforcement_tier}\n")
    stream.write(f"  Proxy URL        : {report.proxy_url}\n")
    stream.write(f"  Generated at     : {report.generated_at}\n")
    stream.write("\n")

    _section("✓ DEFENDED — traffic intercepted and scrubbed")
    for item in report.defended:
        stream.write(f"  [{item.category}]  {item.description}\n")
        if verbose:
            for line in item.detail.splitlines():
                stream.write(f"    {line}\n")
            stream.write("\n")

    stream.write("\n")
    _section("✗ NOT DEFENDED — outside the protection boundary")
    for item in report.undefended:
        stream.write(f"  [{item.category}]  {item.description}\n")
        if verbose:
            for line in item.detail.splitlines():
                stream.write(f"    {line}\n")
            stream.write("\n")

    stream.write("\n")
    _section("⚠  BYPASS PATHS — ways to circumvent protection in this tier")
    for i, path in enumerate(report.bypass_paths, 1):
        stream.write(f"  {i}. {path}\n")

    stream.write("\n")
    _section("Threat actor model")
    for line in report.threat_actor_model.splitlines():
        stream.write(f"  {line}\n")

    stream.write("\n")
    _section("Assurance statement (no false assurance)")
    for line in report.assurance_statement.splitlines():
        stream.write(f"  {line}\n")

    _hr("═")
    stream.write("\n")
