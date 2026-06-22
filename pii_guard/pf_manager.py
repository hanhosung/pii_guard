"""
macOS pf(4) anchor rule manager for PII-Guard egress-lockdown tier.

Sub-AC 6b-i — pf firewall rule manager
---------------------------------------
This module builds a pf(4) anchor ruleset that blocks direct outbound TCP
connections to known LLM-provider IP ranges, enforcing that traffic must pass
through the local PII-Guard proxy (cooperative-gateway tier) rather than
reaching provider endpoints directly.

Anchor model
~~~~~~~~~~~~
All rules live inside a named pf anchor so that PII-Guard does not touch the
host's baseline ruleset:

    anchor name : ``piiguard``   (overridable via ``PIIGUARD_PF_ANCHOR``)
    table name  : ``piiguard_llm_ips``

Loading the anchor requires two pfctl calls:

1.  ``sudo pfctl -a piiguard -f -``
        Pipes the generated ruleset text (table definition + block rules) to
        the anchor via stdin.

2.  On disable, flush the anchor's rules **and** destroy the table:
        ``sudo pfctl -a piiguard -F rules``
        ``sudo pfctl -a piiguard -T flush -t piiguard_llm_ips``

Threat model note
~~~~~~~~~~~~~~~~~
This is the **egress-lockdown** enforcement tier, a step above the
cooperative-gateway tier implemented in ``launcher.py``.  It blocks direct
LLM API connections at the network layer so that agents that ignore env-var
injection still cannot reach provider endpoints.

Limitations declared (HonestThreatModel):

* macOS pf does not support per-process rules; blocking is at the IP/port
  level only.  The PII-Guard proxy itself must be allowed out-of-band (e.g.
  run as a different uid, or exempt 127.0.0.1 traffic — which is already
  exempt from pf by default on macOS).
* Root-level actors (sudo/kernel) are outside the threat model (see Seed).
* Loading rules requires ``sudo`` (or appropriate sudoers entry).
* IP ranges are best-effort and must be kept current as providers change CDN.

Usage
~~~~~
    from pii_guard.pf_manager import PfManager

    mgr = PfManager()
    mgr.enable()           # loads anchor — requires sudo access to pfctl
    # ... run ouroboros workflows / LLM CLIs ...
    mgr.disable()          # flushes anchor rules

    # Context-manager form (disable called on exit, even on exception):
    with PfManager() as mgr:
        subprocess.run(["codex", "--prompt", "summarise this"])

    # Low-level helpers (for testing and custom orchestration):
    from pii_guard.pf_manager import build_anchor_rules, ALL_PROVIDER_IP_RANGES
    rules_text = build_anchor_rules(ALL_PROVIDER_IP_RANGES)
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from typing import Dict, List, Optional, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Name of the pf anchor that holds PII-Guard's lockdown rules.
#: Can be overridden at runtime via the ``PIIGUARD_PF_ANCHOR`` environment var.
DEFAULT_ANCHOR_NAME: str = "piiguard"

#: Name of the pf *table* used to hold the IP range set.
#: Must be a valid pf table identifier (no spaces).
ANCHOR_TABLE_NAME: str = "piiguard_llm_ips"

#: TCP ports blocked by the egress-lockdown anchor.
BLOCKED_PORTS: List[int] = [80, 443]

#: Path to the ``pfctl`` binary.  Override via ``PIIGUARD_PFCTL`` env var.
DEFAULT_PFCTL_PATH: str = "/sbin/pfctl"

#: Path to ``sudo``.  Override via ``PIIGUARD_SUDO`` env var.
DEFAULT_SUDO_PATH: str = "/usr/bin/sudo"

# ─────────────────────────────────────────────────────────────────────────────
# Provider IP ranges
# ─────────────────────────────────────────────────────────────────────────────
# These are best-effort, CIDR-notation ranges.  They must be refreshed
# periodically as providers change their CDN and origin IP allocations.
#
# Sources used:
#   Anthropic  — Cloudflare CDN (as of 2024; api.anthropic.com)
#   OpenAI     — Azure Front Door + Fastly CDN (api.openai.com)
#   Google     — Google APIs / Cloud CDN (generativelanguage.googleapis.com)
#
# The module is intentionally provider-agnostic: callers can pass any dict of
# {name: [cidr, ...]} to build_anchor_rules() or PfManager(ip_ranges=...).
# ─────────────────────────────────────────────────────────────────────────────

#: IP ranges attributed to Anthropic's API endpoint.
ANTHROPIC_IP_RANGES: List[str] = [
    # Cloudflare CDN blocks used by api.anthropic.com
    "104.18.0.0/16",
    "104.19.0.0/16",
    "104.20.0.0/16",
    "104.21.0.0/16",
    "172.64.0.0/13",
    "162.158.0.0/15",
    "198.41.128.0/17",
]

#: IP ranges attributed to OpenAI's API endpoint.
OPENAI_IP_RANGES: List[str] = [
    # Azure Front Door used by api.openai.com
    "13.107.0.0/18",
    "13.107.128.0/22",
    "13.107.246.0/24",
    # Fastly CDN
    "104.244.40.0/22",
    "151.101.0.0/16",
]

#: IP ranges attributed to Google Generative Language API (Gemini).
GOOGLE_IP_RANGES: List[str] = [
    # Google APIs / Cloud CDN
    "74.125.0.0/16",
    "142.250.0.0/15",
    "172.217.0.0/16",
    "216.58.192.0/19",
    "64.233.160.0/19",
    "66.249.64.0/19",
]

#: Combined mapping of all default LLM-provider IP ranges.
ALL_PROVIDER_IP_RANGES: Dict[str, List[str]] = {
    "anthropic": ANTHROPIC_IP_RANGES,
    "openai": OPENAI_IP_RANGES,
    "google": GOOGLE_IP_RANGES,
}


# ─────────────────────────────────────────────────────────────────────────────
# Rule-building helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_cidrs(ip_ranges: Dict[str, List[str]]) -> List[str]:
    """
    Flatten a provider-keyed IP-range dict into a deduplicated list of CIDRs.

    Parameters
    ----------
    ip_ranges:
        ``{provider_name: [cidr, ...]}`` mapping.  Empty providers are skipped.

    Returns
    -------
    list[str]
        Flat, deduplicated list of CIDR strings, preserving insertion order.
    """
    seen: set = set()
    cidrs: List[str] = []
    for provider_cidrs in ip_ranges.values():
        for cidr in provider_cidrs:
            cidr = cidr.strip()
            if cidr and cidr not in seen:
                seen.add(cidr)
                cidrs.append(cidr)
    return cidrs


def build_table_definition(cidrs: Sequence[str], table_name: str = ANCHOR_TABLE_NAME) -> str:
    """
    Build the pf ``table`` definition line(s) for the given CIDRs.

    The table uses the ``persist`` flag so pf does not auto-remove it when
    the reference count drops to zero (important during rule reloads).

    Parameters
    ----------
    cidrs:
        Iterable of CIDR strings to include in the table.
    table_name:
        pf table identifier (must match the block rule's ``<table>`` reference).

    Returns
    -------
    str
        Multi-line pf table definition.

    Example
    -------
    >>> print(build_table_definition(["1.2.3.0/24", "5.6.7.0/24"]))
    table <piiguard_llm_ips> persist { \\
        1.2.3.0/24, \\
        5.6.7.0/24  \\
    }
    """
    if not cidrs:
        raise ValueError("build_table_definition: cidrs must not be empty")
    # Format: each CIDR on its own line, comma-separated, last without comma
    cidr_list = list(cidrs)
    lines = []
    for i, cidr in enumerate(cidr_list):
        comma = "," if i < len(cidr_list) - 1 else " "
        lines.append(f"    {cidr}{comma} \\")
    body = "\n".join(lines)
    return f"table <{table_name}> persist {{ \\\n{body}\n}}"


def build_block_rule(
    table_name: str = ANCHOR_TABLE_NAME,
    ports: Sequence[int] = BLOCKED_PORTS,
) -> str:
    """
    Build the pf ``block out`` rule that drops outbound TCP to the table.

    The rule uses ``quick`` so evaluation stops on the first match (no
    later rules can override).

    Parameters
    ----------
    table_name:
        pf table identifier (must match the ``build_table_definition`` call).
    ports:
        TCP port numbers to block (default ``[80, 443]``).

    Returns
    -------
    str
        A single pf block rule string.

    Example
    -------
    >>> build_block_rule()
    'block out quick proto tcp to <piiguard_llm_ips> port { 80 443 }'
    """
    if not ports:
        raise ValueError("build_block_rule: ports must not be empty")
    port_list = " ".join(str(p) for p in ports)
    return f"block out quick proto tcp to <{table_name}> port {{ {port_list} }}"


def build_anchor_rules(
    ip_ranges: Dict[str, List[str]],
    *,
    table_name: str = ANCHOR_TABLE_NAME,
    ports: Sequence[int] = BLOCKED_PORTS,
    anchor_name: str = DEFAULT_ANCHOR_NAME,
) -> str:
    """
    Build the complete pf ruleset text for the PII-Guard anchor.

    The returned string is suitable for piping directly to:
        ``sudo pfctl -a <anchor_name> -f -``

    Parameters
    ----------
    ip_ranges:
        ``{provider_name: [cidr, ...]}`` mapping.  At least one non-empty
        provider is required.
    table_name:
        pf table name used in both the table definition and block rule.
    ports:
        TCP destination ports to block.
    anchor_name:
        Name of the pf anchor (informational only — included in the header
        comment, not in the ruleset body which pf already scopes to the anchor).

    Returns
    -------
    str
        Complete pf ruleset text (header comment + table + block rule).

    Raises
    ------
    ValueError
        If ``ip_ranges`` is empty or contains no non-empty CIDR lists.
    """
    cidrs = collect_all_cidrs(ip_ranges)
    if not cidrs:
        raise ValueError(
            "build_anchor_rules: ip_ranges produced no CIDRs — "
            "cannot build an empty anchor ruleset"
        )

    header = textwrap.dedent(f"""\
        # PII-Guard egress-lockdown anchor: {anchor_name}
        # Auto-generated by pii_guard.pf_manager — do not edit manually.
        # Blocks direct outbound TCP to LLM provider IPs; traffic must route
        # through the local PII-Guard proxy (http://127.0.0.1:4444).
    """)
    table_def = build_table_definition(cidrs, table_name=table_name)
    block_rule = build_block_rule(table_name=table_name, ports=ports)

    return f"{header}\n{table_def}\n\n{block_rule}\n"


# ─────────────────────────────────────────────────────────────────────────────
# pfctl command builders
# ─────────────────────────────────────────────────────────────────────────────

def _pfctl_path() -> str:
    """Return the pfctl binary path (env override → default)."""
    return os.environ.get("PIIGUARD_PFCTL", DEFAULT_PFCTL_PATH)


def _sudo_path() -> str:
    """Return the sudo binary path (env override → default)."""
    return os.environ.get("PIIGUARD_SUDO", DEFAULT_SUDO_PATH)


def build_load_command(anchor_name: str) -> List[str]:
    """
    Return the ``sudo pfctl`` argv for loading rules into an anchor via stdin.

    The caller must supply the rules text as the subprocess's stdin.

    Parameters
    ----------
    anchor_name:
        pf anchor name (e.g. ``"piiguard"``).

    Returns
    -------
    list[str]
        Command array suitable for ``subprocess.run(..., input=rules_text)``.

    Example
    -------
    >>> build_load_command("piiguard")
    ['/usr/bin/sudo', '/sbin/pfctl', '-a', 'piiguard', '-f', '-']
    """
    return [_sudo_path(), _pfctl_path(), "-a", anchor_name, "-f", "-"]


def build_flush_rules_command(anchor_name: str) -> List[str]:
    """
    Return the argv for flushing all rules in an anchor (``pfctl -F rules``).

    Flushing rules does NOT destroy persistent tables — call
    :func:`build_flush_table_command` separately for a full teardown.

    Parameters
    ----------
    anchor_name:
        pf anchor name.

    Returns
    -------
    list[str]
        Command array.

    Example
    -------
    >>> build_flush_rules_command("piiguard")
    ['/usr/bin/sudo', '/sbin/pfctl', '-a', 'piiguard', '-F', 'rules']
    """
    return [_sudo_path(), _pfctl_path(), "-a", anchor_name, "-F", "rules"]


def build_flush_table_command(anchor_name: str, table_name: str) -> List[str]:
    """
    Return the argv for flushing (destroying) a pf table inside an anchor.

    Parameters
    ----------
    anchor_name:
        pf anchor name.
    table_name:
        pf table name to destroy.

    Returns
    -------
    list[str]
        Command array.

    Example
    -------
    >>> build_flush_table_command("piiguard", "piiguard_llm_ips")
    ['/usr/bin/sudo', '/sbin/pfctl', '-a', 'piiguard', '-T', 'flush', '-t', 'piiguard_llm_ips']
    """
    return [
        _sudo_path(), _pfctl_path(),
        "-a", anchor_name,
        "-T", "flush",
        "-t", table_name,
    ]


def build_show_rules_command(anchor_name: str) -> List[str]:
    """
    Return the argv for showing current rules in an anchor (read-only).

    Parameters
    ----------
    anchor_name:
        pf anchor name.

    Returns
    -------
    list[str]
        Command array.

    Example
    -------
    >>> build_show_rules_command("piiguard")
    ['/usr/bin/sudo', '/sbin/pfctl', '-a', 'piiguard', '-s', 'rules']
    """
    return [_sudo_path(), _pfctl_path(), "-a", anchor_name, "-s", "rules"]


# ─────────────────────────────────────────────────────────────────────────────
# PfManager — high-level enable / disable API
# ─────────────────────────────────────────────────────────────────────────────

class PfRuleError(RuntimeError):
    """Raised when a pfctl invocation fails."""


class PfManager:
    """
    High-level manager for the PII-Guard pf(4) egress-lockdown anchor.

    Loads and unloads a named pf anchor containing block rules that prevent
    direct outbound TCP connections to LLM provider IP ranges, forcing traffic
    through the local PII-Guard proxy.

    Parameters
    ----------
    ip_ranges:
        Provider IP ranges to include in the anchor ruleset.  Defaults to
        :data:`ALL_PROVIDER_IP_RANGES` (Anthropic + OpenAI + Google).
    anchor_name:
        pf anchor name.  Defaults to :data:`DEFAULT_ANCHOR_NAME` (``"piiguard"``).
        Can also be set via the ``PIIGUARD_PF_ANCHOR`` environment variable.
    table_name:
        pf table name.  Defaults to :data:`ANCHOR_TABLE_NAME`.
    ports:
        TCP ports to block.  Defaults to :data:`BLOCKED_PORTS` (``[80, 443]``).
    check_output:
        If ``True`` (default), ``subprocess.run`` is called with
        ``check=True``; a non-zero pfctl exit code raises :exc:`PfRuleError`.
        Set to ``False`` to suppress exceptions (useful for best-effort teardown).

    Usage
    ~~~~~
    ::

        mgr = PfManager()
        mgr.enable()
        # ... run workflows ...
        mgr.disable()

        # Context manager:
        with PfManager() as mgr:
            ...

    Testing
    ~~~~~~~
    Inject a mock for ``_run_pfctl`` to avoid needing root or a real pf stack::

        with unittest.mock.patch.object(mgr, "_run_pfctl") as mock_run:
            mgr.enable()
            # assert mock_run.call_args_list ...
    """

    def __init__(
        self,
        ip_ranges: Optional[Dict[str, List[str]]] = None,
        *,
        anchor_name: Optional[str] = None,
        table_name: str = ANCHOR_TABLE_NAME,
        ports: Optional[List[int]] = None,
        check_output: bool = True,
    ) -> None:
        self.ip_ranges: Dict[str, List[str]] = (
            ip_ranges if ip_ranges is not None else ALL_PROVIDER_IP_RANGES
        )
        self.anchor_name: str = (
            anchor_name
            or os.environ.get("PIIGUARD_PF_ANCHOR", DEFAULT_ANCHOR_NAME)
        )
        self.table_name: str = table_name
        self.ports: List[int] = ports if ports is not None else list(BLOCKED_PORTS)
        self.check_output: bool = check_output
        self._enabled: bool = False

    # ── Internal subprocess wrapper ───────────────────────────────────────────

    def _run_pfctl(
        self,
        cmd: List[str],
        *,
        input_text: Optional[str] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run a pfctl command via subprocess.

        This thin wrapper is the single seam for unit tests — mock this method
        to avoid needing root privileges or a real pf stack.

        Parameters
        ----------
        cmd:
            Full argv list (including ``sudo`` and ``pfctl``).
        input_text:
            If not ``None``, the string is fed to the process's stdin.
        check:
            If ``True`` and the exit code is non-zero, raises
            :exc:`PfRuleError`.

        Returns
        -------
        subprocess.CompletedProcess

        Raises
        ------
        PfRuleError
            When ``check=True`` and pfctl exits non-zero.
        """
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise PfRuleError(
                f"pfctl not found at {cmd[1]!r}: {exc}"
            ) from exc

        if check and result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else ""
            raise PfRuleError(
                f"pfctl command failed (exit {result.returncode}): "
                f"{' '.join(cmd)}\n"
                f"stderr: {stderr}"
            )
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def build_rules(self) -> str:
        """
        Return the complete pf anchor ruleset text for the current configuration.

        The text is suitable for piping to ``pfctl -a <anchor> -f -``.
        """
        return build_anchor_rules(
            self.ip_ranges,
            table_name=self.table_name,
            ports=self.ports,
            anchor_name=self.anchor_name,
        )

    def enable(self) -> None:
        """
        Load the egress-lockdown anchor rules into pf.

        Builds the ruleset from the configured IP ranges and feeds it to
        ``sudo pfctl -a <anchor_name> -f -`` via stdin.

        Raises
        ------
        PfRuleError
            If pfctl exits non-zero (unless ``check_output=False``).
        ValueError
            If the configured IP ranges would produce an empty ruleset.
        """
        rules_text = self.build_rules()
        cmd = build_load_command(self.anchor_name)
        self._run_pfctl(cmd, input_text=rules_text, check=self.check_output)
        self._enabled = True

    def disable(self) -> None:
        """
        Flush the egress-lockdown anchor rules from pf.

        Executes two pfctl calls in sequence:

        1. ``sudo pfctl -a <anchor_name> -F rules``  — remove block rules.
        2. ``sudo pfctl -a <anchor_name> -T flush -t <table_name>``  — destroy
           the persistent IP table.

        Both calls are made with ``check=False`` so that a partially-loaded
        anchor (e.g. if ``enable()`` was interrupted) still tears down cleanly.

        The ``_enabled`` flag is cleared even if pfctl calls fail, to prevent
        double-teardown on subsequent ``disable()`` calls.
        """
        # Always clear the flag first so double-disable is safe
        self._enabled = False

        errors: List[str] = []

        # 1. Flush rules
        try:
            cmd = build_flush_rules_command(self.anchor_name)
            self._run_pfctl(cmd, check=False)
        except PfRuleError as exc:
            errors.append(f"flush rules: {exc}")

        # 2. Flush / destroy the IP table
        try:
            cmd = build_flush_table_command(self.anchor_name, self.table_name)
            self._run_pfctl(cmd, check=False)
        except PfRuleError as exc:
            errors.append(f"flush table: {exc}")

        if errors and self.check_output:
            raise PfRuleError("disable() errors:\n" + "\n".join(errors))

    def status(self) -> Optional[str]:
        """
        Return the current rules loaded in the anchor, or ``None`` on error.

        This is a read-only operation that does not change firewall state.

        Returns
        -------
        str or None
            pfctl output (may be empty if no rules are loaded), or ``None``
            if the command fails (e.g. insufficient privileges).
        """
        try:
            cmd = build_show_rules_command(self.anchor_name)
            result = self._run_pfctl(cmd, check=False)
            return result.stdout or ""
        except PfRuleError:
            return None

    @property
    def is_enabled(self) -> bool:
        """``True`` if ``enable()`` has been called and ``disable()`` has not."""
        return self._enabled

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "PfManager":
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disable()
        return False  # never suppress exceptions

    def __repr__(self) -> str:
        return (
            f"PfManager("
            f"anchor_name={self.anchor_name!r}, "
            f"table_name={self.table_name!r}, "
            f"ports={self.ports!r}, "
            f"enabled={self._enabled})"
        )
