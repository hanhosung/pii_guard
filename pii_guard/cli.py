"""
PII-Guard command-line interface.

Sub-AC 6b-ii: CLI enable/disable commands for the pf(4) egress-lockdown tier.
Sub-AC 3c:    ``serve`` sub-command — run the proxy as a foreground process
              whose network-layer kill behaviour is verified by integration tests.
Sub-AC 5d-ii: ``pin-list propose`` sub-command — sole write path for pin-list
              changes via the out-of-band interactive approval flow.

Usage
-----
::

    # Run the intercepting proxy (Sub-AC 3c)
    piiguard serve --upstream-url https://api.anthropic.com
    piiguard serve --upstream-url https://api.openai.com --port 4445
    # Prints "READY <port>" to stdout once listening, then blocks.
    # SIGKILL or SIGTERM closes all connections fail-closed (OS RST).

    # Egress lockdown (requires sudo / root)
    piiguard egress enable    # load pf anchor — blocks direct LLM API TCP
    piiguard egress disable   # flush pf anchor — restores direct access
    piiguard egress status    # show current anchor rules

    # Pin-list approval flow (Sub-AC 5d-ii)
    piiguard pin-list propose --policy-path /path/to/policy.yaml \\
        --add hash=sha256:abc,category=EMAIL,action=allow,label="dev relay"
    # Shows diff, prompts for Y/N, commits on yes / discards on no.

    # Alternate invocation (no install needed)
    python3 -m pii_guard.cli serve --upstream-url https://api.anthropic.com
    python3 -m pii_guard.cli egress enable

Exit codes
----------
0   success (serve exits 0 only if stopped gracefully via SIGTERM/SIGINT)
1   pfctl error (rules failed to load / flush)
2   argument / usage error
3   unsupported platform (pfctl not found)
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional

from .boundary import EnforcementTier, get_protection_boundary, print_boundary_report
from .engine import Engine
from .ledger import Ledger
from .pf_manager import PfManager, PfRuleError
from .pinlist_approval import PinListApprovalGate, run_interactive_approval
from .policy import PinListEntry
from .proxy import PIIGuardProxy


# ─────────────────────────────────────────────────────────────────────────────
# Version
# ─────────────────────────────────────────────────────────────────────────────

_VERSION = "0.1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_err(msg: str) -> None:
    """Write an error message to stderr."""
    print(f"pii-guard: error: {msg}", file=sys.stderr)


def _print_ok(msg: str) -> None:
    """Write an info message to stdout."""
    print(f"pii-guard: {msg}")


def _print_warn(msg: str) -> None:
    """Write a warning to stderr."""
    print(f"pii-guard: warning: {msg}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> int:
    """
    Start the PII-Guard intercepting proxy as a foreground process.

    Sub-AC 3c — Fail-closed on process crash
    ----------------------------------------
    The proxy binds a TCP server socket.  When the process receives SIGKILL
    (or any fatal signal), the OS immediately closes all file descriptors and
    issues TCP RST on every open connection — both the client-facing listening
    socket and any accepted per-request sockets.  Clients receive a connection-
    reset error; no buffered upstream response is delivered.  This is the
    operating-system-level fail-closed guarantee for process crashes.

    Readiness signalling
    --------------------
    Once the socket is bound and listening, this function writes::

        READY <port>\\n

    to stdout and flushes.  Parent processes (e.g. integration tests) read
    this line to learn the ephemeral port and to confirm the proxy is ready.

    Blocking
    --------
    The process then blocks in ``signal.pause()`` until it receives any
    signal.  SIGTERM / SIGINT trigger a graceful shutdown; SIGKILL terminates
    immediately (OS handles connection reset).
    """
    # Load policy (proximity keywords/window + NER-filter knobs come from here;
    # falls back to secure built-in defaults when no file is given/found).
    from .policy import load_policy
    policy = load_policy(getattr(args, "policy", None))

    # Secure-by-default: wire the Stage-2 Korean NER engine into the proxy so
    # unstructured PII (person names, addresses, organizations) is masked, not
    # just Stage-1 regex categories. The runner is subprocess-isolated and
    # degrades gracefully to Stage-1 if NER deps/model are unavailable or the
    # worker OOMs/times out (AC 3). Use --no-ner to run Stage-1 only.
    runner = None
    if not getattr(args, "no_ner", False):
        from .stage2.runner import Stage2NERRunner

        runner = Stage2NERRunner()
    engine = Engine(
        stage2_runner=runner,
        proximity_config=policy.proximity,
        ner_backend=policy.ner_backend,   # R18: gliner(기본)/spacy. env PIIGUARD_NER_BACKEND가 우선
    )

    # Warm up the NER worker before serving so the model loads ONCE, outside the
    # per-block timeout. Heavy backends (e.g. GLiNER cold-load ~15s) otherwise
    # exceed the per-block timeout on the first request, get killed, and degrade
    # to Stage-1 on *every* request (names/addresses leak). Engine() above set
    # PIIGUARD_NER_BACKEND, so the worker loads the selected backend. Best-effort.
    if runner is not None:
        import os as _os
        _backend = _os.environ.get("PIIGUARD_NER_BACKEND", "gliner")
        sys.stderr.write(f"[PII-Guard] warming up Stage-2 NER backend ({_backend}) — "
                         f"loading model, this can take ~15s...\n")
        sys.stderr.flush()
        _ok = runner.warmup()
        sys.stderr.write(
            f"[PII-Guard] NER warmup {'ready' if _ok else 'FAILED — will degrade per-request'}\n"
        )
        sys.stderr.flush()

    proxy = PIIGuardProxy(
        args.upstream_url,
        host=args.host,
        port=args.port,
        engine=engine,
        log_masked=getattr(args, "log_masked", False),
    )
    proxy.start()

    # Signal readiness to parent (used by integration tests and process supervisors)
    sys.stdout.write(f"READY {proxy.port}\n")
    sys.stdout.flush()

    # Install SIGTERM handler — fail-closed: exit immediately without waiting
    # for in-flight requests to complete.
    #
    # Rationale (Sub-AC 3c):
    #   Calling proxy.stop() (HTTPServer.shutdown()) from a signal handler would
    #   wait for the current request handler thread to finish, which may block
    #   for tens of seconds while urllib waits for a slow upstream.  During that
    #   window the handler *could* still receive an upstream response and forward
    #   it to the client — violating the fail-closed guarantee.
    #
    #   os._exit() skips all Python finalizers and calls the C _exit() directly,
    #   causing the OS to immediately close every FD and send TCP RST on every
    #   open connection (client-facing and upstream-facing alike).  Clients see
    #   ConnectionResetError — fail-closed without exception.
    def _on_sigterm(signum, frame):  # pragma: no cover
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        # Block the main thread indefinitely; daemon proxy thread keeps serving.
        signal.pause()
    except KeyboardInterrupt:
        # SIGINT (Ctrl-C): exit immediately for the same fail-closed reason.
        os._exit(0)

    return 0


def cmd_egress_enable(args: argparse.Namespace) -> int:
    """
    Load the PII-Guard pf(4) egress-lockdown anchor.

    Builds the block-rules anchor from ALL_PROVIDER_IP_RANGES and loads it
    into pf via ``sudo pfctl -a piiguard -f -``.

    Returns an exit code: 0 = success, 1 = pfctl error, 3 = pfctl not found.
    """
    anchor_name: str = args.anchor_name or os.environ.get(
        "PIIGUARD_PF_ANCHOR", "piiguard"
    )

    _print_ok(f"enabling egress-lockdown anchor '{anchor_name}' …")

    mgr = PfManager(anchor_name=anchor_name)
    try:
        mgr.enable()
    except ValueError as exc:
        _print_err(f"configuration error: {exc}")
        return 2
    except PfRuleError as exc:
        _print_err(str(exc))
        if "not found" in str(exc).lower():
            _print_err(
                "pfctl binary not found — is this macOS with pf(4) installed?"
            )
            return 3
        _print_err(
            "hint: this command requires sudo / root privileges. "
            "Run: sudo piiguard egress enable"
        )
        return 1

    if args.verbose:
        rules = mgr.status() or "(no rules returned)"
        print()
        print("Loaded anchor rules:")
        print(rules)

    _print_ok("egress-lockdown enabled. Direct outbound TCP to LLM provider IPs is now blocked.")
    _print_ok("To restore direct access, run: piiguard egress disable")
    return 0


def cmd_egress_disable(args: argparse.Namespace) -> int:
    """
    Flush the PII-Guard pf(4) egress-lockdown anchor.

    Runs:
    1. ``sudo pfctl -a <anchor> -F rules``
    2. ``sudo pfctl -a <anchor> -T flush -t piiguard_llm_ips``

    Returns an exit code: 0 = success, 1 = pfctl error.
    """
    anchor_name: str = args.anchor_name or os.environ.get(
        "PIIGUARD_PF_ANCHOR", "piiguard"
    )

    _print_ok(f"disabling egress-lockdown anchor '{anchor_name}' …")

    mgr = PfManager(anchor_name=anchor_name)
    try:
        mgr.disable()
    except PfRuleError as exc:
        _print_err(str(exc))
        _print_warn(
            "partial teardown may have occurred. Check anchor state with: "
            "piiguard egress status"
        )
        return 1

    _print_ok("egress-lockdown disabled. Direct outbound TCP to LLM provider IPs is now allowed.")
    return 0


def cmd_egress_status(args: argparse.Namespace) -> int:
    """
    Show the current rules loaded in the PII-Guard pf(4) anchor.

    Runs ``sudo pfctl -a <anchor> -s rules`` and prints the result.
    Returns 0 if rules are present, 0 with a 'not active' message if empty,
    or 1 if pfctl cannot be queried.
    """
    anchor_name: str = args.anchor_name or os.environ.get(
        "PIIGUARD_PF_ANCHOR", "piiguard"
    )

    mgr = PfManager(anchor_name=anchor_name)
    rules = mgr.status()

    if rules is None:
        _print_warn(
            f"could not query anchor '{anchor_name}'. "
            "pfctl may require sudo or pf may be disabled."
        )
        return 1

    if rules.strip():
        print(f"Anchor '{anchor_name}' rules:")
        print(rules.rstrip())
        _print_ok("egress-lockdown is ACTIVE.")
    else:
        _print_ok(f"Anchor '{anchor_name}' has no rules — egress-lockdown is NOT active.")

    return 0


def cmd_ledger_purge(args: argparse.Namespace) -> int:
    """
    Delete ALL ledger files and reset ledger state.

    Sub-AC 3 — Purge command
    -------------------------
    Locates the active ledger file and every rotated archive, deletes them
    all, and resets all in-process ledger state (``_initialized``,
    ``_file_created_at``).

    By default a fresh empty seed file is created immediately after the
    purge so that the operator can confirm the ledger path is accessible
    and carries the correct 0o600 / 0o700 permissions before the first
    real event is written.  Pass ``--no-reseed`` to skip seed-file creation
    and leave the ledger directory empty.

    Exit codes
    ----------
    0   purge succeeded (and seed file created unless --no-reseed)
    1   OS error during purge (e.g. permission denied)

    Environment
    -----------
    PIIGUARD_LEDGER_PATH
        Default ledger path when ``--ledger-path`` is not supplied.
        Overrides the built-in default of ``~/.piiguard/ledger.jsonl``.
    """
    ledger_path_str: str = (
        args.ledger_path
        or os.environ.get("PIIGUARD_LEDGER_PATH", "")
        or str(Path.home() / ".piiguard" / "ledger.jsonl")
    )
    ledger_path = Path(ledger_path_str)

    _print_ok(f"purging ledger at {ledger_path} …")

    # Use a one-shot ephemeral key — HMAC is irrelevant for a purge operation.
    hmac_key = os.urandom(32)
    ledger = Ledger(ledger_path, hmac_key)

    try:
        ledger.purge()
    except OSError as exc:
        _print_err(f"failed to purge ledger: {exc}")
        return 1

    _print_ok("all ledger files deleted; in-process state fully reset.")

    if not getattr(args, "no_reseed", False):
        try:
            ledger.initialize()
            _print_ok(
                f"fresh seed ledger created at {ledger_path} "
                f"(mode 0o600, parent dir 0o700)."
            )
        except OSError as exc:
            _print_warn(f"purge succeeded but could not create seed file: {exc}")
            # Non-fatal: the ledger is still purged; the seed file is optional.

    return 0


def _parse_pin_list_entry_arg(raw: str) -> PinListEntry:
    """
    Parse a ``key=value,...`` string into a :class:`~pii_guard.policy.PinListEntry`.

    Expected format::

        hash=sha256:abc,category=EMAIL,action=allow
        hash=sha256:abc,category=EMAIL,action=allow,label="internal relay"

    Raises ``argparse.ArgumentTypeError`` on invalid format.
    """
    parts: dict = {}
    for token in raw.split(","):
        token = token.strip()
        if "=" not in token:
            raise argparse.ArgumentTypeError(
                f"Invalid pin-list entry token {token!r} — expected key=value pairs "
                "separated by commas."
            )
        k, _, v = token.partition("=")
        parts[k.strip().lower()] = v.strip().strip('"').strip("'")

    h = parts.get("hash", "")
    cat = parts.get("category", "")
    action = parts.get("action", "")
    label = parts.get("label", "")

    missing = [f for f, v in [("hash", h), ("category", cat), ("action", action)] if not v]
    if missing:
        raise argparse.ArgumentTypeError(
            f"Pin-list entry is missing required field(s): {', '.join(missing)}. "
            "Expected: hash=<hash>,category=<CATEGORY>,action=<action>[,label=<text>]"
        )

    _valid_actions = {"allow", "mask", "block", "tokenize_roundtrip"}
    if action not in _valid_actions:
        raise argparse.ArgumentTypeError(
            f"Invalid action {action!r} — must be one of: {', '.join(sorted(_valid_actions))}"
        )

    return PinListEntry(hash=h, category=cat, action=action, label=label)


def cmd_pin_list_propose(args: argparse.Namespace) -> int:
    """
    Interactive out-of-band approval flow for pin-list changes (Sub-AC 5d-ii).

    This is the **sole write path** for pin-list mutations.  The command:

    1. Reads the current pin-list from the policy YAML.
    2. Displays a diff of the proposed changes (entries to add and remove).
    3. Prompts the user for explicit confirmation (Y/N).
    4. On YES — writes the updated pin-list + ``pin_list_approved: true``
       atomically to the policy file.
    5. On NO  — discards the change; the policy file is not modified.

    Protection guarantee
    --------------------
    Any attempt to write a pin-list change to the policy YAML *without*
    going through this command (i.e. without setting ``pin_list_approved: true``)
    is caught and rejected by ``PolicyLoader`` on the next hot-reload.

    Exit codes
    ----------
    0   approved and committed
    1   rejected / discarded
    2   argument error
    """
    policy_path: str = (
        args.policy_path
        or os.environ.get("PIIGUARD_POLICY_PATH", "")
    )
    if not policy_path:
        _print_err(
            "No policy file specified.  Use --policy-path or set "
            "PIIGUARD_POLICY_PATH."
        )
        return 2

    # Build the proposed new pin-list from the --add flags
    new_entries: List[PinListEntry] = list(args.add or [])

    if not new_entries:
        _print_warn(
            "No --add entries specified.  "
            "The proposed list is empty (all current entries would be removed)."
        )

    result = run_interactive_approval(
        policy_path,
        new_entries,
        input_fn=None,   # default: real input()
        output_fn=None,  # default: real print()
    )

    if result.approved and result.committed:
        return 0
    return 1


def cmd_boundary(args: argparse.Namespace) -> int:
    """
    Print (or emit as JSON) the PII-Guard protection-boundary report.

    Sub-AC 6c — Protection-boundary declaration
    -------------------------------------------
    Returns a structured report enumerating:
      • What IS defended (process classes, protocols, ports)
      • What is NOT defended (explicit, no false assurance)
      • Bypass paths for the current enforcement tier
      • Threat-actor model declaration

    Exit codes
    ----------
    0   success
    2   invalid mode argument
    """
    mode = args.mode or EnforcementTier.COOPERATIVE_GATEWAY
    try:
        report = get_protection_boundary(
            enforcement_tier=mode,
            proxy_url=args.proxy_url,
        )
    except ValueError as exc:
        _print_err(str(exc))
        return 2

    if args.json:
        print(report.as_json())
    else:
        print_boundary_report(report, stream=sys.stdout, verbose=args.verbose)

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the top-level argument parser for the ``piiguard`` CLI.

    Sub-commands
    ------------
    serve           — run the intercepting proxy (Sub-AC 3c)
    egress enable   — load pf(4) egress-lockdown anchor
    egress disable  — flush pf(4) egress-lockdown anchor
    egress status   — show current anchor rules
    """
    parser = argparse.ArgumentParser(
        prog="piiguard",
        description=(
            "PII-Guard: local-first PII/secret detection gateway.\n\n"
            "Use 'piiguard <command> --help' for command-specific help."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_VERSION}"
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>",
    )
    subparsers.required = True

    # ── serve sub-command (Sub-AC 3c) ─────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="run the PII-Guard intercepting proxy (foreground)",
        description=(
            "Start the PII-Guard intercepting proxy that scrubs PII/secrets from\n"
            "outbound LLM requests before forwarding to the real upstream endpoint.\n\n"
            "The proxy binds to the specified host/port (or an OS-assigned ephemeral\n"
            "port when --port 0 is used) and writes 'READY <port>' to stdout once\n"
            "it is accepting connections.\n\n"
            "Fail-closed guarantee (Sub-AC 3c):\n"
            "  A SIGKILL or unhandled crash causes the OS to RST all open TCP\n"
            "  connections immediately — clients receive a connection-reset error,\n"
            "  never a silently forwarded upstream response.\n\n"
            "Set the client's base_url to http://<host>:<port> to redirect traffic\n"
            "through the proxy (e.g. ANTHROPIC_BASE_URL=http://127.0.0.1:4444)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    serve_parser.add_argument(
        "--upstream-url",
        required=True,
        metavar="URL",
        help="base URL of the real upstream LLM endpoint (e.g. https://api.anthropic.com)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=0,
        metavar="PORT",
        help="local port to bind (0 = OS-assigned ephemeral port, default)",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        metavar="HOST",
        help="local address to bind (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--no-ner",
        action="store_true",
        help=(
            "disable Stage-2 Korean NER (run Stage-1 regex only). NER is ON by "
            "default so person names / addresses / organizations are masked; this "
            "flag is an escape hatch for minimal/low-resource deployments."
        ),
    )
    serve_parser.add_argument(
        "--policy",
        metavar="PATH",
        default=None,
        help=(
            "path to a policy YAML. The 'proximity:' block (trigger keywords, "
            "window, NER-filter knobs) is read from here; omit to use secure "
            "built-in defaults."
        ),
    )
    serve_parser.add_argument(
        "--log-masked",
        action="store_true",
        help=(
            "print the masked payload + detection summary to stdout before "
            "forwarding to the upstream (e.g. real Anthropic). Confirms PII is "
            "masked/blocked before leaving the host. Only the MASKED payload is "
            "logged — never the raw request body."
        ),
    )
    serve_parser.set_defaults(func=cmd_serve)

    # ── egress sub-command group ───────────────────────────────────────────────
    egress_parser = subparsers.add_parser(
        "egress",
        help="manage the pf(4) egress-lockdown firewall anchor",
        description=(
            "Manage the pf(4) egress-lockdown anchor that blocks direct outbound\n"
            "TCP connections to LLM provider IP ranges (Anthropic, OpenAI, Google).\n\n"
            "Requires root / sudo privileges for enable and disable operations.\n\n"
            "Threat model: this blocks agents that honour the OS network stack\n"
            "but does NOT protect against root-level or kernel-level actors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    egress_sub = egress_parser.add_subparsers(
        title="egress sub-commands",
        dest="egress_command",
        metavar="<sub-command>",
    )
    egress_sub.required = True

    # Shared anchor-name option
    _anchor_kwargs = dict(
        metavar="ANCHOR",
        default=None,
        help=(
            "pf anchor name (default: 'piiguard', or PIIGUARD_PF_ANCHOR env var)"
        ),
    )

    # -- egress enable --
    enable_parser = egress_sub.add_parser(
        "enable",
        help="load the egress-lockdown pf anchor (requires sudo)",
        description=(
            "Load the PII-Guard pf(4) anchor that blocks direct outbound TCP\n"
            "connections to LLM provider IP ranges on ports 80 and 443.\n\n"
            "Effect: ouroboros workflows and LLM CLIs must route through the\n"
            "local PII-Guard proxy (http://127.0.0.1:4444) rather than calling\n"
            "provider APIs directly.\n\n"
            "Requires: macOS pf(4) + sudo / root"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    enable_parser.add_argument("--anchor-name", **_anchor_kwargs)
    enable_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="print the loaded pf ruleset after enabling",
    )
    enable_parser.set_defaults(func=cmd_egress_enable)

    # -- egress disable --
    disable_parser = egress_sub.add_parser(
        "disable",
        help="flush the egress-lockdown pf anchor (requires sudo)",
        description=(
            "Flush all rules from the PII-Guard pf(4) anchor and destroy the\n"
            "IP table, restoring direct outbound access to LLM provider endpoints.\n\n"
            "Requires: macOS pf(4) + sudo / root"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    disable_parser.add_argument("--anchor-name", **_anchor_kwargs)
    disable_parser.set_defaults(func=cmd_egress_disable)

    # -- egress status --
    status_parser = egress_sub.add_parser(
        "status",
        help="show current pf anchor rules",
        description=(
            "Query the PII-Guard pf(4) anchor and display the currently loaded\n"
            "rules. Output is empty if egress-lockdown is not active.\n\n"
            "Requires: macOS pf(4) + sudo / root (read-only)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    status_parser.add_argument("--anchor-name", **_anchor_kwargs)
    status_parser.set_defaults(func=cmd_egress_status)

    # ── ledger sub-command group (Sub-AC 3) ───────────────────────────────────
    ledger_parser = subparsers.add_parser(
        "ledger",
        help="manage the PII-Guard audit ledger",
        description=(
            "Commands for managing the PII-Guard append-only audit ledger.\n\n"
            "The ledger records block / mask / fail / coverage-gap events as\n"
            "HMAC-keyed metadata entries.  No recoverable PII or secret values\n"
            "are ever persisted — only keyed hashes and structural metadata.\n\n"
            "Sub-commands\n"
            "------------\n"
            "  purge   — delete all ledger files and reset state"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ledger_sub = ledger_parser.add_subparsers(
        title="ledger sub-commands",
        dest="ledger_command",
        metavar="<sub-command>",
    )
    ledger_sub.required = True

    # -- ledger purge --
    ledger_purge_parser = ledger_sub.add_parser(
        "purge",
        help="delete all ledger files and reset state (Sub-AC 3)",
        description=(
            "Delete the active ledger file and every rotated archive, then\n"
            "fully reset all in-process ledger state.\n\n"
            "By default a fresh empty seed file is immediately re-created at\n"
            "the ledger path with 0o600 permissions (parent directory 0o700)\n"
            "so that the operator can confirm readiness.  Pass --no-reseed to\n"
            "leave the ledger directory empty after the purge.\n\n"
            "Verification (Sub-AC 3)\n"
            "-----------------------\n"
            "(c) No ledger files remain after purge; in-process state is reset.\n"
            "(d) The fresh seed file (when created) carries 0o600/0o700 perms.\n\n"
            "The ledger path is resolved in order:\n"
            "  1. --ledger-path flag\n"
            "  2. PIIGUARD_LEDGER_PATH environment variable\n"
            "  3. ~/.piiguard/ledger.jsonl (built-in default)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ledger_purge_parser.add_argument(
        "--ledger-path",
        metavar="PATH",
        default=None,
        help=(
            "path to the active ledger file "
            "(default: $PIIGUARD_LEDGER_PATH or ~/.piiguard/ledger.jsonl)"
        ),
    )
    ledger_purge_parser.add_argument(
        "--no-reseed",
        action="store_true",
        default=False,
        help=(
            "skip creating a fresh seed file after purging; "
            "the ledger directory is left empty"
        ),
    )
    ledger_purge_parser.set_defaults(func=cmd_ledger_purge)

    # ── boundary sub-command (Sub-AC 6c) ──────────────────────────────────────
    boundary_parser = subparsers.add_parser(
        "boundary",
        help="show the protection-boundary report (what is and isn't defended)",
        description=(
            "Print a structured report enumerating exactly what PII-Guard\n"
            "defends in the current enforcement tier and — critically — what\n"
            "it does NOT defend, with explicit bypass paths and a non-false-\n"
            "assurance declaration.\n\n"
            "Enforcement tiers\n"
            "-----------------\n"
            "  cooperative_gateway  (default)\n"
            "    Base-URL env-var injection only.  Processes spawned via\n"
            "    ProcessLauncher or inside env_patch() are protected; all\n"
            "    others are NOT.\n\n"
            "  egress_lockdown\n"
            "    pf(4) firewall anchor blocks direct outbound TCP to known\n"
            "    LLM provider CIDR ranges.  Processes that ignore env vars\n"
            "    are still blocked at the OS network layer.  Root actors\n"
            "    and non-standard ports remain outside the boundary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    boundary_parser.add_argument(
        "--mode",
        choices=["cooperative_gateway", "egress_lockdown"],
        default="cooperative_gateway",
        metavar="MODE",
        help=(
            "enforcement tier to report on: 'cooperative_gateway' (default) "
            "or 'egress_lockdown'"
        ),
    )
    boundary_parser.add_argument(
        "--proxy-url",
        default="http://127.0.0.1:4444",
        metavar="URL",
        help="proxy base URL shown in the report (default: http://127.0.0.1:4444)",
    )
    boundary_parser.add_argument(
        "--json",
        action="store_true",
        help="emit the report as machine-readable JSON instead of human text",
    )
    boundary_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="include detailed explanation for each defended/undefended item",
    )
    boundary_parser.set_defaults(func=cmd_boundary)

    # ── pin-list sub-command group (Sub-AC 5d-ii) ──────────────────────────────
    pinlist_parser = subparsers.add_parser(
        "pin-list",
        help="manage the pin-list via the out-of-band approval flow (Sub-AC 5d-ii)",
        description=(
            "Commands for managing the PII-Guard pin-list.\n\n"
            "The pin-list is part of the security control plane — entries here\n"
            "control which specific PII values receive per-value treatment\n"
            "(allow / mask / block / tokenize_roundtrip).\n\n"
            "Any change to the pin-list MUST go through the interactive approval\n"
            "flow (this command).  Programmatic changes via the agent API are\n"
            "permanently blocked (AGENT_MUTATION_BLOCKED).  Direct YAML writes\n"
            "without pin_list_approved: true are rejected by PolicyLoader on\n"
            "the next hot-reload.\n\n"
            "Sub-commands\n"
            "------------\n"
            "  propose   — interactive approval flow (show diff, prompt, commit/discard)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    pinlist_sub = pinlist_parser.add_subparsers(
        title="pin-list sub-commands",
        dest="pinlist_command",
        metavar="<sub-command>",
    )
    pinlist_sub.required = True

    # -- pin-list propose --
    propose_parser = pinlist_sub.add_parser(
        "propose",
        help="propose a pin-list change and approve/reject interactively",
        description=(
            "Propose a new pin-list for the policy file.\n\n"
            "The command reads the current pin-list, shows a diff of what will\n"
            "change, and prompts for explicit confirmation (Y/N).\n\n"
            "  YES → writes the new pin-list + pin_list_approved: true atomically.\n"
            "  NO  → discards the change; the policy file is not touched.\n\n"
            "The --add flag specifies the COMPLETE new pin-list (not a delta).\n"
            "All current entries not mentioned in --add will be removed.\n\n"
            "Entry format\n"
            "------------\n"
            "  hash=<hash>,category=<CATEGORY>,action=<action>[,label=<text>]\n\n"
            "Example\n"
            "-------\n"
            "  piiguard pin-list propose \\\n"
            "    --policy-path ~/.piiguard/policy.yaml \\\n"
            "    --add hash=sha256:abc,category=EMAIL,action=allow,label='dev relay' \\\n"
            "    --add hash=sha256:def,category=PHONE,action=mask\n\n"
            "Exit codes\n"
            "----------\n"
            "  0   change was approved and committed\n"
            "  1   change was rejected / discarded\n"
            "  2   argument error"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    propose_parser.add_argument(
        "--policy-path",
        metavar="PATH",
        default=None,
        help=(
            "path to the policy YAML file "
            "(default: $PIIGUARD_POLICY_PATH)"
        ),
    )
    propose_parser.add_argument(
        "--add",
        metavar="ENTRY",
        type=_parse_pin_list_entry_arg,
        action="append",
        default=[],
        help=(
            "a pin-list entry to include in the proposed list "
            "(key=value,...).  Repeat --add for multiple entries.  "
            "The complete proposed list replaces the current one."
        ),
    )
    propose_parser.set_defaults(func=cmd_pin_list_propose)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the ``piiguard`` CLI.

    Parameters
    ----------
    argv:
        Argument list (default: ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code.  Callers should pass this to ``sys.exit()``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
