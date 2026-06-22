"""
Process-launch wrapper that auto-injects PII-Guard proxy environment variables
into every child process spawned by ouroboros workflows or LLM CLIs.

Sub-AC 6a — Auto-inject base_url into ouroboros-spawned processes
-----------------------------------------------------------------

Supported provider environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provider          | Env vars injected
------------------|---------------------------------------------------
Anthropic/Claude  | ANTHROPIC_BASE_URL
OpenAI            | OPENAI_BASE_URL, OPENAI_API_BASE
Google/Gemini     | GEMINI_BASE_URL, GOOGLE_GENAI_BASE_URL

All vars are set to the local PII-Guard proxy address (default
``http://127.0.0.1:4444``).  The proxy address can be overridden via
``PIIGUARD_HOST`` and ``PIIGUARD_PORT`` env vars, or passed directly to
``ProcessLauncher(proxy_base_url=...)``.

Override semantics
~~~~~~~~~~~~~~~~~~
By default, env vars that are **already set** in the parent process
environment are **not overwritten** — a developer who has deliberately
pointed a var at their own testing proxy keeps that setting.  Pass
``allow_override=True`` to force-set all vars regardless.

Usage (direct subprocess wrapper)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from pii_guard.launcher import ProcessLauncher

    launcher = ProcessLauncher()                      # reads PIIGUARD_HOST/PORT
    result = launcher.run(["my-llm-cli", "--flag"])   # env vars injected

Usage (context-manager — patches os.environ for the current process)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    launcher = ProcessLauncher()
    with launcher.env_patch():
        # Any subprocess.Popen / os.exec / subprocess.run call inside here
        # inherits the proxy env vars via os.environ automatically.
        subprocess.run(["codex", "--prompt", "hello"])

Usage (build_proxy_env — standalone helper for custom launchers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from pii_guard.launcher import build_proxy_env

    env = build_proxy_env("http://127.0.0.1:4444", base_env=dict(os.environ))
    subprocess.Popen(cmd, env=env)

Threat model note
~~~~~~~~~~~~~~~~~
This is the *cooperative gateway* enforcement tier: it relies on the
child process honouring the env vars.  A compromised child with a
hard-coded base URL or root privileges can bypass this.  For stronger
enforcement, use the egress-lockdown tier (out of scope for this module).
"""
from __future__ import annotations

import contextlib
import os
import subprocess
from typing import Dict, Iterator, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Default proxy host (overridden by PIIGUARD_HOST env var)
DEFAULT_PROXY_HOST: str = "127.0.0.1"

#: Default proxy port (overridden by PIIGUARD_PORT env var)
DEFAULT_PROXY_PORT: int = 4444

#: Mapping from provider name → list of env var names to inject.
#: Ordered: primary var first, then legacy / alternative vars.
PROVIDER_ENV_VARS: Dict[str, List[str]] = {
    "anthropic": [
        "ANTHROPIC_BASE_URL",       # Anthropic Python SDK ≥ 0.20
    ],
    "openai": [
        "OPENAI_BASE_URL",          # OpenAI Python SDK v1+
        "OPENAI_API_BASE",          # OpenAI SDK <v1 / LangChain / LiteLLM
    ],
    "gemini": [
        "GEMINI_BASE_URL",          # Ouroboros workflow convention
        "GOOGLE_GENAI_BASE_URL",    # google-genai Python SDK convention
    ],
}

#: Flat list of all env var names that will be injected, in order.
ALL_PROXY_ENV_VARS: List[str] = [
    var
    for provider_vars in PROVIDER_ENV_VARS.values()
    for var in provider_vars
]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper
# ─────────────────────────────────────────────────────────────────────────────

def build_proxy_env(
    proxy_base_url: str,
    base_env: Optional[Dict[str, str]] = None,
    *,
    allow_override: bool = False,
) -> Dict[str, str]:
    """
    Return a copy of *base_env* with PII-Guard proxy env vars injected.

    Parameters
    ----------
    proxy_base_url:
        Full URL of the local PII-Guard proxy, e.g. ``"http://127.0.0.1:4444"``.
    base_env:
        Starting environment dict.  Defaults to ``os.environ`` if *None*.
    allow_override:
        If ``True``, overwrite vars that already exist in *base_env*.
        If ``False`` (default), leave pre-existing vars unchanged so that
        a developer's custom proxy setting is respected.

    Returns
    -------
    dict
        New environment dict (the original *base_env* is not mutated).
    """
    env: Dict[str, str] = dict(base_env if base_env is not None else os.environ)

    for var in ALL_PROXY_ENV_VARS:
        if allow_override or var not in env:
            env[var] = proxy_base_url

    return env


# ─────────────────────────────────────────────────────────────────────────────
# ProcessLauncher class
# ─────────────────────────────────────────────────────────────────────────────

class ProcessLauncher:
    """
    Subprocess wrapper that automatically injects PII-Guard proxy env vars into
    every child process, enforcing the cooperative-gateway interception tier.

    Parameters
    ----------
    proxy_base_url:
        Full URL of the local PII-Guard proxy.  If *None*, the URL is
        constructed from the ``PIIGUARD_HOST`` and ``PIIGUARD_PORT`` env vars
        (falling back to ``http://127.0.0.1:4444``).
    allow_override:
        If ``True``, proxy env vars overwrite pre-existing values in the child
        environment.  Default ``False`` respects the developer's custom setting.
    """

    def __init__(
        self,
        proxy_base_url: Optional[str] = None,
        *,
        allow_override: bool = False,
    ) -> None:
        if proxy_base_url is None:
            host = os.environ.get("PIIGUARD_HOST", DEFAULT_PROXY_HOST)
            port = int(os.environ.get("PIIGUARD_PORT", DEFAULT_PROXY_PORT))
            proxy_base_url = f"http://{host}:{port}"
        self.proxy_base_url: str = proxy_base_url
        self.allow_override: bool = allow_override

    # ── Internal helper ───────────────────────────────────────────────────────

    def _make_env(self, caller_env: Optional[Dict[str, str]]) -> Dict[str, str]:
        """
        Build the child-process environment.

        If the caller passed an explicit ``env=`` kwarg to one of the subprocess
        wrappers, merge into *that* dict.  Otherwise merge into the current
        process's ``os.environ``.
        """
        return build_proxy_env(
            self.proxy_base_url,
            base_env=caller_env,
            allow_override=self.allow_override,
        )

    def _inject_env(self, kwargs: dict) -> dict:
        """
        Mutate *kwargs* in-place: set/replace the ``env`` key with a dict that
        has proxy vars injected.  Returns the same dict for convenience.
        """
        kwargs["env"] = self._make_env(kwargs.get("env"))
        return kwargs

    # ── Subprocess wrappers ───────────────────────────────────────────────────

    def popen(self, *args, **kwargs) -> subprocess.Popen:
        """
        Wrapper around :func:`subprocess.Popen` with proxy env vars injected.

        All positional and keyword arguments are forwarded unchanged except that
        the ``env`` kwarg is augmented with the proxy env vars.
        """
        return subprocess.Popen(*args, **self._inject_env(kwargs))

    def run(self, *args, **kwargs) -> subprocess.CompletedProcess:
        """
        Wrapper around :func:`subprocess.run` with proxy env vars injected.
        """
        return subprocess.run(*args, **self._inject_env(kwargs))

    def call(self, *args, **kwargs) -> int:
        """
        Wrapper around :func:`subprocess.call` with proxy env vars injected.
        """
        return subprocess.call(*args, **self._inject_env(kwargs))

    def check_output(self, *args, **kwargs) -> bytes:
        """
        Wrapper around :func:`subprocess.check_output` with proxy env vars injected.
        """
        return subprocess.check_output(*args, **self._inject_env(kwargs))

    # ── Context manager ───────────────────────────────────────────────────────

    @contextlib.contextmanager
    def env_patch(self) -> Iterator[None]:
        """
        Context manager that patches :data:`os.environ` for the current process
        so that *all* child processes spawned inside the block (via any
        mechanism — ``subprocess``, ``os.exec*``, ``multiprocessing``, etc.)
        inherit the proxy env vars.

        On exit the original values (or absence) of each var are restored so
        that the patch is strictly scoped to the ``with`` block.

        Example::

            launcher = ProcessLauncher()
            with launcher.env_patch():
                subprocess.run(["codex", "--prompt", "summarise this"])
                # codex sees ANTHROPIC_BASE_URL=http://127.0.0.1:4444
        """
        saved: Dict[str, Optional[str]] = {
            var: os.environ.get(var) for var in ALL_PROXY_ENV_VARS
        }
        try:
            for var in ALL_PROXY_ENV_VARS:
                if self.allow_override or var not in os.environ:
                    os.environ[var] = self.proxy_base_url
            yield
        finally:
            for var, old_val in saved.items():
                if old_val is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = old_val

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def provider_vars(self) -> Dict[str, List[str]]:
        """Mapping of provider name → list of env var names (read-only copy)."""
        return {k: list(v) for k, v in PROVIDER_ENV_VARS.items()}

    @property
    def all_vars(self) -> List[str]:
        """Flat list of all env var names this launcher injects."""
        return list(ALL_PROXY_ENV_VARS)

    def __repr__(self) -> str:
        return (
            f"ProcessLauncher("
            f"proxy_base_url={self.proxy_base_url!r}, "
            f"allow_override={self.allow_override})"
        )
