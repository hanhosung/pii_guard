"""
PII-Guard pin-list mutation guard (Sub-AC 5d-i).

Classifies incoming pin-list mutation requests as agent-sourced vs.
out-of-band and rejects agent-sourced mutations with a structured error.

Design
------
Pin-list entries control which PII values receive special per-value
treatment (allow, mask, block, tokenize_roundtrip) — they are part of
the **security control plane** and must be modifiable only by a human
user, never by an agent process.

The two source categories::

    AGENT       — any request arriving through the proxy HTTP API or any
                  programmatic code path reachable by an LLM agent.
                  Always blocked with AGENT_MUTATION_BLOCKED.

    OUT_OF_BAND — a direct file-system edit to the policy YAML file,
                  outside the agent's write-permission scope (the control
                  plane directory is user-owned, not agent-writable).
                  Allowed only when ``pin_list_approved: true`` is also
                  present in the same file (explicit user sign-off).

Proxy endpoint guard
---------------------
The proxy exposes ``POST /pii-guard/control/pin-list`` as the canonical
agent-accessible mutation endpoint.  Every request to that path is
immediately classified as ``AGENT`` and rejected.  No pin-list state is
read, written, or modified.

PolicyLoader integration
------------------------
When ``PolicyLoader._try_load()`` detects a pin-list change in the YAML
file (via file-system mtime polling), it invokes the guard with source
``OUT_OF_BAND``.  The guard then checks for ``pin_list_approved: true``
before accepting the new entries.

Error format
------------
Blocked mutations return a JSON-serialisable dict::

    {
        "error": {
            "type": "AGENT_MUTATION_BLOCKED",
            "message": "<human-readable explanation>",
            "source": "agent"
        }
    }

The ``type`` field is always the constant ``AGENT_MUTATION_BLOCKED`` for
agent-sourced mutations.  This is the canonical error type consumed by
tests and callers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public constants
# ─────────────────────────────────────────────────────────────────────────────

#: Structured error type returned for any agent-sourced mutation attempt.
#: This is the canonical string consumed by tests and callers.
AGENT_MUTATION_BLOCKED: str = "AGENT_MUTATION_BLOCKED"

#: Error type returned when an out-of-band change lacks ``pin_list_approved``.
PIN_LIST_NOT_APPROVED: str = "PIN_LIST_NOT_APPROVED"

#: HTTP path prefix that the proxy uses as the agent-accessible control endpoint.
CONTROL_PIN_LIST_PATH: str = "/pii-guard/control/pin-list"


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

class MutationSource(str, Enum):
    """
    Classification of a pin-list mutation request's origin.

    Used as the *source* parameter to :meth:`PinListMutationGuard.check`.
    """

    AGENT = "agent"
    """
    Request originated from an agent-accessible code path.

    This includes any HTTP request to the proxy control endpoint and any
    programmatic API call reachable by an LLM agent process.  All
    mutations from this source are unconditionally blocked.
    """

    OUT_OF_BAND = "out_of_band"
    """
    Request originated from a direct user file-system edit.

    The policy YAML lives outside the agent's write-permission scope.
    Mutations from this source are allowed only when the user also sets
    ``pin_list_approved: true`` in the same file.
    """


@dataclass
class MutationResult:
    """
    Result of a pin-list mutation check by :class:`PinListMutationGuard`.

    Attributes
    ----------
    allowed:
        Whether the mutation may proceed.  ``True`` only for out-of-band
        mutations that carry explicit user approval.
    source:
        Classified origin string — ``"agent"`` or ``"out_of_band"``.
    error_type:
        Structured error type string when ``allowed`` is ``False``.
        One of ``AGENT_MUTATION_BLOCKED`` or ``PIN_LIST_NOT_APPROVED``.
    error_message:
        Human-readable explanation returned to the caller.
    """

    allowed: bool
    source: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def as_error_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable error dict for HTTP responses.

        Format::

            {
                "error": {
                    "type": "<error_type>",
                    "message": "<error_message>",
                    "source": "<source>"
                }
            }
        """
        return {
            "error": {
                "type": self.error_type or "UNKNOWN_ERROR",
                "message": self.error_message or "Mutation blocked.",
                "source": self.source,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# PinListMutationGuard
# ─────────────────────────────────────────────────────────────────────────────

class PinListMutationGuard:
    """
    Interceptor that classifies pin-list mutation requests and rejects
    agent-sourced mutations with a structured ``AGENT_MUTATION_BLOCKED`` error.

    Control-plane isolation guarantee
    -----------------------------------
    No agent-accessible code path can modify the pin-list.  The only
    legitimate mutation path is:

    1. The **user** edits the policy YAML on disk (out-of-band relative
       to the agent process, in a directory the agent cannot write).
    2. The **user** also sets ``pin_list_approved: true`` in the same file
       (explicit approval that requires out-of-band access to the control-
       plane directory).
    3. The :class:`~pii_guard.policy.PolicyLoader` watcher detects the
       change, calls this guard with ``OUT_OF_BAND``, and applies the new
       pin-list only when ``approved=True``.

    Any request arriving via the proxy API, any programmatic call, or any
    code path reachable by an LLM agent is classified as ``AGENT`` and is
    immediately blocked — **no state read, no state write, no side effect**.

    Usage::

        guard = PinListMutationGuard()

        # Agent-sourced request (always blocked):
        result = guard.check(MutationSource.AGENT)
        assert not result.allowed
        assert result.error_type == AGENT_MUTATION_BLOCKED

        # Out-of-band file edit without approval (blocked):
        result = guard.check(MutationSource.OUT_OF_BAND, approved=False)
        assert not result.allowed
        assert result.error_type == PIN_LIST_NOT_APPROVED

        # Out-of-band file edit with approval (accepted):
        result = guard.check(MutationSource.OUT_OF_BAND, approved=True)
        assert result.allowed
    """

    def check(
        self,
        source: MutationSource,
        approved: bool = False,
    ) -> MutationResult:
        """
        Evaluate whether a pin-list mutation should be allowed.

        Parameters
        ----------
        source:
            The classified origin of the mutation request.

            Pass :attr:`MutationSource.AGENT` for any programmatic request
            from the proxy API or agent code path.

            Pass :attr:`MutationSource.OUT_OF_BAND` when the change was
            detected in the policy YAML file on disk (file-system watcher).

        approved:
            Whether the user set ``pin_list_approved: true`` in the policy
            file for this reload cycle.  Only relevant for
            ``OUT_OF_BAND`` mutations; ignored for ``AGENT`` (always blocked).

        Returns
        -------
        MutationResult
            ``allowed=True`` only for out-of-band mutations with
            ``approved=True``.  ``allowed=False`` for all agent-sourced
            mutations and for unapproved out-of-band mutations.
        """
        if source == MutationSource.AGENT or source == MutationSource.AGENT.value:
            log.warning(
                "PII-Guard pin-list guard: mutation BLOCKED — "
                "request classified as agent-sourced (source=%r). "
                "Pin-list changes must be made out-of-band by editing "
                "the policy YAML directly with pin_list_approved: true.",
                source if isinstance(source, str) else source.value,
            )
            return MutationResult(
                allowed=False,
                source=MutationSource.AGENT.value,
                error_type=AGENT_MUTATION_BLOCKED,
                error_message=(
                    "Pin-list mutations via the agent API are not permitted. "
                    "To modify the pin-list, edit the policy YAML file "
                    "directly (outside the agent process) and set "
                    "'pin_list_approved: true' after reviewing the changes. "
                    "This restriction ensures pin-list changes require "
                    "explicit out-of-band user approval."
                ),
            )

        # ── Out-of-band source ────────────────────────────────────────────────
        if not approved:
            log.warning(
                "PII-Guard pin-list guard: mutation BLOCKED — "
                "out-of-band change detected but 'pin_list_approved' is "
                "not set to true. Set 'pin_list_approved: true' in the "
                "policy file after reviewing the new pin-list entries."
            )
            return MutationResult(
                allowed=False,
                source=MutationSource.OUT_OF_BAND.value,
                error_type=PIN_LIST_NOT_APPROVED,
                error_message=(
                    "Pin-list change detected but 'pin_list_approved' is "
                    "not set to true. Set 'pin_list_approved: true' in "
                    "the policy file after reviewing the changes."
                ),
            )

        log.info(
            "PII-Guard pin-list guard: mutation APPROVED — "
            "out-of-band change with explicit user approval accepted."
        )
        return MutationResult(
            allowed=True,
            source=MutationSource.OUT_OF_BAND.value,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

#: Default module-level guard instance — safe for concurrent use since
#: :meth:`PinListMutationGuard.check` carries no mutable state.
DEFAULT_GUARD: PinListMutationGuard = PinListMutationGuard()


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def classify_source(is_agent_request: bool) -> MutationSource:
    """
    Map a boolean origin flag to a :class:`MutationSource` enum value.

    Parameters
    ----------
    is_agent_request:
        ``True`` if the request came via the proxy API or any programmatic
        path reachable by an LLM agent process.
        ``False`` if the request came from a file-system edit (out-of-band).

    Returns
    -------
    MutationSource
        :attr:`MutationSource.AGENT` when *is_agent_request* is ``True``,
        :attr:`MutationSource.OUT_OF_BAND` otherwise.
    """
    return MutationSource.AGENT if is_agent_request else MutationSource.OUT_OF_BAND
