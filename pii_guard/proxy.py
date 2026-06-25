"""
PII-Guard HTTP intercepting proxy (Sub-AC 2b-ii).

[이 파일이 하는 일 — 한 줄 요약]
LLM 클라이언트(에이전트)가 외부 LLM으로 보내는 HTTP 요청을 '중간에서 가로채는' 프록시다.
요청 본문을 탐지·마스킹 파이프라인(Engine)에 통과시킨 뒤, 깨끗해진 페이로드만 진짜 업스트림
(api.anthropic.com 등)으로 전달한다. 위험(시크릿 등)하면 아예 차단(400)한다.
응답이 돌아오면 [PLACEHOLDER]를 원래 값으로 복원해서 에이전트에게 돌려준다.

Routes inbound LLM client requests through the PII/secret detection + masking
pipeline, then forwards the sanitised payload to the real upstream LLM endpoint.

Provider routing (path-based)  ← 경로(path)만 보고 어느 LLM인지 판별
------------------------------
  POST /v1/messages                               → Claude (Anthropic Messages API)
  POST /v1/chat/completions                       → OpenAI chat-completions
  POST /v1/completions                            → OpenAI legacy completions
  POST /v1beta/models/*:generateContent           → Gemini generateContent
  POST /v1beta/models/*:streamGenerateContent     → Gemini streaming
  Any other path                                  → pass through unchanged (no scrub)

Blocking
--------
When the scrubber returns ``should_block=True``, the proxy returns HTTP 400 with
a JSON error body and does **NOT** forward the payload to the upstream.

Session state
-------------
A single :class:`~pii_guard.engine.Engine` is used per :class:`PIIGuardProxy`
instance, shared across requests for cross-request placeholder consistency.  The
:attr:`PIIGuardProxy.restoration_map` property exposes the accumulating
``placeholder → original`` mapping for rehydration and test inspection.

Thread safety
-------------
:class:`~pii_guard.engine.Engine` and :class:`~pii_guard.session_map.SessionMap`
are **not** thread-safe.  All inbound requests are serialised through a
``threading.Lock`` on the engine so that concurrent client connections do not
race on session state.  For high-throughput deployments, create one
:class:`PIIGuardProxy` per concurrent session instead.

Usage (context manager)::

    from pii_guard.proxy import PIIGuardProxy

    with PIIGuardProxy("https://api.anthropic.com") as proxy:
        # proxy.base_url == "http://127.0.0.1:<port>"
        # Set ANTHROPIC_BASE_URL=proxy.base_url in your client
        ...

Usage (manual start/stop)::

    proxy = PIIGuardProxy("https://api.anthropic.com", port=4444)
    proxy.start()
    # ... serve requests ...
    proxy.stop()
"""
from __future__ import annotations

import json                                   # JSON 직렬화/파싱
import re                                      # 경로(Gemini) 패턴 매칭용 정규식
import socket                                  # 소켓 옵션(SO_LINGER) 설정용
import struct                                  # SO_LINGER 값을 바이트로 묶을 때 사용
import sys                                      # 콘솔 출력(stdout)용
import threading                                # 동시 요청 직렬화를 위한 락 + 서버 스레드
import urllib.error                             # 업스트림 통신 오류 타입
import urllib.request                           # 업스트림으로 요청 전달(HTTP 클라이언트)
from http.server import BaseHTTPRequestHandler, HTTPServer  # 표준 라이브러리 HTTP 서버
from typing import Any, Dict, Optional, Tuple   # 타입 힌트

from .engine import Engine                      # 탐지·마스킹 엔진(이 프록시의 두뇌)
from .pinlist_guard import (                    # pin-list 변경 차단(에이전트가 자기 화이트리스트 못 하게)
    AGENT_MUTATION_BLOCKED,
    CONTROL_PIN_LIST_PATH,
    MutationSource,
    PinListMutationGuard,
)
from .providers.claude import scrub_claude_request   # Claude 포맷 스크러버
from .providers.gemini import scrub_gemini_request   # Gemini 포맷 스크러버
from .providers.openai import scrub_openai_request   # OpenAI 포맷 스크러버
from .response_rehydrator import ResponsePostProcessor, RehydrationResult  # 비스트리밍 응답 복원
from .streaming_rehydrator import StreamingSSERehydrator                   # 스트리밍(SSE) 응답 복원
from .tripwire import TripwireResult, sweep_raw_body                       # 전체 바디 안전망 스윕


# ─────────────────────────────────────────────────────────────────────────────
# Path routing constants  (경로 → 프로바이더 판별 상수)
# ─────────────────────────────────────────────────────────────────────────────

#: Claude(Anthropic Messages API)를 식별하는 경로
_CLAUDE_PATHS: Tuple[str, ...] = (
    "/v1/messages",
)

#: OpenAI(chat-completions / 레거시 completions)를 식별하는 경로
_OPENAI_PATHS: Tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/completions",
)

#: Gemini(v1beta/v1 generateContent)를 식별하는 정규식 패턴
_GEMINI_PATH_RE = re.compile(
    r"^/v1(?:beta)?/models/[^/?]+:(generateContent|streamGenerateContent)"
)

#: 차단/오류 응답에 쓰는 JSON content-type
_JSON_CONTENT_TYPE = "application/json"

#: 스트리밍(SSE) 응답을 읽을 때 한 번에 읽는 크기(바이트)
_STREAM_CHUNK_SIZE: int = 4096

#: 요청이 차단됐을 때 클라이언트에 돌려주는 본문(미리 만들어 둠)
_BLOCKED_RESPONSE = json.dumps({
    "error": {
        "type": "pii_blocked",
        "message": (
            "PII-Guard: request blocked because PII or a secret was detected "
            "in the payload. Sensitive content was not forwarded to the LLM."
        ),
    }
}).encode("utf-8")

#: pin-list 제어 경로들 — 이 경로로 오는 요청은 무조건 '에이전트發 변경'으로 보고 차단(AGENT_MUTATION_BLOCKED)
_CONTROL_PIN_LIST_PATHS: Tuple[str, ...] = (
    CONTROL_PIN_LIST_PATH,
    "/pii-guard/control/pinlist",       # 다른 표기(붙여쓰기)
    "/pii-guard/control/pin_list",      # 언더스코어 변형
    "/piiguard/control/pin-list",       # 하이픈 없는 패키지 접두 변형
)

#: 요청 본문이 올바른 JSON이 아닐 때 돌려주는 본문
_INVALID_JSON_RESPONSE = json.dumps({
    "error": {
        "type": "invalid_request",
        "message": "PII-Guard: request body must be valid JSON.",
    }
}).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Provider detection helper  (프로바이더 판별 헬퍼)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_provider(path: str) -> Optional[str]:
    """
    경로(path)를 보고 프로바이더 이름을 반환. 못 알아보면 None.
    반환: "claude" | "openai" | "gemini" | None
    """
    # Claude 경로면(쿼리스트링이 붙은 경우도 허용) "claude"
    if any(path == p or path.startswith(p + "?") for p in _CLAUDE_PATHS):
        return "claude"
    # OpenAI 경로면 "openai"
    if any(path == p or path.startswith(p + "?") for p in _OPENAI_PATHS):
        return "openai"
    # Gemini 패턴에 맞으면 "gemini"
    if _GEMINI_PATH_RE.match(path):
        return "gemini"
    return None  # 위 어디에도 안 맞으면 미지의 경로(스크럽 없이 통과 대상)


# ─────────────────────────────────────────────────────────────────────────────
# PIIGuardProxy  (인터셉트 프록시 본체)
# ─────────────────────────────────────────────────────────────────────────────

class PIIGuardProxy:
    """
    Lightweight HTTP proxy that intercepts outbound LLM requests and scrubs
    PII/secrets before forwarding to the upstream endpoint.
    (외부로 나가는 LLM 요청을 가로채 PII/시크릿을 지운 뒤 업스트림으로 보내는 경량 HTTP 프록시.)

    Parameters
    ----------
    upstream_url:
        Base URL of the real LLM upstream, e.g. ``"https://api.anthropic.com"``.
    engine:
        A pre-constructed Engine; if None a fresh one with secure defaults is made.
    host / port:
        Local bind address/port. port=0 → OS가 빈 포트 자동 할당.
    unknown_field_action / unscannable_action:
        스크러버에 넘기는 정책(미지 필드/검사 불가 시 block 등).
    rehydrate_responses:
        응답의 [CATEGORY_N]을 원래 값으로 되돌릴지(기본 True).
    terminal_restore:
        사람이 보는 터미널 출력까지 복원할지(기본 False — 사용자에겐 토큰 유지).
    log_masked:
        업스트림 전송 직전 '마스킹된 페이로드'를 콘솔에 찍을지(기본 False).
    """

    def __init__(
        self,
        upstream_url: str,                          # 진짜 LLM 업스트림 주소
        engine: Optional[Engine] = None,            # (선택) 미리 만든 엔진
        *,                                          # 이 뒤 인자들은 반드시 키워드로 전달
        host: str = "127.0.0.1",                    # 로컬만 바인딩(외부 노출 안 함)
        port: int = 0,                              # 0이면 빈 포트 자동 할당
        unknown_field_action: str = "block",        # 미지 필드 정책(기본 차단)
        unscannable_action: str = "block",          # 검사 불가 정책(기본 차단)
        rehydrate_responses: bool = True,           # 응답 복원 on(기본)
        terminal_restore: bool = False,             # 터미널 복원 off(기본)
        log_masked: bool = False,                   # 마스킹 페이로드 콘솔 출력 off(기본)
    ) -> None:
        self.upstream_url: str = upstream_url.rstrip("/")          # 끝 슬래시 제거(경로 붙일 때 중복 방지)
        self.engine: Engine = engine if engine is not None else Engine()  # 엔진 없으면 기본 엔진 생성
        self._engine_lock = threading.Lock()                       # 엔진은 스레드 안전 X → 락으로 직렬화
        # log_masked=True면 업스트림 전송 직전에 '마스킹된 페이로드'를 stdout에 출력한다.
        # 운영자가 "진짜 LLM에 PII가 평문으로 안 나간다"를 눈으로 확인하기 위함.
        # 단, 마스킹된 페이로드만 찍고 원본 요청은 절대 콘솔에 쓰지 않는다(no-raw-in-logs).
        self._log_masked = log_masked
        self._unknown_field_action = unknown_field_action          # 미지 필드 정책 저장
        self._unscannable_action = unscannable_action              # 검사 불가 정책 저장
        self._rehydrate_responses = rehydrate_responses           # 응답 복원 여부 저장
        self._response_processor = ResponsePostProcessor(terminal_restore=terminal_restore)  # 응답 복원기
        # pin-list 변경 가드 — 제어 엔드포인트로 오는 요청을 '에이전트發'으로 보고 차단.
        self._pin_list_guard = PinListMutationGuard()

        # 마지막 스크럽 결과를 테스트 점검용으로 보관(락으로 보호)
        self._last_scrub_result: Optional[Any] = None
        self._last_scrub_lock = threading.Lock()

        # 마지막 트립와이어 결과 보관(테스트/진단용)
        self._last_tripwire_result: Optional[TripwireResult] = None
        self._last_tripwire_lock = threading.Lock()

        # 마지막 복원 결과 보관(테스트용)
        self._last_rehydration_result: Optional[RehydrationResult] = None
        self._last_rehydration_lock = threading.Lock()

        # 핸들러 클래스 안에서 이 프록시 인스턴스를 참조하려고 별칭을 만든다.
        proxy_ref = self

        class _Handler(BaseHTTPRequestHandler):
            """요청 한 건을 처리하는 HTTP 핸들러(프록시당 내부 클래스로 정의)."""

            # 기본 액세스 로그 출력을 끈다(테스트 출력 깔끔하게 유지)
            def log_message(self, fmt: str, *args) -> None:  # pragma: no cover
                pass

            def do_POST(self) -> None:
                # ── pin-list 제어 엔드포인트 가드 ──
                # pin-list 제어 경로로 온 POST는 무조건 '에이전트發'으로 보고 즉시 차단.
                # 본문도 안 읽고, 상태도 안 바꾼다.
                path_no_qs = self.path.split("?")[0]          # 쿼리스트링 떼고 순수 경로만
                if path_no_qs in _CONTROL_PIN_LIST_PATHS:     # 제어 경로면
                    proxy_ref._handle_pin_list_mutation(self) # 차단 처리
                    return
                proxy_ref._handle_post(self)                  # 그 외 POST는 일반 스크럽 경로로

            def do_GET(self) -> None:
                # 헬스체크: GET /health → 200 OK (프로세스 살아있는지 확인용)
                if self.path == "/health":
                    proxy_ref._send_json(self, 200, {"status": "ok"})
                else:
                    proxy_ref._pass_through(self, "GET")      # 그 외 GET은 통과 처리(현재는 405)

        self._server = HTTPServer((host, port), _Handler)     # 실제 HTTP 서버 생성(핸들러 연결)

        # ── fail-closed 소켓 설정 ──
        # 리스닝 소켓에 SO_LINGER=0을 건다.
        #  - 정상 종료(stop) 시: close가 FIN이 아니라 즉시 RST가 되어, 종료 시작 후 새 연결이 성립하지 못함.
        #  - SIGKILL/크래시 시: OS가 모든 FD를 닫으며 열린 연결마다 RST 전송 → 클라는 연결 끊김 오류를 받지,
        #    '잘못 포워딩된 응답'을 받지 않는다(= 네트워크 레벨 fail-closed).
        try:
            self._server.socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_LINGER,
                struct.pack("ii", 1, 0),  # l_onoff=1, l_linger=0 → close 시 하드 RST
            )
        except OSError:  # pragma: no cover — 플랫폼이 SO_LINGER 미지원일 수 있음
            pass

        # 실제 바인딩된 포트를 읽어둔다(port=0으로 줬을 때 중요)
        _bound_host, _bound_port = self._server.server_address
        self._host = _bound_host
        self._port = _bound_port
        self._thread: Optional[threading.Thread] = None       # 서버 실행 스레드(아직 미시작)

    # ── Properties (읽기 전용 속성) ─────────────────────────────────────────────

    @property
    def host(self) -> str:
        """로컬 바인드 주소(예: "127.0.0.1")."""
        return self._host

    @property
    def port(self) -> int:
        """로컬 바인드 포트(start 후 확정값)."""
        return self._port

    @property
    def base_url(self) -> str:
        """프록시의 전체 주소(예: "http://127.0.0.1:4444"). 클라의 base_url로 쓰면 됨."""
        return f"http://{self._host}:{self._port}"

    @property
    def restoration_map(self) -> Dict[str, str]:
        """현재 '플레이스홀더 → 원본' 매핑의 읽기 전용 스냅샷(복원/테스트 점검용)."""
        return self.engine.restoration_map

    @property
    def terminal_restore(self) -> bool:
        """터미널 출력 복원이 켜졌는지(기본 False — 사용자에겐 [CATEGORY_N] 토큰 유지)."""
        return self._response_processor.terminal_restore

    @property
    def last_tripwire_result(self) -> Optional[TripwireResult]:
        """가장 최근 트립와이어 스윕 결과(없으면 None). 스레드 안전 스냅샷."""
        with self._last_tripwire_lock:
            return self._last_tripwire_result

    @property
    def last_rehydration_result(self) -> Optional[RehydrationResult]:
        """가장 최근 응답 복원 결과(없으면 None). 스레드 안전 스냅샷."""
        with self._last_rehydration_lock:
            return self._last_rehydration_result

    # ── Lifecycle (시작/종료) ───────────────────────────────────────────────────

    def start(self) -> "PIIGuardProxy":
        """프록시를 데몬 스레드로 시작하고 self를 반환(반환 시점에 연결 수락 준비 완료)."""
        self._thread = threading.Thread(
            target=self._server.serve_forever,   # 서버 루프를 백그라운드로
            daemon=True,                         # 메인 종료 시 함께 종료되는 데몬 스레드
            name="pii-guard-proxy",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """프록시 서버를 종료하고 백그라운드 스레드를 정리."""
        self._server.shutdown()                  # serve_forever 루프 중단
        if self._thread is not None:
            self._thread.join(timeout=5)         # 스레드 종료 대기(최대 5초)
            self._thread = None

    def __enter__(self) -> "PIIGuardProxy":      # with 블록 진입 시 자동 start
        return self.start()

    def __exit__(self, *_exc) -> None:           # with 블록 탈출 시 자동 stop
        self.stop()

    # ── Core request handler (핵심 요청 처리) ───────────────────────────────────

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        """
        메인 POST 처리: 요청 본문을 스크럽하고, 차단이 아니면 업스트림으로 전달.
        순서: ①본문 읽기 → ②JSON 파싱 → ③프로바이더 판별+스크럽+트립와이어 → ④차단 or 전달
        """
        # ── 1. 본문 읽기 ──
        try:
            content_length = int(handler.headers.get("Content-Length", 0) or 0)  # 본문 길이
            raw_body = handler.rfile.read(content_length)                        # 그 길이만큼 읽기
        except (ValueError, OSError):
            self._send_json(handler, 400, {"error": "failed to read request body"})  # 읽기 실패 → 400
            return

        # ── 2. JSON 파싱 ──
        try:
            payload: Dict[str, Any] = json.loads(raw_body) if raw_body else {}   # 본문을 dict로
        except (json.JSONDecodeError, ValueError):
            self._send_response_bytes(                                           # JSON 아니면 400
                handler, 400, _INVALID_JSON_RESPONSE, _JSON_CONTENT_TYPE
            )
            return

        # ── 3. 프로바이더 판별 + 스크럽 ──
        path = handler.path
        provider = _detect_provider(path)            # 경로로 claude/openai/gemini 판별

        if provider is not None:                     # 아는 프로바이더면(스크럽 대상)
            scrub_result = self._scrub(payload, provider)   # 해당 포맷 스크러버로 PII 제거
            with self._last_scrub_lock:
                self._last_scrub_result = scrub_result       # 테스트용으로 마지막 결과 보관

            # ── 3b. 전체 바디 트립와이어 스윕 ──
            # 마스킹된 페이로드를 전체로 한 번 더 훑는다. 구조 파서가 안 본 필드에 PII가 남아 있으면
            # 그건 진짜 사각지대 → 트립와이어가 잡는다.
            tripwire_result = self._run_tripwire(scrub_result.sanitized_payload)
            with self._last_tripwire_lock:
                self._last_tripwire_result = tripwire_result

            # 차단 결정 병합: 구조 스크러버 OR 트립와이어 중 하나라도 차단을 요구하면 차단.
            if scrub_result.should_block or tripwire_result.should_block:
                self._log_traffic(path, provider, scrub_result,
                                  tripwire_result, blocked=True)   # (옵션) 차단 로그
                self._send_response_bytes(                          # 400 차단 응답(업스트림 미전달)
                    handler, 400, _BLOCKED_RESPONSE, _JSON_CONTENT_TYPE
                )
                return

            forwarded_payload = scrub_result.sanitized_payload      # 전달할 것은 '마스킹된' 페이로드
            self._log_traffic(path, provider, scrub_result,
                              tripwire_result, blocked=False)        # (옵션) 전달 로그
        else:
            # 모르는 경로 → 그대로 통과(스크럽 안 함)
            forwarded_payload = payload

        # ── 4. 업스트림으로 전달 ──
        self._forward(handler, path, forwarded_payload)

    def _log_traffic(self, path, provider, scrub_result,
                     tripwire_result, *, blocked: bool) -> None:
        """
        (log_masked=True일 때만) 마스킹된 페이로드 + 탐지 요약을 stdout에 출력.
        실제 업스트림 호출 시 'PII가 마스킹/차단된 채 나가는지'를 운영자가 확인하게 해준다.
        원본 요청은 절대 안 찍는다(마스킹된 것만).
        """
        if not self._log_masked:                     # 옵션 꺼져 있으면 아무것도 안 함
            return

        # 구조 스크러버의 필드별 이벤트에서 탐지 목록을 모은다.
        dets = []  # (카테고리, 액션, 플레이스홀더)
        for ev in getattr(scrub_result, "field_events", []) or []:
            for d in getattr(ev, "detections", []) or []:
                action = str(getattr(d, "action", "")).split(".")[-1]
                dets.append((d.category, action, getattr(d, "placeholder_token", "")))

        out = ["", "=" * 72]                         # 출력 버퍼(구분선)
        verdict = "✗ BLOCKED — NOT forwarded (fail-closed)" if blocked \
            else "→ FORWARD to upstream (masked)"    # 차단/전달 판정 문구
        out.append(f"[PII-Guard] {verdict}")
        out.append(f"  upstream : {self.upstream_url}{path}   (provider={provider})")
        if dets:
            out.append(f"  detections ({len(dets)}):")
            for cat, action, ph in dets:             # 카테고리→플레이스홀더 요약(원본값 X)
                mark = "BLOCK" if action == "BLOCK" else "mask "
                out.append(f"    [{mark}] {cat:<13} → {ph}")
        else:
            out.append("  detections: none")
        if getattr(tripwire_result, "should_block", False):
            out.append("  tripwire : BLOCK-category PII found in a non-standard field")
        # 업스트림으로 가는(또는 갈 뻔한) '마스킹된' 페이로드 자체를 출력.
        try:
            masked_json = json.dumps(
                scrub_result.sanitized_payload, ensure_ascii=False, indent=2
            )
        except Exception:  # noqa: BLE001
            masked_json = "<unserialisable>"
        out.append("  masked payload (sent to upstream):" if not blocked
                   else "  masked payload (withheld — shown for inspection):")
        out.append("\n".join("    " + ln for ln in masked_json.splitlines()))
        out.append("=" * 72)
        print("\n".join(out), file=sys.stdout, flush=True)  # 한 번에 콘솔 출력

    def _handle_pin_list_mutation(self, handler: BaseHTTPRequestHandler) -> None:
        """
        pin-list 제어 엔드포인트 요청을 가로채 차단(403).
        에이전트가 자기 유출을 화이트리스트하지 못하게 막는 컨트롤플레인 보호.
        본문도 안 읽고, 상태도 안 바꾸며, 분류 결과만으로 오류 응답을 만든다.
        """
        result = self._pin_list_guard.check(MutationSource.AGENT)  # 출처=에이전트로 분류 → 항상 차단
        error_body = json.dumps(result.as_error_dict(), ensure_ascii=False).encode("utf-8")
        self._send_response_bytes(handler, 403, error_body, _JSON_CONTENT_TYPE)  # 403 + 구조화 오류

    def _scrub(self, payload: Dict[str, Any], provider: str) -> Any:
        """
        프로바이더별 스크러버를 엔진 락 아래에서 실행(동시성 안전).
        반환: 해당 프로바이더의 스크럽 결과 dataclass(sanitized_payload/field_events/should_block 등).
        """
        with self._engine_lock:                      # 엔진/세션맵은 스레드 안전 X → 락으로 한 번에 하나만
            if provider == "claude":
                return scrub_claude_request(
                    payload,
                    self.engine,
                    unknown_field_action=self._unknown_field_action,
                    unscannable_action=self._unscannable_action,
                )
            elif provider == "openai":
                return scrub_openai_request(
                    payload,
                    self.engine,
                    unknown_field_action=self._unknown_field_action,
                    unscannable_action=self._unscannable_action,
                )
            elif provider == "gemini":
                return scrub_gemini_request(
                    payload,
                    self.engine,
                    unknown_field_action=self._unknown_field_action,
                    unscannable_action=self._unscannable_action,
                )
            else:
                raise ValueError(f"Unknown provider: {provider!r}")  # 방어(여기 올 일은 없음)

    def _run_tripwire(self, sanitized_payload: Dict[str, Any]) -> TripwireResult:
        """
        마스킹된 페이로드 전체를 JSON 문자열로 만들어 트립와이어 스윕에 통과.
        구조 파서가 못 본 필드에 PII가 남아 있으면 여기서 잡힌다(사각지대 포착).
        트립와이어 자체가 실패해도 프록시가 죽지 않게, 빈(차단 안 함) 결과로 폴백.
        """
        try:
            sanitized_json = json.dumps(sanitized_payload, ensure_ascii=False)
            return sweep_raw_body(sanitized_json)
        except Exception:  # noqa: BLE001
            # 트립와이어가 깨져도 프록시는 계속 → 구조 스크러버의 결정만 따른다.
            return TripwireResult()

    def _forward(
        self,
        handler: BaseHTTPRequestHandler,
        path: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        *payload*를 JSON으로 만들어 업스트림(upstream_url + path)으로 POST 전달.
        - Content-Length/Host/Transfer-Encoding 빼고 들어온 헤더(인증 등)는 그대로 복사.
        - 응답은 복원(rehydrate) 후 클라이언트로. 스트리밍(SSE) 응답은 별도 경로로 처리.
        """
        forwarded_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")  # 보낼 본문
        upstream_url = self.upstream_url + path      # 최종 업스트림 URL(경로 그대로 붙임)

        req = urllib.request.Request(                # 업스트림으로 보낼 요청 객체
            upstream_url,
            data=forwarded_body,
            method="POST",
        )

        # 들어온 헤더를 나가는 요청에 복사(인증·content-type 등 보존). 단 아래 3개는 재계산.
        _skip_headers = {"content-length", "host", "transfer-encoding"}
        for key, value in handler.headers.items():
            if key.lower() not in _skip_headers:
                req.add_header(key, value)
        req.add_header("Content-Length", str(len(forwarded_body)))  # 새 본문 길이로 재설정
        if not handler.headers.get("Content-Type"):                 # content-type 없으면 기본 지정
            req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:   # 업스트림 호출(30초 타임아웃)
                content_type = resp.headers.get("Content-Type", "")
                is_sse = "text/event-stream" in content_type        # 스트리밍 응답 여부

                if is_sse and self._rehydrate_responses:
                    # ── 스트리밍(SSE) 경로 ── 청크별로 즉시 복원·전달(TTFT 보존)
                    self._forward_streaming(handler, resp, path)
                else:
                    # ── 비스트리밍(버퍼) 경로 ── 전체 받아서 복원 후 전달
                    resp_body = resp.read()
                    resp_body = self._rehydrate_response(resp_body, path)  # [CAT_N]→원본 복원
                    handler.send_response(resp.status)
                    for key, value in resp.headers.items():
                        if key.lower() not in {"transfer-encoding", "connection", "content-length"}:
                            handler.send_header(key, value)
                    handler.send_header("Content-Length", str(len(resp_body)))  # 복원 후 길이로
                    handler.end_headers()
                    handler.wfile.write(resp_body)                  # 클라이언트로 전송
        except urllib.error.HTTPError as exc:        # 업스트림이 4xx/5xx로 응답한 경우
            resp_body = exc.read()
            handler.send_response(exc.code)          # 그 상태코드/본문을 그대로 전달
            for key, value in exc.headers.items():
                if key.lower() not in {"transfer-encoding", "connection"}:
                    handler.send_header(key, value)
            handler.end_headers()
            handler.wfile.write(resp_body)
        except urllib.error.URLError as exc:         # 업스트림 연결 자체 실패
            self._send_json(
                handler, 502,
                {"error": f"upstream connection failed: {exc.reason}"}
            )
        except OSError as exc:                        # 그 외 I/O 오류
            self._send_json(
                handler, 502,
                {"error": f"upstream I/O error: {exc}"}
            )

    def _forward_streaming(
        self,
        handler: BaseHTTPRequestHandler,
        resp: Any,
        path: str,
    ) -> None:
        """
        스트리밍 SSE 응답을 '룩어헤드 복원'과 함께 전달.
        업스트림 스트림을 4KiB씩 읽어 StreamingSSERehydrator에 먹이고, 복원된 출력을
        '즉시' 클라이언트로 쓴다(전체를 기다리지 않음 → TTFT 보존). 복원 안 된 [CAT_N]은 안 나간다.
        """
        provider = _detect_provider(path)            # 스트림 포맷 판별용

        with self._engine_lock:                       # 현재 복원맵 스냅샷을 락 아래에서 복사
            restoration_map = dict(self.engine.restoration_map)

        rehydrator = StreamingSSERehydrator(          # 청크 경계까지 처리하는 스트리밍 복원기
            restoration_map=restoration_map,
            provider=provider,
        )

        # ── 클라이언트로 응답 헤더 먼저 전송(스트리밍이라 Content-Length 없음) ──
        handler.send_response(resp.status)
        for key, value in resp.headers.items():
            if key.lower() not in {
                "transfer-encoding", "connection", "content-length"
            }:
                handler.send_header(key, value)
        handler.send_header("Connection", "close")   # 스트림 끝을 클라가 알도록 close 사용
        handler.end_headers()

        # ── 청크 단위 스트리밍 ──
        # resp.read(n) 대신 read1(n)을 쓰는 이유: read(n)은 n바이트가 다 모일 때까지 블록 →
        # 여러 조각으로 오는 작은 SSE 프레임의 TTFT를 망친다. read1은 소켓 버퍼에 있는 만큼만
        # 한 번 읽어 즉시 돌려줘 청크별 전달이 가능하다.
        raw_reader = getattr(resp, "fp", None)
        use_read1 = raw_reader is not None and hasattr(raw_reader, "read1")

        try:
            while True:
                if use_read1:
                    chunk = raw_reader.read1(_STREAM_CHUNK_SIZE)   # 진짜 스트리밍 읽기
                else:
                    # 폴백: 표준 read(작은 응답에선 EOF까지 블록될 수 있음)
                    chunk = resp.read(_STREAM_CHUNK_SIZE)
                if not chunk:                          # 더 읽을 게 없으면(스트림 끝)
                    break
                output = rehydrator.feed_chunk(chunk)  # 청크를 복원기에 먹임(경계 토큰은 보류)
                if output:                             # 확정된 출력이 있으면
                    handler.wfile.write(output)        # 즉시 클라이언트로
                    handler.wfile.flush()

            # ── 룩어헤드 버퍼 꼬리 비우기 ── 마지막에 남은 보류분 방출
            tail = rehydrator.flush()
            if tail:
                handler.wfile.write(tail)
                handler.wfile.flush()

        except OSError:
            # 클라가 끊었거나 업스트림이 갑자기 닫힘 → 조용히 종료
            pass

    def _rehydrate_response(self, resp_body: bytes, path: str) -> bytes:
        """
        비스트리밍 응답 본문에 복원 적용.
        경로로 프로바이더 판별 → 현재 세션 복원맵 → ResponsePostProcessor로 [CAT_N]→원본.
        복원이 꺼져 있으면(rehydrate_responses=False) 원본 그대로 반환.
        """
        if not self._rehydrate_responses:            # 복원 비활성 → 그대로
            return resp_body

        provider = _detect_provider(path)

        with self._engine_lock:                       # 복원맵은 락 아래에서 접근
            restoration_map = self.engine.restoration_map

        if not restoration_map:                       # 마스킹된 게 없으면 복원할 것도 없음
            return resp_body

        rehydration_result = self._response_processor.process(  # 실제 복원 수행
            response_body=resp_body,
            restoration_map=restoration_map,
            provider=provider,
        )

        with self._last_rehydration_lock:
            self._last_rehydration_result = rehydration_result  # 테스트용 보관

        return rehydration_result.agent_body          # 에이전트에게 줄 복원된 본문

    def _pass_through(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        """미지의/GET 요청 처리(현재는 POST/GET 외엔 405로 응답)."""
        self._send_json(
            handler, 405, {"error": f"method {method} not supported"}
        )

    # ── Response helpers (응답 보조 함수) ───────────────────────────────────────

    @staticmethod
    def _send_json(
        handler: BaseHTTPRequestHandler,
        status: int,
        data: Any,
    ) -> None:
        """dict를 JSON으로 만들어 응답."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        PIIGuardProxy._send_response_bytes(
            handler, status, body, _JSON_CONTENT_TYPE
        )

    @staticmethod
    def _send_response_bytes(
        handler: BaseHTTPRequestHandler,
        status: int,
        body: bytes,
        content_type: str,
    ) -> None:
        """원시 바이트 본문을 주어진 상태코드/콘텐츠타입으로 응답."""
        handler.send_response(status)                       # 상태코드
        handler.send_header("Content-Type", content_type)  # 콘텐츠 타입
        handler.send_header("Content-Length", str(len(body)))  # 길이
        handler.end_headers()
        handler.wfile.write(body)                          # 본문 전송
