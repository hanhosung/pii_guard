"""
PII-Guard Stage-1 text scanning engine — top-level API.

[이 파일이 하는 일 — 한 줄 요약]
탐지의 '지휘자(오케스트레이터)'이자 외부에 노출되는 공용 진입점이다.
텍스트를 받아 ① Stage1(정규식·체크섬) → ② Stage1.5(proximity) → ③ Stage2(NER)를
순서대로 돌리고, 그 결과로 마스킹/차단된 텍스트(RedactionResult)를 돌려준다.
또한 LLM 응답에 들어온 플레이스홀더를 원래 값으로 되돌리는 복원(rehydrate)도 제공한다.

Usage::

    from pii_guard.engine import Engine

    engine = Engine()
    result = engine.scan("Send the report to alice@example.com")
    print(result.redacted_text)   # "Send the report to [EMAIL_1]"
    print(result.summary())       # {"total_detections": 1, "categories": {...}, ...}

    # Rehydrate an inbound LLM response (round-trip restoration):
    restored = engine.rehydrate(llm_response_text)

Session-level state (placeholder counters and the restoration map) is held
inside the :class:`SessionMap` owned by this :class:`Engine` instance so the
same real value always produces the same placeholder within a session.
"""
from __future__ import annotations

import re                                         # 정규식(allowlist 패턴 컴파일 등)
from typing import TYPE_CHECKING, Dict, List, Optional  # 타입 힌트

from .categories import ALL_CATEGORIES, CategorySpec   # 18+개 카테고리 정의(기본 탐지 규칙 집합)
from .detector import scan_text                        # Stage1 실행 함수(정규식·체크섬 탐지)
from .masker import apply_redactions, rehydrate_text   # 마스킹 적용 / 복원 함수
from .models import RedactionResult                    # 스캔 결과 타입
from .proximity import DEFAULT_PROXIMITY_CONFIG, ProximityConfig  # proximity 기본 설정/설정 타입
from .proximity import merge as proximity_merge        # proximity 결과를 Stage1에 합치는 함수
from .proximity import scan as proximity_scan          # proximity 탐지 함수(Stage1.5)
from .session_map import SessionMap                    # 원본↔플레이스홀더 매핑(세션 메모리)

if TYPE_CHECKING:                                  # 타입 검사 시에만 import(런타임 순환참조/무거운 로딩 방지)
    from .stage2.runner import Stage2NERRunner     # Stage2 NER 워커(실제 import는 호출 시점에)


class Engine:
    """
    Stateful scanning engine for a single session.
    (한 세션을 담당하는, 상태를 가진 스캔 엔진. 같은 값은 세션 내내 같은 플레이스홀더로 매핑된다.)

    Parameters
    ----------
    categories:
        Override the default category list.
    allowlist_patterns:
        Compiled regex patterns; matches are skipped (project allow-list).
    min_confidence_override:
        Hard minimum confidence; rules below this are ignored.
    hmac_key:
        Secret bytes for keyed-hash ledger correlation.  A random key is
        generated at startup and never persisted.
    stage2_runner:
        Optional :class:`~pii_guard.stage2.runner.Stage2NERRunner` instance.
        When provided, Stage-2 NER is attempted after Stage-1 for each block.
        On Stage-2 failure the engine degrades gracefully to Stage-1 results
        and sets ``coverage_gap=True`` with ``stage2_gap_reason`` on the result.
        Pass ``None`` (default) to run Stage-1 only.
    """

    def __init__(
        self,
        categories: Optional[List[CategorySpec]] = None,       # (선택) 카테고리 목록 교체
        allowlist_patterns: Optional[List[re.Pattern]] = None, # (선택) 허용목록 정규식(이건 PII 아님으로 건너뜀)
        min_confidence_override: Optional[float] = None,       # (선택) 신뢰도 하한(이 미만 규칙 무시)
        hmac_key: Optional[bytes] = None,                      # (선택) Ledger keyed-hash용 비밀키
        stage2_runner: Optional["Stage2NERRunner"] = None,     # (선택) Stage2 NER 워커. None이면 Stage1만
        proximity_enabled: bool = True,                        # proximity 켜기(기본 on)
        proximity_config: Optional[ProximityConfig] = None,    # (선택) 정책에서 온 proximity 설정
        ner_backend: Optional[str] = None,                     # (선택) Stage2 NER 백엔드(정책값). env가 우선
    ) -> None:
        import os                                  # 환경변수 설정/난수 키 생성용
        self._categories = categories or ALL_CATEGORIES   # 안 주면 기본 전체 카테고리 사용
        self._allowlist = allowlist_patterns or []         # 안 주면 빈 허용목록(아무것도 안 건너뜀)
        self._min_confidence = min_confidence_override     # 신뢰도 하한(없으면 None)
        self._hmac_key: bytes = hmac_key or os.urandom(32) # 키 없으면 32바이트 무작위 키 생성(디스크 저장 안 함)

        # Stage2 NER 워커(서브프로세스 격리). None이면 NER 미사용(Stage1만 동작).
        self._stage2_runner = stage2_runner

        # Stage1.5 양성 proximity 설정. 정책(proximity:)에서 오거나, 없으면 안전한 내장 기본값.
        self._proximity_config = proximity_config or DEFAULT_PROXIMITY_CONFIG
        # 실제 활성 여부 = 파라미터 on AND 설정의 enabled on (둘 다 켜져야 동작)
        self._proximity_enabled = proximity_enabled and self._proximity_config.enabled

        # 음성 proximity(NER 오탐 필터) 설정을 환경변수로 내보낸다.
        # NER은 나중에 spawn되는 별도 프로세스라, 환경변수를 통해 설정을 물려준다.
        cfg = self._proximity_config
        # 항상 두 값을 (재)설정한다 → 이전 Engine이 남긴 옛 값이 새 Engine으로 새지 않도록.
        os.environ["PIIGUARD_NER_FILTER_OFF"] = "" if cfg.ner_filter_enabled else "1"  # 필터 끄면 "1"
        os.environ["PIIGUARD_NER_EXTRA_STOPWORDS"] = ",".join(cfg.ner_extra_stopwords) # 추가 deny-list를 콤마로

        # Stage2 NER 백엔드 선택(R18)도 워커 서브프로세스로 env를 통해 전파한다.
        # 우선순위 env > 정책 > 기본('gliner') — 이미 env가 있으면 사용자/CI 오버라이드를 존중하고,
        # 없으면 정책값(ner_backend)을, 그것도 없으면 기본값을 env에 확정해 워커가 동일하게 읽게 한다.
        from .stage2.backend import ENV_NER_BACKEND, resolve_ner_backend  # 경량 모듈(무거운 의존 없음)
        # resolve가 알 수 없는 값이면 여기서 즉시 ValueError → serve 시작 시점에 명확히 실패(P3).
        os.environ[ENV_NER_BACKEND] = resolve_ner_backend(ner_backend).value

        # 세션 단위 가변 상태(절대 디스크에 안 씀). 플레이스홀더 번호 부여는 모두 SessionMap이 담당.
        self._session_map = SessionMap()

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, text: str) -> RedactionResult:
        """
        Scan *text* and return a RedactionResult with the redacted version
        and full detection metadata.
        (*text*를 스캔해, 마스킹된 텍스트 + 탐지 상세가 담긴 RedactionResult를 반환.)

        Detection pipeline
        ------------------
        1. **Stage 1** (always): regex + checksum scanning via
           :func:`~pii_guard.detector.scan_text`.
        2. **Stage 2** (optional): NER via the subprocess runner set at
           construction.  On any Stage-2 failure the engine falls back to
           Stage-1 results, sets ``result.coverage_gap = True``, and records
           the failure in ``result.stage2_gap_reason``.

        Returns
        -------
        RedactionResult
        """
        if not isinstance(text, str):              # 방어: 문자열이 아니면
            raise TypeError(f"scan() expects str, got {type(text).__name__}")  # 명확한 오류 발생

        # ── Stage 1: 정규식 / 체크섬 ──────────────────────────────────────────
        stage1_detections = scan_text(             # 정규식·체크섬으로 1차 탐지 실행
            text,
            categories=self._categories,           # 사용할 카테고리 집합
            allowlist_patterns=self._allowlist,    # 허용목록(이건 PII 아님으로 제외)
            min_confidence_override=self._min_confidence,  # 신뢰도 하한
        )

        # ── Stage 1.5: 양성 proximity(문맥 기반 정형 PII 승격) ────────────────
        # 애매한 포맷(비표준 계좌·맨 사업자번호·한글 비번 라벨)을 '트리거가 근처에 있을 때만' 승격.
        # Stage1 결과에 합쳐 넣으므로(겹치면 정리됨) 이후 Stage2도 이 결과를 본다.
        if self._proximity_enabled:                # proximity가 켜져 있으면
            stage1_detections = proximity_merge(   # 기존 결과 + proximity 결과를 병합
                stage1_detections, proximity_scan(text, self._proximity_config)
            )

        # ── Stage 2: NER (별도 프로세스, 선택) ────────────────────────────────
        final_detections = stage1_detections       # 기본은 Stage1(+1.5) 결과
        stage2_gap_reason: Optional[str] = None     # Stage2가 실패하면 그 사유를 담을 변수

        if self._stage2_runner is not None:        # NER 워커가 설정돼 있으면
            s2 = self._stage2_runner.scan(text, stage1_detections)  # NER 실행(Stage1 결과를 넘겨 병합)
            if s2.coverage_gap:                    # NER이 실패(타임아웃/OOM 등)했으면
                # 실패 → Stage1 결과 유지 + 사각지대(gap) 사유 기록
                stage2_gap_reason = s2.fail_reason
                # final_detections는 Stage1 그대로 둠(안전한 바닥)
            else:                                  # NER 성공 시
                # 성공 → Stage1+Stage2 병합 결과 사용
                final_detections = s2.detections

        # ── 마스킹 적용 ───────────────────────────────────────────────────────
        result = apply_redactions(                 # 탐지된 스팬들을 실제로 치환/처리
            text,
            final_detections,
            session_map=self._session_map,         # 같은 값→같은 토큰 보장을 위해 세션 맵 사용
        )

        # Stage2가 실패했다면 결과에 사각지대 정보를 표시(침묵 통과 금지 — 가시화)
        if stage2_gap_reason is not None:
            result.coverage_gap = True             # "여기 검사 못 한 부분 있음" 플래그
            result.stage2_gap_reason = stage2_gap_reason  # 그 사유

        return result                              # 최종 결과 반환

    def rehydrate(self, text: str) -> str:
        """
        Replace [PLACEHOLDER] tokens in an inbound LLM response with the
        original values from this session's restoration map.
        (LLM 응답 속 [플레이스홀더]를 이 세션의 복원맵으로 원래 값으로 되돌린다.)

        NOTE: terminal output restoration must remain OFF — the proxy calls
        this only for agent-visible content, never for user-visible output.
        (주의: 사람이 보는 터미널 출력 복원은 OFF여야 함 — 에이전트가 보는 콘텐츠에만 호출.)
        """
        return self._session_map.rehydrate(text)   # 복원은 세션 맵에 위임

    @property
    def session_map(self) -> SessionMap:
        """Direct access to the underlying :class:`SessionMap` for this session."""
        return self._session_map                   # 이 세션의 SessionMap 직접 접근(읽기용)

    @property
    def restoration_map(self) -> Dict[str, str]:
        """Read-only snapshot of the current session's placeholder→original map."""
        return self._session_map.restoration_map   # 현재 '플레이스홀더→원본' 매핑의 읽기 전용 스냅샷

    def reset_session(self) -> None:
        """Clear per-session state (counters and restoration map)."""
        self._session_map.reset()                  # 세션 상태(번호 카운터·복원맵) 초기화

    def add_allowlist(self, pattern: str, flags: int = 0) -> None:
        """Add a regex pattern string to the project allow-list."""
        self._allowlist.append(re.compile(pattern, flags))  # 허용목록에 정규식 추가(이 패턴은 PII 아님 처리)
