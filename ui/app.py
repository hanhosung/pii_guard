"""
PII-Guard UI — Streamlit front-end for interactive PII detection / masking / blocking.

Type a chat message or upload one or more text files; the real PII-Guard
:class:`~pii_guard.engine.Engine` (Stage-1 regex + Stage-2 Korean NER) scans each
input and the result — masked text, per-entity detections, and block decision —
is rendered in a console-style output area AND echoed to the terminal stdout
(the console where ``streamlit run`` is running).

Run:
    cd /Users/ho/workspace/Monoly_genAI/pii_guard
    .venv/bin/python -m streamlit run ui/app.py
    # then open the printed http://localhost:8501 URL

Notes
-----
- Block-category hits (secrets, RRN, card, passport, …) mean the proxy would
  REJECT the whole request (fail-closed); the UI flags the input as BLOCKED.
- Mask-category hits (person, address, org, email, phone, …) are replaced with
  indexed placeholders ([CATEGORY_N]); the input is SAFE-TO-FORWARD.
- Binary / undecodable files are treated as unscannable → BLOCK (fail-closed),
  matching the proxy's secure-by-default policy.
"""
from __future__ import annotations

import sys
from typing import List

import streamlit as st

from pii_guard.engine import Engine
from pii_guard.stage2.runner import Stage2NERRunner

# Allow `streamlit run ui/app.py` to import the sibling scanner module.
sys.path.insert(0, __import__("os").path.dirname(__file__))
from scanner import render_console_block, scan_text, verdict  # noqa: E402


# ── Engine (built once, reused across reruns) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading PII-Guard engine (Presidio + Korean NER)…")
def get_engine(enable_ner: bool) -> Engine:
    if enable_ner:
        return Engine(stage2_runner=Stage2NERRunner())
    return Engine()


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PII-Guard", page_icon="🛡️", layout="wide")
st.title("🛡️ PII-Guard — 개인정보 탐지 · 마스킹 · 차단")
st.caption("채팅 메시지를 입력하거나 파일을 업로드하면 PII를 탐지해 마스킹/차단하고 콘솔에 출력합니다.")

with st.sidebar:
    st.header("설정")
    enable_ner = st.toggle(
        "Stage-2 한국어 NER 사용", value=True,
        help="끄면 정규식(Stage-1)만 — 한국어 이름/주소/조직은 탐지 안 됨",
    )
    st.markdown("---")
    st.markdown("**탐지 카테고리**")
    st.markdown(
        "- 🔴 **차단**: API키, 주민번호, 카드, 여권\n"
        "- 🟡 **마스킹**: 이름·주소·조직(NER), 이메일, 전화, 계좌"
    )
    st.markdown("---")
    st.markdown("**테스트 예시 값**")
    st.code(
        "김민수 / 서울 강남구 테헤란로 123 / 삼성전자\n"
        "minsu@corp.co.kr / 010-1234-5678\n"
        "AKIAIOSFODNN7EXAMPLE (AWS키-차단)\n"
        "710310-4151262 (주민번호-차단)\n"
        "4111-1111-1111-1111 (카드-차단)",
        language="text",
    )

engine = get_engine(enable_ner)

tab_chat, tab_files = st.tabs(["💬 채팅 메시지", "📁 파일 업로드 (다중)"])

console_reports: List[str] = []

# ── Tab 1: chat message ───────────────────────────────────────────────────────
with tab_chat:
    default_msg = (
        "안녕하세요 김민수입니다. 연락처 010-1234-5678, 이메일 minsu@corp.co.kr, "
        "주민번호 710310-4151262, 서울 강남구 테헤란로 123 삼성전자 근무."
    )
    msg = st.text_area("채팅 메시지 입력", value=default_msg, height=140)
    if st.button("🔍 메시지 스캔", type="primary"):
        res = scan_text(engine, msg)
        v, color = verdict(res)
        st.markdown(f"### :{color}[{v}]")
        c1, c2 = st.columns(2)
        c1.text_area("원문", res["original"], height=120)
        c2.text_area("마스킹 결과", res["masked"], height=120)
        if res["rows"]:
            st.dataframe(res["rows"], use_container_width=True)
        report = render_console_block("chat message", res)
        console_reports.append(report)

# ── Tab 2: file upload (multiple) ─────────────────────────────────────────────
with tab_files:
    files = st.file_uploader(
        "텍스트 파일 업로드 (여러 개 가능: .txt .md .json .csv .log .py 등)",
        accept_multiple_files=True,
    )
    if files and st.button("🔍 파일 스캔", type="primary"):
        for f in files:
            raw = f.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Unscannable (binary / non-UTF-8) → fail-closed BLOCK
                st.markdown(f"#### 📄 {f.name}")
                st.markdown(":red[🔴 BLOCKED — 디코드 불가(바이너리) → fail-closed 차단]")
                console_reports.append(
                    "=" * 70 + f"\nINPUT: file {f.name}\n" + "-" * 70 +
                    "\nVERDICT: 🔴 BLOCKED — unscannable (non-UTF-8) → fail-closed\n" +
                    "=" * 70
                )
                continue
            res = scan_text(engine, text)
            v, color = verdict(res)
            st.markdown(f"#### 📄 {f.name}")
            st.markdown(f":{color}[{v}]")
            with st.expander("원문 / 마스킹 비교", expanded=False):
                st.text_area("원문", res["original"], height=120, key=f"orig_{f.name}")
                st.text_area("마스킹", res["masked"], height=120, key=f"mask_{f.name}")
            if res["rows"]:
                st.dataframe(res["rows"], use_container_width=True)
            console_reports.append(render_console_block(f"file {f.name}", res))

# ── Console output (rendered + echoed to terminal stdout) ─────────────────────
if console_reports:
    full = "\n\n".join(console_reports)
    st.markdown("### 🖥️ 콘솔 출력")
    st.code(full, language="text")
    # Echo to the real terminal where `streamlit run` is running.
    print("\n" + full, file=sys.stdout, flush=True)
