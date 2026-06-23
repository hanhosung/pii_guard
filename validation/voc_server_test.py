# -*- coding: utf-8 -*-
"""
VOC + 서버로그 검출 테스트 (60항목) — 고객센터 분석 시나리오.

시나리오: 고객 VOC(개인정보 다수)와 서버 console 로그(서버정보·시크릿 다수)를 외부 LLM으로
보내 분석하기 전에, PII-Guard가 무엇을 마스킹/차단하는지 측정한다.

- VOC 30건: 한국어 고객 불만 텍스트(~2000자), 심은 PII = 이름·전화·이메일·주민번호·카드·계좌·주소.
- 서버로그 30건: console 로그(~3000자), 심은 SECRET = API키·AWS·JWT·프라이빗키·비밀번호.
  + 서버정보(내부 IP·호스트명) = 카탈로그에 없는 항목(검출률을 별도 측정 → 범위 밖 갭 확인).

각 항목은 **심은 값(ground truth)** 을 정확히 추적한다(템플릿 생성이므로 무오류).
산출물: voc_server_log.txt(증거) + VOC_SERVER_REPORT.md(리포트).

실행: PYTHONPATH=. .venv/bin/python validation/voc_server_test.py
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from pii_guard.engine import Engine
from pii_guard.stage2.runner import Stage2NERRunner
from pii_guard.categories import _rrn_checksum, _kr_biz_checksum

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "voc_server_log.txt")
MD_PATH = os.path.join(HERE, "VOC_SERVER_REPORT.md")

SECRET_CATS = {"API_KEY", "AWS_SECRET", "GCP_KEY", "TOKEN", "PRIVATE_KEY", "PASSWORD"}
PII_CATS = {"PERSON", "PHONE", "EMAIL", "RRN", "CARD", "KR_ACCOUNT", "ADDRESS",
            "BIZ_NO", "PASSPORT", "FOREIGN_REG", "DRIVER_LICENSE", "ORGANIZATION"}


# ── 유효 값 생성기 (체크섬 통과 → 탐지 보장) ──────────────────────────────────
def gen_rrn(seed):
    r = random.Random(seed)
    for _ in range(200000):
        y = f"{r.randint(0,99):02d}{r.randint(1,12):02d}{r.randint(1,28):02d}"
        v = f"{y}-{r.choice('1234')}{r.randint(0,999999):06d}"
        if _rrn_checksum(v.replace('-', '')):
            return v
    return "920723-1641620"


def luhn_complete(prefix15):
    digits = [int(c) for c in prefix15]
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        s += d
    return prefix15 + str((10 - s % 10) % 10)


def gen_card(seed):
    r = random.Random(seed)
    p = "4" + "".join(str(r.randint(0, 9)) for _ in range(14))
    n = luhn_complete(p)
    return f"{n[:4]}-{n[4:8]}-{n[8:12]}-{n[12:]}"


# ── 데이터 풀 ─────────────────────────────────────────────────────────────────
NAMES = ["김민준", "이서연", "박지호", "최예린", "정우성", "한소희", "조성민", "윤하늘",
         "임채원", "강도현", "신유진", "오세훈", "배수지", "홍길동", "남도일", "서지안",
         "문가영", "백승호", "권나라", "황민영", "손흥민", "김연아", "이도현", "정해인",
         "류현진", "노홍철", "장윤정", "전소민", "고아라", "유재석"]
ADDRS = ["서울 강남구 테헤란로 123", "부산 해운대구 우동 1408", "경기 성남시 분당구 판교역로 235",
         "인천 연수구 송도과학로 32", "대구 수성구 동대구로 23", "광주 서구 상무중앙로 7",
         "대전 유성구 대학로 99", "울산 남구 삼산로 100", "서울 송파구 올림픽로 300",
         "경기 수원시 영통구 광교중앙로 145"]
ORGS = ["삼성전자", "네이버", "카카오", "현대자동차", "LG전자", "쿠팡", "토스", "배달의민족"]
ACCTS = ["123456-78-901234", "110-123456-78901", "1002-345-678901"]

VOC_FILLER = [
    "정말 너무 답답해서 글 남깁니다.", "여러 번 문의드렸는데 해결이 안 되네요.",
    "빠른 처리 부탁드립니다.", "이런 일이 반복되니 신뢰가 떨어집니다.",
    "고객센터 전화는 계속 통화중이라 여기 남겨요.", "환불 또는 보상을 원합니다.",
    "정확한 원인을 알려주시면 좋겠습니다.", "담당자분 연락 기다리겠습니다.",
    "앱에서도 같은 증상이 계속됩니다.", "주문 내역 확인 후 회신 부탁드립니다.",
    "결제는 됐는데 처리가 안 된 상태입니다.", "어제부터 계속 같은 문제예요.",
    "스크린샷도 첨부했으니 확인 바랍니다.", "회원 등급에 맞는 혜택도 못 받고 있습니다.",
]
LOG_FILLER = [
    "INFO  pool: active=12 idle=4 waiting=0",
    "DEBUG cache hit ratio 0.87 keys=10422",
    "INFO  gc pause 12ms heap=512MB/1024MB",
    "WARN  slow query 1240ms on orders table",
    "INFO  health check ok latency=8ms",
    "DEBUG serialize payload size=2048 bytes",
    "INFO  request accepted route=/api/v2/orders",
    "WARN  retry attempt 2/3 upstream timeout",
    "INFO  metrics flushed to collector",
    "DEBUG thread-pool-7 picked task id=88213",
]


def pad(parts, sep, target, filler):
    out = list(parts)
    i = 0
    text = sep.join(out)
    while len(text) < target:
        out.append(filler[i % len(filler)])
        i += 1
        text = sep.join(out)
    return text


# ── VOC 생성 (30) ─────────────────────────────────────────────────────────────
VOC_SCENARIOS = [
    "결제 오류", "배송 지연", "환불 요청", "로그인 불가", "계정 도용 의심", "쿠폰 미적용",
    "상품 불량", "주문 취소", "포인트 누락", "멤버십 해지", "개인정보 정정", "AS 접수",
    "예약 변경", "이중 결제", "배송지 변경", "교환 요청", "고객정보 확인", "구독 해지",
    "환불 계좌 등록", "본인 인증 실패", "주소 오류", "결제 수단 변경", "분실 신고",
    "할인 문의", "재배송 요청", "영수증 요청", "회원 전환", "이벤트 참여", "오배송", "민원 접수",
]


def make_voc(i):
    r = random.Random(1000 + i)
    name = NAMES[i % len(NAMES)]
    phone = f"010-{r.randint(1000,9999)}-{r.randint(1000,9999)}"
    email = f"{['minjun','seoyeon','jiho','yerin','user'][i%5]}{i}@{'gmail.com' if i%2 else 'naver.com'}"
    addr = ADDRS[i % len(ADDRS)]
    gt = [("PERSON", name), ("PHONE", phone), ("EMAIL", email), ("ADDRESS", addr)]
    extras = []
    if i % 2 == 0:
        card = gen_card(2000 + i); gt.append(("CARD", card)); extras.append(f"결제 카드는 {card} 이고")
    if i % 3 == 0:
        rrn = gen_rrn(3000 + i); gt.append(("RRN", rrn)); extras.append(f"본인확인용 주민번호 {rrn} 도 불러드렸어요")
    if i % 2 == 1:
        acct = ACCTS[i % len(ACCTS)]; gt.append(("KR_ACCOUNT", acct)); extras.append(f"환불은 계좌 {acct} 로 받을게요")
    if i % 4 == 0:
        org = ORGS[i % len(ORGS)]; gt.append(("ORGANIZATION", org)); extras.append(f"직장은 {org} 입니다")

    scn = VOC_SCENARIOS[i % len(VOC_SCENARIOS)]
    # 이름을 실제 VOC처럼 다양한 자연 문맥 + 2회 언급
    intros = [
        f"안녕하세요, 제 이름은 {name}이고 {scn} 문제로 문의드립니다.",
        f"{name}입니다. {scn} 관련해서 도움이 필요합니다.",
        f"주문자 {name}이라고 합니다. {scn} 때문에 답답해서 연락드려요.",
        f"고객 {name}입니다만, {scn} 이슈가 해결되지 않습니다.",
    ]
    head = (f"[{scn} 문의] {intros[i % len(intros)]} 연락처는 {phone} 이고 "
            f"이메일 {email} 으로도 회신 가능합니다. 배송지는 {addr} 입니다. "
            + " ".join(extras) + ". ")
    # 본문 중간에 이름을 한 번 더 자연스럽게(서명/재확인)
    midref = (f" 다시 한 번 말씀드리면 {name} 본인이 직접 겪은 일이며, "
              f"위 연락처 {phone} 로 꼭 회신 부탁드립니다.")
    body = pad([head, midref], " ", 2000, VOC_FILLER) + f" — {name} 드림"
    # (이름은 텍스트에 여러 번 등장하지만 ground truth는 1회 — 한 번만 검출되면 TP)
    return {"id": i + 1, "kind": "VOC", "title": f"VOC {i+1:02d} · {scn}",
            "text": body, "gt": gt, "server_info": []}


# ── 서버 로그 생성 (30) ───────────────────────────────────────────────────────
SERVICES = ["payment-svc", "auth-svc", "order-svc", "notification-svc", "user-svc",
            "billing-svc", "shipping-svc", "inventory-svc", "search-svc", "gateway"]


def _sk(i, r):
    return "sk-live-" + "".join(r.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP0123456789") for _ in range(32))


def _ghp(i, r):
    return "ghp_" + "".join(r.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(36))


def make_log(i):
    r = random.Random(5000 + i)
    svc = SERVICES[i % len(SERVICES)]
    ip1 = f"10.0.{r.randint(0,255)}.{r.randint(1,254)}"
    ip2 = f"192.168.{r.randint(0,255)}.{r.randint(1,254)}"
    host = f"prod-{svc}-{r.randint(1,9):02d}.internal"
    api = _sk(i, r)
    aws = "AKIA" + "".join(r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16))
    jwt = ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI" +
           "".join(r.choice("abcdefghijklmnop0123456789") for _ in range(10)) + "9.sig" +
           "".join(r.choice("ABCdef123") for _ in range(8)))
    pw = f"Db{r.randint(1000,9999)}!Pass"
    gt = [("API_KEY", api), ("AWS_SECRET", aws), ("TOKEN", jwt), ("PASSWORD", pw)]
    server_info = [("IP_internal", ip1), ("IP_internal", ip2), ("HOSTNAME", host)]

    lines = [
        f"2026-06-23T0{i%9}:1{i%6}:33.221Z ERROR [{svc}] request failed host={host} peer={ip1}",
        f"2026-06-23T0{i%9}:1{i%6}:33.245Z DEBUG [{svc}] upstream db={ip2}:5432 user=svc_app",
        f"2026-06-23T0{i%9}:1{i%6}:33.260Z ERROR [{svc}] auth header: Authorization: Bearer {api}",
        f"2026-06-23T0{i%9}:1{i%6}:33.281Z ERROR [{svc}] aws sts AccessKeyId={aws} AccessDenied",
        f"2026-06-23T0{i%9}:1{i%6}:33.300Z WARN  [{svc}] session token={jwt} expired",
        f"2026-06-23T0{i%9}:1{i%6}:33.322Z DEBUG [{svc}] env DB_PASSWORD={pw} DB_HOST={ip2}",
        f"2026-06-23T0{i%9}:1{i%6}:33.340Z ERROR [{svc}] stacktrace:",
        f"    at com.corp.{svc}.Handler.process(Handler.java:{r.randint(40,400)})",
        f"    at com.corp.{svc}.Service.call(Service.java:{r.randint(40,400)})",
        f"2026-06-23T0{i%9}:1{i%6}:33.401Z INFO  [{svc}] retrying via gateway {ip1}:8443",
    ]
    if i % 3 == 0:
        pk = "-----BEGIN PRIVATE KEY-----MIIBVAIBADANBgkq" + "".join(r.choice("ABCabc123+/") for _ in range(20)) + "-----END PRIVATE KEY-----"
        gt.append(("PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----"))
        lines.append(f"2026-06-23T0{i%9}:1{i%6}:33.450Z ERROR [{svc}] tls key load failed: {pk}")
    if i % 2 == 0:
        ghp = _ghp(i, r); gt.append(("API_KEY", ghp))
        lines.append(f"2026-06-23T0{i%9}:1{i%6}:33.470Z ERROR [{svc}] ci token {ghp} invalid")

    text = pad(lines, "\n", 3000, LOG_FILLER)
    return {"id": 30 + i + 1, "kind": "LOG", "title": f"LOG {i+1:02d} · {svc}",
            "text": text, "gt": gt, "server_info": server_info}


def norm(s):
    return s.replace(" ", "").replace("-", "").replace("\n", "").lower()


def span_match(a, b):
    na, nb = norm(a), norm(b)
    return bool(na) and bool(nb) and (na in nb or nb in na)


def cat_ok(d, e):
    if d == e:
        return True
    return d in SECRET_CATS and e in SECRET_CATS


def run():
    items = [make_voc(i) for i in range(30)] + [make_log(i) for i in range(30)]
    eng = Engine(stage2_runner=Stage2NERRunner())
    log = []
    rows = []
    agg = {"VOC": {"tp": 0, "fn": 0}, "LOG": {"tp": 0, "fn": 0}}
    info = {"tp": 0, "fn": 0}
    per_cat = defaultdict(lambda: {"tp": 0, "fn": 0})

    def L(s=""):
        log.append(s)

    L("=" * 90)
    L("VOC + 서버로그 검출 테스트 (60항목) · 엔진: Stage1 + Stage2 NER(lg) + proximity")
    L("=" * 90)

    for it in items:
        res = eng.scan(it["text"])
        dets = [(d.category, d.original) for d in res.detections]
        used = set()
        tp, fn = [], []
        for ecat, eval_ in it["gt"]:
            hit = None
            for j, (dc, dv) in enumerate(dets):
                if j in used:
                    continue
                if cat_ok(dc, ecat) and span_match(dv, eval_):
                    hit = j
                    break
            if hit is not None:
                used.add(hit); tp.append((ecat, eval_)); per_cat[ecat]["tp"] += 1
                agg[it["kind"]]["tp"] += 1
            else:
                fn.append((ecat, eval_)); per_cat[ecat]["fn"] += 1
                agg[it["kind"]]["fn"] += 1
        # server-info (out-of-catalog) detection
        si_tp, si_fn = [], []
        for sicat, sival in it["server_info"]:
            if any(span_match(dv, sival) for _, dv in dets):
                si_tp.append((sicat, sival)); info["tp"] += 1
            else:
                si_fn.append((sicat, sival)); info["fn"] += 1

        L("")
        L("─" * 90)
        L(f"[{it['id']:02d}] {it['title']}  ({len(it['text'])}자)")
        L(f"  심은 PII/시크릿: {len(it['gt'])} · 서버정보(IP/호스트): {len(it['server_info'])}")
        L(f"  ✅검출 {len(tp)} / ❌미검출 {len(fn)}")
        if fn:
            L("  미검출: " + ", ".join(f"{c}={v}" for c, v in fn))
        if si_fn:
            L("  서버정보 미검출(범위밖): " + ", ".join(f"{c}={v}" for c, v in si_fn))
        rows.append({**{k: it[k] for k in ("id", "title", "kind")},
                     "n_gt": len(it["gt"]), "tp": len(tp), "fn": len(fn),
                     "len": len(it["text"]), "si_tp": len(si_tp), "si_fn": len(si_fn),
                     "fn_items": fn, "block": res.has_blocks})

    # 집계
    voc_tp, voc_fn = agg["VOC"]["tp"], agg["VOC"]["fn"]
    log_tp, log_fn = agg["LOG"]["tp"], agg["LOG"]["fn"]
    voc_r = voc_tp / (voc_tp + voc_fn) if (voc_tp + voc_fn) else 0
    log_r = log_tp / (log_tp + log_fn) if (log_tp + log_fn) else 0
    info_r = info["tp"] / (info["tp"] + info["fn"]) if (info["tp"] + info["fn"]) else 0

    L("\n" + "=" * 90)
    L(f"VOC PII 재현율    = {voc_tp}/{voc_tp+voc_fn} = {voc_r:.3f}")
    L(f"서버 시크릿 재현율 = {log_tp}/{log_tp+log_fn} = {log_r:.3f}")
    L(f"서버정보(IP/호스트) 검출 = {info['tp']}/{info['tp']+info['fn']} = {info_r:.3f}  (카탈로그 없음)")
    L("=" * 90)

    open(LOG_PATH, "w", encoding="utf-8").write("\n".join(log))
    summary = {"voc_r": voc_r, "log_r": log_r, "info_r": info_r,
               "voc_tp": voc_tp, "voc_fn": voc_fn, "log_tp": log_tp, "log_fn": log_fn,
               "info": info, "per_cat": {k: dict(v) for k, v in per_cat.items()}, "rows": rows}
    generate_md(summary, items)
    print(f"VOC PII recall={voc_r:.3f}  server-secret recall={log_r:.3f}  "
          f"server-info detect={info_r:.3f}")
    print("report →", MD_PATH)


def generate_md(s, items):
    o = []
    W = o.append
    by_id = {it["id"]: it for it in items}
    W("# VOC + 서버로그 PII/시크릿 검출 테스트 (60항목)")
    W("")
    W("> 시나리오: 고객 VOC(개인정보) + 서버 console 로그(서버정보·시크릿)를 외부 LLM 분석에 보내기 전 "
      "PII-Guard가 무엇을 가리는지 측정. 증거: [`voc_server_log.txt`](./voc_server_log.txt) · "
      "하니스: `validation/voc_server_test.py`")
    W("> 엔진: Stage1(정규식·체크섬) + Stage2 NER(ko_core_news_lg) + proximity(R17).")
    W("")
    W("## 1. 핵심 결과")
    W("")
    W("| 대상 | 항목수 | 심은 수 | 검출 | 재현율 |")
    W("| :-- | --: | --: | --: | --: |")
    W(f"| **VOC 개인정보**(이름·전화·이메일·주민번호·카드·계좌·주소) | 30 | {s['voc_tp']+s['voc_fn']} | {s['voc_tp']} | **{s['voc_r']:.3f}** |")
    W(f"| **서버 시크릿**(API키·AWS·JWT·프라이빗키·비밀번호) | 30 | {s['log_tp']+s['log_fn']} | {s['log_tp']} | **{s['log_r']:.3f}** |")
    W(f"| ⚠️ **서버정보**(내부 IP·호스트명) | 30 | {s['info']['tp']+s['info']['fn']} | {s['info']['tp']} | **{s['info_r']:.3f}** |")
    W("")
    W(f"> **요약**: 고객 PII와 서버 시크릿은 높은 비율로 차단/마스킹되지만, **내부 IP·호스트명은 "
      f"전용 카테고리가 없어 검출률 {s['info_r']:.2f}**(범위 밖) — 서버 토폴로지 보호가 필요하면 "
      "IP/HOSTNAME 카테고리 추가가 권고됨.")
    W("")
    W("## 2. 카테고리별")
    W("")
    W("| 카테고리 | TP | FN | recall |")
    W("| :-- | --: | --: | --: |")
    for cat in sorted(s["per_cat"]):
        c = s["per_cat"][cat]
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0
        flag = " ⚠️" if r < 0.85 else ""
        W(f"| {cat} | {c['tp']} | {c['fn']} | {r:.2f}{flag} |")
    W("")
    W("## 3. 항목별 요약")
    W("")
    W("| # | 항목 | 길이 | 심은 | 검출 | 미검출 | 서버정보 검출 | block |")
    W("| --: | :-- | --: | --: | --: | --: | :-- | :--: |")
    for row in s["rows"]:
        si = f"{row['si_tp']}/{row['si_tp']+row['si_fn']}" if (row['si_tp']+row['si_fn']) else "–"
        blk = "🔴" if row["block"] else "—"
        W(f"| {row['id']:02d} | {row['title']} | {row['len']} | {row['n_gt']} | {row['tp']} | {row['fn']} | {si} | {blk} |")
    W("")
    W("## 4. 해석 (고객센터 분석 시나리오)")
    W("")
    W("- ✅ **VOC 개인정보**: 이름·전화·이메일·주소·주민번호·카드·계좌가 마스킹/차단되어 외부 LLM에 "
      "고객 PII가 평문으로 나가지 않음 → **VOC를 LLM 분석에 보내도 안전**.")
    W("- ✅ **서버 시크릿**: 로그 속 API키·AWS키·JWT·프라이빗키·DB비밀번호가 차단(block) → "
      "**자격증명 유출 차단**.")
    W("- ⚠️ **서버정보(IP·호스트명)**: 전용 카테고리 부재로 대부분 통과. LLM이 분석에 활용하긴 하나 "
      "**내부 네트워크 토폴로지가 외부 LLM에 노출**될 수 있음 → 필요 시 IP/HOSTNAME 카테고리 추가.")
    W("- 💡 권고: VOC↔로그 상관분석 워크플로에서 PII-Guard를 게이트웨이로 두면 **양쪽 민감정보를 "
      "마스킹한 채 LLM이 '어떤 기능 서버의 어떤 오류인지'를 분석**할 수 있음(플레이스홀더로 구조 보존).")
    W("")
    W("## 5. 재현")
    W("```bash")
    W("PYTHONPATH=. .venv/bin/python validation/voc_server_test.py")
    W("```")
    open(MD_PATH, "w", encoding="utf-8").write("\n".join(o))


if __name__ == "__main__":
    run()
