# `training/` — GLiNER 파인튜닝 서브시스템 (ADR-13 파일럿)

ORG 정밀도·잔여 recall을 끌어올리기 위한 **오프박스 GLiNER 파인튜닝** 도구.
목표 시나리오: **수천~수만 건의 보유 라벨 데이터로 본격 파인튜닝**(좁은 목표는 수백~2,000도 가능).

> **런타임은 변경 없음.** 산출 모델은 기존 슬롯 `PIIGUARD_GLINER_MODEL=<경로>`로 로드된다(코어 무변경).
> 이 디렉터리는 **런타임 패키지(`pii_guard/`)가 아니며**, 코어를 import하지 않는다(설계 격리, ADR-2/13).

## 파이프라인

```
보유 라벨 데이터(raw {text,spans})
        │  (+ 선택) augment.py 합성 보강(positive·hard-negative)
        ▼
ingest.py   → gliner.jsonl  (문자-span → 토큰-span 변환·검증)
        ▼
split.py    → train/val/test.jsonl  (누설 방지: 평가셋과 겹치면 제거)
        ▼
train.py    → runs/<name>/final  (베이스 urchade Apache, train_model)
        ▼
eval.py     → 베이스 vs 파인튜닝 코퍼스 벤치마크 비교 + 채택 게이트
        ▼
배포: PIIGUARD_GLINER_MODEL=runs/<name>/final
```

## 빠른 실행 (스모크 — 합성 데이터)

```bash
PY=.venv/bin/python
# 1) 합성 보강(보조)  2) 적재  3) 분할
$PY -m training.augment /tmp/syn.jsonl --n-pos 60 --n-neg 60
$PY -m training.ingest  /tmp/syn.jsonl /tmp/gliner.jsonl
$PY -m training.split   /tmp/gliner.jsonl /tmp/ds
# 4) 학습 배선 검증(CPU 1스텝)        5) 사전/사후 평가
$PY -m training.train --train /tmp/ds/train.jsonl --val /tmp/ds/val.jsonl --out /tmp/ft --smoke
$PY -m training.eval  --finetuned /tmp/ft/final
```

본 학습(실데이터)은 GPU 호스트에서 `--epochs 3 --batch 8 --lr 5e-6` 수준으로.

## 누설 방지 (필수)
학습셋과 **평가셋(코퍼스 seed=42·외부 6리포트)**은 겹치면 안 된다. `split.py --eval-fp <지문파일>`로
평가셋 원문/지문을 주면 학습에서 제거한다. (지문 = 정규화 텍스트 SHA1.)

## 데이터 거버넌스 (실 PII 학습 시)
- 학습 데이터는 **오프박스 학습 호스트에서만**, **레포 미커밋**(아래 `.gitignore`), 암호화 보관(P4).
- 모델 가중치는 **희소 문자열 암기 위험** → 배포 전 암기/유출 수동 점검(학습 샘플을 모델이 그대로 재현하는지).
- 라이선스: 베이스 `urchade/gliner_multi_pii-v1` = **Apache-2.0** → 파인튜닝 가중치도 Apache(상업 가능).
  학습 데이터는 보유자 소유. (NC 모델 `taeminlee/gliner_ko`는 상업 파인튜닝 금지.)

## 채택 게이트 (eval.py)
파인튜닝 모델은 ① 전 임계값 통과 ② 어떤 카테고리도 recall 회귀 없음 ③ 목표(ORG 정밀도 등) 개선일 때만
기본 모델로 승격. 통과 시 `PIIGUARD_GLINER_MODEL`을 그 경로로 전환.

## 런타임 통합 주의 (라벨 정렬)
GLiNER는 라벨-텍스트 매칭이라 **학습 라벨(PERSON/ADDRESS/ORGANIZATION) = 런타임 질의 라벨**이어야 한다.
파인튜닝 모델 채택 시 `pii_guard/stage2/gliner_ner.py::_GLINER_LABELS`를 캐노니컬 집합으로 정렬할 것(schema.md §3).

## 의존성
`pip install 'pii-guard[ner-gliner]'` (gliner + torch). 본 학습은 추가로 GPU 권장.
`sample/raw_sample.jsonl` = 입력 포맷 예시(합성·비실데이터).
