"""
training/train.py — GLiNER 파인튜닝 실행 (ADR-13).

베이스 = Apache-2.0 `urchade/gliner_multi_pii-v1`(상업 가능) → 산출물도 Apache.
GLiNER 학습 포맷 jsonl(`ingest`→`split` 산출)을 받아 `model.train_model`로 학습, 모델 저장.

실행(오프박스·GPU 권장):
    .venv/bin/python -m training.train --train data/train.jsonl --val data/val.jsonl \
        --out runs/ko_pii_ft --epochs 3 --batch 8 --lr 5e-6
배선 검증용 초소형 스모크(CPU, 1스텝):
    .venv/bin/python -m training.train --train ... --val ... --out /tmp/ft_smoke --smoke

⚠️ 실 PII 학습 데이터는 오프박스 호스트에서만·레포 미커밋(.gitignore)·암호화 보관(P4).
   학습 후 배포 전 **암기 점검**(eval.py --memorization) 권장.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

BASE_MODEL = "urchade/gliner_multi_pii-v1"   # Apache-2.0


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(train_path: str, val_path: str, out_dir: str, *,
        base: str = BASE_MODEL, epochs: float = 3.0, batch: int = 8,
        lr: float = 5e-6, smoke: bool = False) -> str:
    # 무거운 의존은 함수 안에서 import(파일 import만으로 torch 로딩 안 되게)
    from gliner import GLiNER
    from gliner.training import TrainingArguments

    train_ds = _read_jsonl(train_path)
    val_ds = _read_jsonl(val_path)
    print(f"[train] base={base}  train={len(train_ds)}  val={len(val_ds)}  "
          f"{'(SMOKE: 1 step)' if smoke else f'epochs={epochs} batch={batch} lr={lr}'}")

    model = GLiNER.from_pretrained(base)

    args_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=100,
        num_train_epochs=epochs,
        save_strategy="epoch",
        report_to="none",
    )
    if smoke:                          # 배선만 확인 — 1스텝
        args_kwargs.update(max_steps=1, num_train_epochs=1, save_strategy="no")

    training_args = TrainingArguments(**args_kwargs)
    model.train_model(train_ds, val_ds, training_args=training_args, output_dir=out_dir)

    save_dir = str(Path(out_dir) / "final")
    model.save_pretrained(save_dir)
    print(f"[train] saved → {save_dir}  (배포: PIIGUARD_GLINER_MODEL={save_dir})")
    return save_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--smoke", action="store_true", help="CPU 1스텝 배선 검증")
    a = ap.parse_args()
    run(a.train, a.val, a.out, base=a.base, epochs=a.epochs,
        batch=a.batch, lr=a.lr, smoke=a.smoke)
