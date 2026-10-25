"""Label the radiology reports with any OpenAI-compatible LLM server (vLLM / Ollama / LM Studio / commercial API).

Settings are read from `.env` (keys never go into code or chat):
    LLM_BASE_URL=http://SERVER:8000/v1
    LLM_MODEL=Qwen/Qwen3-32B
    LLM_API_KEY=...            (may be empty)

Usage:
    python scripts/llm_label.py --pilot --backend ollama --url http://SERVER:11434 --model qwen3.5:27b
    python scripts/llm_label.py --pilot          # the 58 gold studies only -> AUC report
    python scripts/llm_label.py --all            # all 4,407 reports (resumes where it stopped)
    python scripts/llm_label.py --all --score    # 0-100 graded scores instead of letters (keeps ranking information)

Output: data/external/llm_<model>/codes.csv (P/A/N/U letters or scores + raw answer) and labels.csv
(StudyInstanceUID + 12 probabilities; usable directly as a source in src/labels.py).

Results in this competition (58 gold studies): Qwen2.5-7B letters 0.797, Qwen3.5-27B letters 0.828,
Qwen3.5-27B scores 0.881; public label sets were stronger (~0.90), so none of these was used.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR, TARGET_COLS  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

KEYS = ["ACL", "MCL", "MM", "LM", "MOA", "LOA", "PFOA", "EFF", "SYN", "BAKER", "CONT", "FX"]
KEY2COL = dict(zip(KEYS, TARGET_COLS))
# Letter -> probability. NOT tuned on the gold studies (58 studies are too few for that).
CODE_P = {"P": 0.95, "U": 0.6, "N": 0.08, "A": 0.03}

SYSTEM = """You are an expert musculoskeletal radiologist. You read one knee MRI report (it may be in any language: English, Spanish, French, Dutch, German, Greek, Turkish, Portuguese, Italian, ...) and classify twelve findings strictly from what the report says.

For each finding answer with exactly one letter:
P = present (stated as present, including partial/low-grade/small/mild, post-traumatic or chronic)
A = absent (explicitly normal/intact/"no ...", or the structure is described without this abnormality)
N = not mentioned at all in the report
U = uncertain (suspected, possible, probable, cannot exclude, equivocal, "suggestive of")

Findings and rules:
ACL  = anterior cruciate ligament injury: sprain, partial or complete tear, rupture, mucoid degeneration WITH tear, torn graft. Intact graft after reconstruction = A. Mucoid degeneration without tear = A.
MCL  = medial collateral ligament injury: sprain, strain, grade 1-3 injury, partial/complete tear, periligamentous edema around the MCL, old/healed MCL injury with thickening.
MM   = medial meniscus TEAR (any type: horizontal, vertical, radial, flap, complex, root, bucket-handle, re-tear, "signal reaching the articular surface"). Degeneration / intrasubstance signal NOT reaching the surface = A. Post-meniscectomy without a new tear = A.
LM   = lateral meniscus tear, same rules as MM. Discoid meniscus alone = A.
MOA  = osteoarthritis of the MEDIAL tibiofemoral compartment: cartilage thinning/loss/fissuring, chondral defect or chondromalacia of the medial femoral condyle or medial tibial plateau, medial osteophytes, medial joint space narrowing, medial compartment arthrosis/gonarthrosis.
LOA  = same for the LATERAL tibiofemoral compartment.
PFOA = patellofemoral osteoarthritis: chondromalacia patellae (any grade), patellar or trochlear cartilage thinning/loss/defect, patellofemoral osteophytes or arthrosis.
       If the report states generalized / tricompartmental / global osteoarthritis or gonarthrosis without naming sides, mark MOA, LOA and PFOA all P.
EFF  = joint effusion of any size (small, minimal, moderate, large, hemarthrosis, lipohemarthrosis, "joint fluid increased"). "Physiological / normal amount of fluid" = A.
SYN  = synovitis: synovitis, synovial thickening/proliferation/hypertrophy, effusion-synovitis, Hoffa fat pad edema/synovitis, plica inflammation.
BAKER = Baker's (popliteal) cyst of any size, including ruptured or leaking.
CONT = bone contusion / bone bruise / post-traumatic bone marrow edema. Degenerative subchondral edema or cysts in osteoarthritis = A unless explicitly called contusion/bruise.
FX   = any fracture: avulsion (e.g. Segond), impaction, osteochondral, subchondral insufficiency fracture, stress fracture, tibial plateau / patellar / femoral fracture, "fracture line".

Radiologists usually list abnormalities and may not mention normal structures; if a structure is not discussed at all use N, not A.

Return ONLY a JSON object with exactly these keys and one letter each, no explanation:
{"ACL":"","MCL":"","MM":"","LM":"","MOA":"","LOA":"","PFOA":"","EFF":"","SYN":"","BAKER":"","CONT":"","FX":""}"""

SYSTEM_SCORE = """You are an expert musculoskeletal radiologist. You read one knee MRI report (any language) and, for each of twelve findings, give an integer score 0-100: the probability that an expert radiologist reviewing this knee MRI would label the finding as CLINICALLY PRESENT.

Calibrate the score to both how certain and how significant the report makes the finding:
- explicitly normal / intact / "no ..."                     -> 0-5
- not discussed at all in the report                        -> 5-15 (radiologists usually mention abnormalities)
- trace / physiological / questionable                      -> 10-25
- minimal / tiny / mild early change                        -> 25-40
- suspected / possible / cannot exclude / equivocal         -> 40-60
- small but definite                                        -> 55-70
- definite, moderate                                        -> 80-90
- definite, large / severe / complete / high grade          -> 92-100

Findings:
ACL   anterior cruciate ligament injury (sprain, partial/complete tear, torn graft; intact graft = low)
MCL   medial collateral ligament injury (sprain grade 1-3, partial/complete tear, periligamentous edema)
MM    medial meniscus TEAR (degeneration/intrasubstance signal without surface extension = low; root/flap/bucket-handle/complex = high)
LM    lateral meniscus tear (same rules)
MOA   osteoarthritis of the medial tibiofemoral compartment (cartilage loss/defect, chondromalacia, osteophytes, joint space narrowing)
LOA   same, lateral tibiofemoral compartment
PFOA  patellofemoral osteoarthritis (chondromalacia patellae any grade, patellar/trochlear cartilage loss, PF osteophytes)
      (generalized/tricompartmental osteoarthritis without sides -> score MOA, LOA and PFOA all high)
EFF   joint effusion (score by amount: physiological/trace low, small medium, moderate/large high; hemarthrosis high)
SYN   synovitis (synovial thickening/proliferation, effusion-synovitis, Hoffa fat pad synovitis)
BAKER Baker's / popliteal cyst (any size; ruptured high)
CONT  bone contusion / bone bruise / post-traumatic bone marrow edema (degenerative subchondral edema = low-moderate)
FX    any fracture (avulsion e.g. Segond, impaction, osteochondral, subchondral insufficiency, stress, plateau)

Return ONLY a JSON object with integer values:
{"ACL":0,"MCL":0,"MM":0,"LM":0,"MOA":0,"LOA":0,"PFOA":0,"EFF":0,"SYN":0,"BAKER":0,"CONT":0,"FX":0}"""

NUM = re.compile(r'"(\w+)"\s*:\s*"?(\d{1,3})"?')


def parse_score(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    found = {k: min(100, int(v)) for k, v in NUM.findall(text)}
    return {k: found.get(k, "") for k in KEYS}


LETTER = re.compile(r'"(\w+)"\s*:\s*"([PANUpanu])"')


def parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # Qwen3 thinking block
    found = {k: v.upper() for k, v in LETTER.findall(text)}
    return {k: found.get(k, "") for k in KEYS}


def ask(session: requests.Session, report: str, cfg: dict, retries: int = 4) -> str:
    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": SYSTEM_SCORE if cfg.get("score") else SYSTEM},
                     {"role": "user", "content": "REPORT:\n" + str(report)[:8000]}],
        "temperature": 0.0,
        "max_tokens": cfg["max_tokens"],
    }
    if cfg["no_think"]:
        # Qwen3 / Qwen3.5: disable thinking (vLLM and most servers accept this field)
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {cfg['key']}"} if cfg["key"] else {}
    if cfg["backend"] == "ollama":
        # Ollama's own API: thinking via `think`, JSON output forced via `format`
        body = {"model": cfg["model"], "messages": body["messages"], "stream": False,
                "think": not cfg["no_think"],
                "options": {"temperature": 0, "num_ctx": cfg["num_ctx"], "num_predict": cfg["max_tokens"]}}
        if cfg["no_think"]:
            body["format"] = "json"
    for attempt in range(retries):
        try:
            if cfg["backend"] == "ollama":
                r = session.post(cfg["url"].rstrip("/") + "/api/chat", json=body, timeout=cfg["timeout"])
                r.raise_for_status()
                return r.json()["message"]["content"] or ""
            r = session.post(cfg["url"].rstrip("/") + "/chat/completions", json=body, headers=headers,
                             timeout=cfg["timeout"])
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # network / server error: wait and retry
            if attempt == retries - 1:
                return f"__ERROR__ {type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return ""


def to_probs(codes: pd.DataFrame, score: bool = False) -> pd.DataFrame:
    out = codes[["StudyInstanceUID"]].copy()
    for k in KEYS:
        if score:
            out[KEY2COL[k]] = pd.to_numeric(codes[k], errors="coerce").fillna(10) / 100.0
        else:
            out[KEY2COL[k]] = codes[k].map(CODE_P).fillna(0.15)
    return out


def gold_report(probs: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score
    train = pd.read_csv(DATA_DIR / "train.csv")
    gold = train[train[TARGET_COLS[0]].notna()].set_index("StudyInstanceUID")[TARGET_COLS]
    p = probs.set_index("StudyInstanceUID").reindex(gold.index)
    aucs = {c: roc_auc_score(gold[c], p[c].fillna(0.15)) for c in TARGET_COLS if gold[c].nunique() > 1}
    print(f"\nGold macro AUC: {np.mean(list(aucs.values())):.4f}  (n={p.notna().all(axis=1).sum()})")
    print(pd.Series(aucs).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="the 58 gold studies only")
    ap.add_argument("--all", action="store_true", help="all reports")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--think", action="store_true", help="keep Qwen3 thinking on (slow; raise --max-tokens)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--url", default=os.getenv("LLM_BASE_URL", ""), help="e.g. http://SERVER:11434")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", ""), help="e.g. qwen3.5:27b")
    ap.add_argument("--backend", default=os.getenv("LLM_BACKEND", "openai"), choices=["openai", "ollama"])
    ap.add_argument("--num-ctx", type=int, default=8192, help="Ollama context length")
    ap.add_argument("--score", action="store_true", help="0-100 graded score instead of letters (keeps ranking information)")
    a = ap.parse_args()

    cfg = {"url": a.url, "model": a.model, "backend": a.backend, "num_ctx": a.num_ctx,
           "key": os.getenv("LLM_API_KEY", ""), "max_tokens": a.max_tokens, "no_think": not a.think,
           "timeout": a.timeout, "score": a.score}
    if not cfg["url"] or not cfg["model"]:
        raise SystemExit("pass --url and --model (or set LLM_BASE_URL / LLM_MODEL in .env)")

    train = pd.read_csv(DATA_DIR / "train.csv")
    if a.pilot:
        train = train[train[TARGET_COLS[0]].notna()]
    elif not a.all:
        raise SystemExit("choose --pilot or --all")

    tag = re.sub(r"[^a-zA-Z0-9]+", "_", cfg["model"]).strip("_").lower() + ("_think" if a.think else "") + ("_score" if a.score else "")
    out_dir = DATA_DIR / "external" / f"llm_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_path = out_dir / ("codes_pilot.csv" if a.pilot else "codes.csv")
    done = pd.read_csv(codes_path) if codes_path.exists() else pd.DataFrame(columns=["StudyInstanceUID"])
    done = done[~done.get("raw", pd.Series(dtype=str)).astype(str).str.startswith("__ERROR__")] if len(done) else done
    todo = train[~train.StudyInstanceUID.isin(set(done.StudyInstanceUID))]
    print(f"Model: {cfg['model']} | server: {cfg['url']} | done: {len(done)} | left: {len(todo)}")

    rows = done.to_dict("records")
    session = requests.Session()
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ask, session, r.Report, cfg): r.StudyInstanceUID for r in todo.itertuples()}
        for n, f in enumerate(as_completed(futs), 1):
            raw = f.result()
            rows.append({"StudyInstanceUID": futs[f], "raw": raw, **(parse_score(raw) if a.score else parse(raw))})
            if n % 50 == 0 or n == len(futs):
                pd.DataFrame(rows).to_csv(codes_path, index=False)
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(futs)}  {rate:.2f} reports/s  (~{(len(futs) - n) / max(rate, 1e-6) / 60:.0f} min left)", flush=True)

    codes = pd.DataFrame(rows)
    codes.to_csv(codes_path, index=False)
    n_err = codes.raw.astype(str).str.startswith("__ERROR__").sum()
    empty = codes[KEYS].astype(str).isin(["", "nan"]).mean().mean() if len(codes) else 0
    print(f"\nDone: {len(codes)} reports | errors {n_err} | unparsed cell share {empty:.3f} | "
          f"{(time.time() - t0) / 60:.1f} min")
    probs = to_probs(codes, score=a.score)
    probs.to_csv(out_dir / ("labels_pilot.csv" if a.pilot else "labels.csv"), index=False)
    gold_report(probs)


if __name__ == "__main__":
    main()
