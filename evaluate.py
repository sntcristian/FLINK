"""Evaluates the pipeline on a tabular dataset.

input-dir:
  documents_test.csv     doc_id, text
  annotations_test.csv   doc_id, start_pos, end_pos, surface_form, identifier

output-dir:
  report.txt               pipeline configuration and scores
  output.csv               annotations produced by the pipeline
  tp.csv, fp.csv, fn.csv   true positives, false positives and false negatives
                           of the NER + EL evaluation

Each document is split into sentences with spaCy and the pipeline runs on each
sentence (the graph is built per sentence).

"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.graph_reranker import CENTRALITIES, STRATEGIES
from src.pipeline import Pipeline

QID_PATTERN = re.compile(r"Q\d+")
TOKEN_PATTERN = re.compile(r"\w+")

PRED_COLUMNS = ["doc_id", "start_pos", "end_pos", "surface_form", "identifier", "label", "description",
                "type", "cosine_score", "centrality_score", "final_score", "candidates",
                "sentence_id", "graph_reranked"]
TP_COLUMNS = ["doc_id", "gold_start_pos", "gold_end_pos", "gold_surface_form", "gold_identifier",
              "pred_start_pos", "pred_end_pos", "pred_surface_form", "pred_identifier", "pred_label"]
FP_COLUMNS = ["doc_id", "start_pos", "end_pos", "surface_form", "identifier", "label",
              "overlapping_gold_identifier"]
FN_COLUMNS = ["doc_id", "start_pos", "end_pos", "surface_form", "identifier",
              "overlapping_pred_identifier"]


# ---------- Helpers ----------

def normalize_id(identifier) -> str:
    match = QID_PATTERN.search(str(identifier))
    return match.group() if match else "NIL"


def token_ids(tokens: list[tuple[int, int]], start: int, end: int) -> set[int]:
    """Indices of the document tokens that overlap the character span [start, end)."""
    return {i for i, (s, e) in enumerate(tokens) if s < end and e > start}


def match(gold: list[dict], pred: list[dict], same_id: bool) -> list[tuple[int, int]]:
    """One-to-one matching on token overlap; returns (gold index, pred index) pairs."""
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            overlap = len(g["tokens"] & p["tokens"])
            if overlap and (not same_id or g["qid"] == p["qid"]):
                pairs.append((overlap, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_gold, used_pred, matches = set(), set(), []
    for _, gi, pi in pairs:
        if gi not in used_gold and pi not in used_pred:
            used_gold.add(gi)
            used_pred.add(pi)
            matches.append((gi, pi))
    return matches


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# ---------- Pipeline on documents ----------

def annotate(pipeline: Pipeline, doc_id: str, text: str) -> list[dict]:
    """Runs the pipeline on each sentence of a document; offsets refer to the document."""
    rows = []
    for i, sent in enumerate(pipeline.ner.nlp(text).sents):
        if not sent.text.strip():
            continue
        result = pipeline(sent.text, f"{doc_id}_{i}")
        for m in result["mentions"]:
            top = m["candidates"][0] if m["candidates"] else {}
            rows.append({
                "doc_id": doc_id,
                "start_pos": sent.start_char + m["start_offset"],
                "end_pos": sent.start_char + m["end_offset"],
                "surface_form": m["text"],
                "identifier": m["prediction"] or "NIL",
                "label": top.get("label", ""),
                "description": top.get("description", ""),
                "type": m["type"],
                "cosine_score": top.get("cosine_score", ""),
                "centrality_score": top.get("centrality_score", ""),
                "final_score": top.get("final_score", ""),
                "candidates": "|".join(c["qid"] for c in m["candidates"]),
                "sentence_id": result["sentence_id"],
                "graph_reranked": result["graph_reranked"],
            })
    return rows


# ---------- Evaluation ----------

def evaluate(documents: pd.DataFrame, gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    counts = {"ner": [0, 0, 0], "el": [0, 0, 0]}  # tp, fp, fn
    tp_rows, fp_rows, fn_rows = [], [], []

    for doc_id, text in zip(documents["doc_id"], documents["text"]):
        tokens = [m.span() for m in TOKEN_PATTERN.finditer(text)]

        def mentions(df):
            rows = df[df["doc_id"] == doc_id].to_dict("records")
            for r in rows:
                r["tokens"] = token_ids(tokens, r["start_pos"], r["end_pos"])
                r["qid"] = normalize_id(r["identifier"])
            return rows

        gold, pred = mentions(gold_df), mentions(pred_df)

        ner_matches = match(gold, pred, same_id=False)
        el_matches = match(gold, pred, same_id=True)
        for task, task_matches in (("ner", ner_matches), ("el", el_matches)):
            counts[task][0] += len(task_matches)
            counts[task][1] += len(pred) - len(task_matches)
            counts[task][2] += len(gold) - len(task_matches)

        # Rows for tp.csv, fp.csv, fn.csv (NER + EL only)
        matches = el_matches
        matched_gold = {gi for gi, _ in matches}
        matched_pred = {pi for _, pi in matches}

        for gi, pi in matches:
            g, p = gold[gi], pred[pi]
            tp_rows.append({
                "doc_id": doc_id,
                "gold_start_pos": g["start_pos"], "gold_end_pos": g["end_pos"],
                "gold_surface_form": g["surface_form"], "gold_identifier": g["qid"],
                "pred_start_pos": p["start_pos"], "pred_end_pos": p["end_pos"],
                "pred_surface_form": p["surface_form"], "pred_identifier": p["qid"],
                "pred_label": p["label"],
            })
        for pi, p in enumerate(pred):
            if pi not in matched_pred:
                # Gold identifiers of overlapping gold mentions: empty = spurious mention,
                # otherwise the span was found but linked to the wrong entity.
                overlapping = [g["qid"] for g in gold if g["tokens"] & p["tokens"]]
                fp_rows.append({
                    "doc_id": doc_id, "start_pos": p["start_pos"], "end_pos": p["end_pos"],
                    "surface_form": p["surface_form"], "identifier": p["qid"], "label": p["label"],
                    "overlapping_gold_identifier": "|".join(overlapping),
                })
        for gi, g in enumerate(gold):
            if gi not in matched_gold:
                overlapping = [p["qid"] for p in pred if p["tokens"] & g["tokens"]]
                fn_rows.append({
                    "doc_id": doc_id, "start_pos": g["start_pos"], "end_pos": g["end_pos"],
                    "surface_form": g["surface_form"], "identifier": g["qid"],
                    "overlapping_pred_identifier": "|".join(overlapping),
                })

    return {"counts": counts, "tp": tp_rows, "fp": fp_rows, "fn": fn_rows}


def build_report(args, config, documents, gold_df, pred_df, results, offset_mismatches, runtime) -> str:
    lines = [
        "FLINK: evaluation report",
        f"Date: {datetime.now().isoformat(timespec='seconds')}",
        f"Input dir: {args.input_dir}",
        f"Output dir: {args.output_dir}",
        "",
        "Pipeline configuration",
        *[f"  {k}: {v}" for k, v in config.items()],
        "",
        "Dataset",
        f"  documents: {len(documents)}",
        f"  gold mentions: {len(gold_df)} (NIL: {(gold_df['identifier'].map(normalize_id) == 'NIL').sum()})",
        f"  gold spans not matching their surface form: {offset_mismatches}",
        f"  predicted mentions: {len(pred_df)} (NIL: {(pred_df['identifier'] == 'NIL').sum() if len(pred_df) else 0})",
        f"  runtime: {runtime:.1f} s",
        "",
        "Matching: at least one token overlap, one-to-one",
        "",
        f"{'Task':<22}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>9}{'R':>9}{'F1':>9}",
    ]
    for task, name in (("ner", "NER (span)"), ("el", "NER + EL (span + id)")):
        tp, fp, fn = results["counts"][task]
        p, r, f = prf(tp, fp, fn)
        lines.append(f"{name:<22}{tp:>6}{fp:>6}{fn:>6}{p:>9.3f}{r:>9.3f}{f:>9.3f}")

    ner_tp, el_tp = results["counts"]["ner"][0], results["counts"]["el"][0]
    if ner_tp:
        lines += ["", f"Linking accuracy on matched spans (EL TP / NER TP): {el_tp / ner_tp:.3f}"]
    return "\n".join(lines) + "\n"


# ---------- Main ----------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the entity linking pipeline.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trf-rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trf-model", default="dmis-lab/UltraBERT-Embed-150M")
    parser.add_argument("--graph-rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strategy", choices=STRATEGIES, default="wikidata")
    parser.add_argument("--centrality", choices=CENTRALITIES, default="degree")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--ner-model", default="en_core_web_lg")
    parser.add_argument("--candidates", type=Path, default=None) 
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    read = lambda name: pd.read_csv(args.input_dir / name, dtype={"doc_id": str}, keep_default_na=False)
    documents = read("documents_test.csv")
    gold_df = read("annotations_test.csv")
    gold_df[["start_pos", "end_pos"]] = gold_df[["start_pos", "end_pos"]].astype(int)

    texts = dict(zip(documents["doc_id"], documents["text"]))
    offset_mismatches = sum(
        texts.get(d, "")[s:e] != sf
        for d, s, e, sf in zip(gold_df["doc_id"], gold_df["start_pos"], gold_df["end_pos"], gold_df["surface_form"])
    )

    pipeline = Pipeline(
        trf_rerank=args.trf_rerank,
        trf_model=args.trf_model,
        graph_rerank=args.graph_rerank,
        strategy=args.strategy,
        centrality=args.centrality,
        ner_model=args.ner_model,
        limit=args.limit,
        candidates_path=args.candidates)

    start = time.time()
    pred_rows = []
    for n, (doc_id, text) in enumerate(texts.items(), 1):
        print(f"[{n}/{len(texts)}] {doc_id}")
        pred_rows += annotate(pipeline, doc_id, text)
    runtime = time.time() - start
    
    if args.candidates:                                             
        pipeline.save_candidates()

    pred_df = pd.DataFrame(pred_rows, columns=PRED_COLUMNS)

    results = evaluate(documents, gold_df, pred_df)
    report = build_report(args, pipeline.config, documents, gold_df, pred_df, results, offset_mismatches, runtime)

    pred_df.to_csv(args.output_dir / "output.csv", index=False)
    for name, columns in (("tp", TP_COLUMNS), ("fp", FP_COLUMNS), ("fn", FN_COLUMNS)):
        pd.DataFrame(results[name], columns=columns).to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()