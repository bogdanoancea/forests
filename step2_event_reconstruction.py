#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import hdbscan
from sentence_transformers import SentenceTransformer

from step1_grid_search_eventlike_gate_v7 import (
    build_doc,
    heuristic_entities,
    parse_date_iso,
    sim_matrix_timefiltered_topk,
    split_labels_by_time
)

def load_best_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "best" not in payload:
        raise ValueError("Expected best_params.json to contain a top-level key 'best'.")
    return payload["best"]

def compute_event_centroids(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Minimal, paper-friendly event centroid table:
      - event_id
      - n_articles
      - date_min / date_max / span_days
      - representative_title (+ doc_id)
    Representative is chosen as earliest article in the event (deterministic).
    """
    out = []
    df2 = df.copy()
    df2["event_id"] = labels

    # robust date parsing
    df2["date_dt"] = pd.to_datetime(df2.get("date_iso", pd.Series([None]*len(df2))), errors="coerce")

    for event_id, g in df2.groupby("event_id", dropna=False):
        event_id = int(event_id)
        if event_id == -1:
            continue

        g = g.sort_values(["date_dt", "doc_id"], ascending=[True, True])

        date_min = g["date_dt"].min()
        date_max = g["date_dt"].max()
        span_days = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else np.nan

        rep_row = g.iloc[0]

        out.append({
            "event_id": event_id,
            "n_articles": int(len(g)),
            "date_min": date_min.date().isoformat() if pd.notna(date_min) else "",
            "date_max": date_max.date().isoformat() if pd.notna(date_max) else "",
            "span_days": float(span_days) if pd.notna(span_days) else np.nan,
            "representative_doc_id": int(rep_row["doc_id"]),
            "representative_title": str(rep_row.get("title", "")) if pd.notna(rep_row.get("title", "")) else ""
        })

    return pd.DataFrame(out).sort_values(["n_articles", "event_id"], ascending=[False, True]).reset_index(drop=True)

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input XLSX (proteste.xlsx)")
    ap.add_argument("--params", required=True, help="best_params.json produced by grid search")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--st_model", default="paraphrase-multilingual-MiniLM-L12-v2")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_excel(args.input)
    df = df.copy()
    df["doc_id"] = np.arange(len(df), dtype=int)

    # Build documents + features
    docs = [build_doc(str(r.get("title", "") or ""), str(r.get("text", "") or "")) for _, r in df.iterrows()]
    dates = [parse_date_iso(x) for x in df.get("date_iso", [None] * len(df))]
    ents = [heuristic_entities(d) for d in docs]

    # Embeddings
    model = SentenceTransformer(args.st_model)
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    # Load best params
    p = load_best_params(args.params)

    # Pull optional params safely (keep defaults aligned with grid script)
    sim_kwargs = dict(
        alpha=float(p["alpha"]),
        beta=float(p["beta"]),
        gamma=float(p["gamma"]),
        tau_days=float(p["tau_days"]),
        top_k=int(p["top_k"]),
        time_window_days=int(p.get("time_window_days", 7)),
        require_dates_for_edges=bool(p.get("require_dates_for_edges", True)),
        use_soft_gate=bool(p.get("use_soft_gate", True)),
        time_gate_eps=float(p.get("time_gate_eps", 0.01)),
    )

    # Similarity
    S = sim_matrix_timefiltered_topk(emb, ents, dates, **sim_kwargs)

    # Distance for HDBSCAN (IMPORTANT: float64 contiguous + zero diag)
    D = 1.0 - np.clip(S, 0.0, 1.0)
    D = np.ascontiguousarray(D, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    # Clustering
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(p["min_cluster_size"]),
        min_samples=int(p["min_samples"]),
    )
    labels_raw = clusterer.fit_predict(D).astype(int)

    # Optional temporal post-splitting (use params if present; else enable if you used it in grid)
    do_split = bool(p.get("split_by_time", True))
    split_gap = int(p.get("split_gap_days", p.get("time_window_days", 7)))
    split_min_seg = int(p.get("split_min_segment_size", 2))

    labels = labels_raw
    if do_split:
        labels = split_labels_by_time(labels, dates, split_gap_days=split_gap, min_segment_size=split_min_seg).astype(int)

    # Save outputs
    df_out = df.copy()
    df_out["event_id_raw"] = labels_raw
    df_out["event_id"] = labels

    events_path = os.path.join(args.outdir, "events.csv")
    df_out.to_csv(events_path, index=False, encoding="utf-8")

    centroids = compute_event_centroids(df_out, labels)
    centroids_path = os.path.join(args.outdir, "event_centroids.csv")
    centroids.to_csv(centroids_path, index=False, encoding="utf-8")

    # Quick summary (helps reproducibility logs)
    n = len(labels)
    noise = int((labels == -1).sum())
    k = len(set(labels.tolist()) - {-1})
    print(f"[OK] wrote {events_path}")
    print(f"[OK] wrote {centroids_path}")
    print(f"[SUMMARY] N={n}  K'={k}  noise_fraction={noise/n:.3f}  split_by_time={do_split} gap={split_gap}")

if __name__ == "__main__":
    main()