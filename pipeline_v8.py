#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Illegal Logging Event Reconstruction Pipeline — v8 (unified)
=============================================================

Single-file pipeline with three subcommands:

  python pipeline_v8.py grid_search  --input proteste.xlsx --outdir results/ ...
  python pipeline_v8.py reconstruct  --input proteste.xlsx --params results/best_params.json --outdir results/
  python pipeline_v8.py figures      --input proteste.xlsx --events results/events.csv ...
  python pipeline_v8.py run_all      --input /Users/bogdanoancea/OneDrive/papers/2026/Paduri/output_proteste/proteste.xlsx --outdir results/ \
    --alpha 0.60 0.75 --gamma 0.05 0.10 0.15 \
    --tau_days 2.0 5.0 7.0 --top_k 5 10 15 \
    --min_cluster_size 3 4 5 --min_samples 1 2 \
    --time_window_days 7 \
    --require_dates_for_edges \
    --use_soft_gate \
    --time_gate_eps 0.01 \
    --split_by_time \
    --stability_boot 25 \
    --split_gap_days 7 \
    --split_min_segment_size 2 \
    --min_clusters 5 \
--max_cluster_size 80 \
--max_median_span 30 \
--max_p90_span 90 \
--stability_topk 10 \







Merges functionality from:
  - step1_grid_search_eventlike_gate_v7.py  (grid search + stability)
  - step2_event_reconstruction.py           (event assignment + centroids)
  - make_all_figures.py                     (all paper figures)

v8 changes vs v7
-----------------
- All three scripts merged into one file with subcommands.
- `run_all` subcommand chains grid_search → reconstruct → figures automatically.
- No cross-file imports needed.
- Figure generation reuses in-memory similarity matrix when run via `run_all`.
- Year tick labels fixed to integers (no decimals).
"""

# =====================================================================
# Imports
# =====================================================================

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import hdbscan

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except ImportError:
    HAVE_ST = False


# =====================================================================
# 1. Core utilities (shared across all subcommands)
# =====================================================================

_WS = re.compile(r"\s+")

def norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())

def clean(s: str) -> str:
    s = re.sub(r"https?://\S+", " ", s or "")
    return norm(s.replace("\u00a0", " "))

def build_doc(t: str, x: str) -> str:
    t, x = clean(t), clean(x)
    return f"{t}\n\n{x}" if t and x else (t or x)

def parse_date_iso(s: object) -> Optional[datetime]:
    try:
        return datetime.strptime(str(s), "%Y-%m-%d")
    except Exception:
        return None

def ensure_datetime_pd(s):
    """Robust pandas datetime parsing."""
    return pd.to_datetime(s, errors="coerce", utc=False)


# =====================================================================
# 2. Entity extraction
# =====================================================================

def heuristic_entities(doc: str) -> List[str]:
    ents = set()
    for m in re.finditer(r"\b[A-ZĂÂÎȘȚ]{2,}\b", doc):
        ents.add(m.group(0))
    toks = re.findall(r"[A-Za-zĂÂÎȘȚăâîșț]+|\d+", doc)
    i = 0
    while i < len(toks):
        if toks[i] and toks[i][0].isupper():
            j = i + 1
            while j < len(toks) and toks[j] and toks[j][0].isupper():
                j += 1
            if 2 <= j - i <= 6:
                ents.add(" ".join(toks[i:j]))
            i = j
        else:
            i += 1
    return sorted([norm(e) for e in ents if len(e) >= 3])


# =====================================================================
# 3. Embeddings
# =====================================================================

def embed(texts: List[str], model: str):
    if HAVE_ST:
        st = SentenceTransformer(model)
        emb = st.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32), "st:" + model
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    return X, "tfidf"


# =====================================================================
# 4. Similarity matrix (temporally gated, top-k sparsified)
# =====================================================================

def jacc(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def time_kernel(d1: Optional[datetime], d2: Optional[datetime],
                tau_days: float) -> float:
    if d1 is None or d2 is None:
        return 0.0
    return math.exp(-abs((d1 - d2).days) / float(tau_days))

def sim_matrix_timefiltered_topk(
    emb,
    ents: List[List[str]],
    dates: List[Optional[datetime]],
    alpha: float, beta: float, gamma: float,
    tau_days: float, top_k: int,
    time_window_days: int = 0,
    require_dates_for_edges: bool = True,
    use_soft_gate: bool = True,
    time_gate_eps: float = 0.01,
) -> np.ndarray:
    """
    Time-filtered top-k neighbor selection:
      - candidates restricted to ± time_window_days around date_i
      - then choose semantic top-k among candidates
    """
    n = len(ents)
    k = min(int(top_k), n)

    if hasattr(emb, "tocsr") or hasattr(emb, "tocsc"):
        sem = cosine_similarity(emb)
    else:
        sem = np.clip(emb @ emb.T, -1.0, 1.0)

    daynum = np.array(
        [d.toordinal() if d is not None else -1 for d in dates],
        dtype=np.int64,
    )
    S = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        if require_dates_for_edges and daynum[i] < 0:
            continue

        if time_window_days > 0 and daynum[i] >= 0:
            cand = np.where(
                (daynum >= 0)
                & (np.abs(daynum - daynum[i]) <= int(time_window_days))
            )[0]
        else:
            cand = np.arange(n, dtype=np.int64)

        cand = cand[cand != i]
        if len(cand) == 0:
            continue

        sims = sem[i, cand]
        kk = min(k, len(cand))
        top_idx = np.argpartition(-sims, kk - 1)[:kk]
        idx = cand[top_idx]

        for j in idx:
            di, dj = dates[i], dates[j]

            if require_dates_for_edges and (di is None or dj is None):
                continue
            if time_window_days > 0 and di is not None and dj is not None:
                if abs((di - dj).days) > int(time_window_days):
                    continue

            T = time_kernel(di, dj, tau_days=float(tau_days))
            E = jacc(ents[i], ents[j])
            Sem = float(sem[i, j])

            if use_soft_gate:
                S[i, j] = (float(alpha) * T
                           + (float(beta) * Sem + float(gamma) * E)
                           * (float(time_gate_eps) + T))
            else:
                S[i, j] = (float(alpha) * T
                           + float(beta) * Sem
                           + float(gamma) * E)

    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S


# =====================================================================
# 5. Clustering + temporal post-splitting
# =====================================================================

def cluster_hdbscan(S: np.ndarray, mcs: int, ms: int) -> np.ndarray:
    D = 1.0 - np.clip(S, 0.0, 1.0)
    D = np.ascontiguousarray(D, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    c = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(mcs),
        min_samples=int(ms),
    )
    return c.fit_predict(D)


def split_labels_by_time(
    labels: np.ndarray,
    dates: List[Optional[datetime]],
    split_gap_days: int,
    min_segment_size: int = 2,
) -> np.ndarray:
    """
    Split each non-noise cluster into time-contiguous segments.
    Cut whenever the gap between consecutive dated articles exceeds
    split_gap_days.  Segments smaller than min_segment_size → noise.
    Undated items are conservatively assigned as noise.
    """
    n = len(labels)
    new_labels = np.full(n, -1, dtype=int)

    by: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        by.setdefault(int(lab), []).append(i)

    next_lab = 0
    for lab, idxs in by.items():
        idxs = sorted(
            idxs,
            key=lambda i: (dates[i] is None, dates[i] or datetime.max),
        )
        dated = [i for i in idxs if dates[i] is not None]
        undated = [i for i in idxs if dates[i] is None]

        if len(dated) == 0:
            if len(idxs) >= min_segment_size:
                for i in idxs:
                    new_labels[i] = next_lab
                next_lab += 1
            continue

        # split dated indices by gap
        segments: List[List[int]] = []
        cur = [dated[0]]
        for i in dated[1:]:
            if (dates[i] - dates[cur[-1]]).days > int(split_gap_days):
                segments.append(cur)
                cur = [i]
            else:
                cur.append(i)
        segments.append(cur)

        seg_ids: List[int] = []
        for seg in segments:
            if len(seg) < min_segment_size:
                for i in seg:
                    new_labels[i] = -1
                continue
            seg_id = next_lab
            next_lab += 1
            seg_ids.append(seg_id)
            for i in seg:
                new_labels[i] = seg_id

        # undated items → noise (conservative)
        for u in undated:
            new_labels[u] = -1

    return new_labels


# =====================================================================
# 6. Diagnostics
# =====================================================================

def cluster_stats(labels: np.ndarray) -> Dict[str, float]:
    n = len(labels)
    noise = int((labels == -1).sum())
    labs = labels[labels != -1]
    uniq = set(labs.tolist())
    sizes = [int((labels == k).sum()) for k in uniq]
    return {
        "noise_fraction": float(noise / n) if n else 0.0,
        "n_clusters": float(len(uniq)),
        "cluster_size_max": float(max(sizes)) if sizes else 0.0,
        "cluster_size_median": float(np.median(sizes)) if sizes else 0.0,
    }


def temporal_spans(
    labels: np.ndarray, dates: List[Optional[datetime]]
) -> Dict[str, float]:
    clusters: Dict[int, List[int]] = {}
    for i, l in enumerate(labels):
        if int(l) == -1:
            continue
        clusters.setdefault(int(l), []).append(i)

    spans = []
    for idxs in clusters.values():
        ds = [dates[i] for i in idxs if dates[i] is not None]
        if len(ds) >= 2:
            spans.append((max(ds) - min(ds)).days)

    if not spans:
        return {
            "median_span_days": float("nan"),
            "p90_span_days": float("nan"),
            "max_span_days": float("nan"),
        }
    arr = np.asarray(spans, dtype=float)
    return {
        "median_span_days": float(np.median(arr)),
        "p90_span_days": float(np.quantile(arr, 0.9)),
        "max_span_days": float(np.max(arr)),
    }


def silhouette_cosine(emb, labels: np.ndarray) -> float:
    mask = labels != -1
    if mask.sum() < 3:
        return float("nan")
    labs = labels[mask]
    if len(set(labs.tolist())) < 2:
        return float("nan")

    if hasattr(emb, "tocsr") or hasattr(emb, "tocsc"):
        X = emb[mask]
        sim = cosine_similarity(X)
    else:
        X = emb[mask]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        sim = X @ X.T

    D = 1.0 - np.clip(sim, 0.0, 1.0)
    D = np.ascontiguousarray(D, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    return float(silhouette_score(D, labs, metric="precomputed"))


# =====================================================================
# 7. Stability evaluation (bootstrap ARI)
# =====================================================================

def stability_eval(
    emb,
    ents: List[List[str]],
    dates: List[Optional[datetime]],
    sim_params: Dict[str, object],
    clust_params: Dict[str, int],
    split_by_time: bool,
    split_gap_days: int,
    split_min_segment_size: int,
    n_boot: int,
    frac: float,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    n = len(ents)

    S = sim_matrix_timefiltered_topk(emb, ents, dates, **sim_params)
    base = cluster_hdbscan(
        S, clust_params["min_cluster_size"], clust_params["min_samples"]
    )
    if split_by_time:
        base = split_labels_by_time(
            base, dates,
            split_gap_days=split_gap_days,
            min_segment_size=split_min_segment_size,
        )

    aris = []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, int(frac * n), replace=False)
        idx.sort()

        emb_s = emb[idx]
        ents_s = [ents[i] for i in idx]
        dates_s = [dates[i] for i in idx]

        sim_params_s = dict(sim_params)
        sim_params_s["top_k"] = min(
            int(sim_params_s["top_k"]), len(idx)
        )

        S_s = sim_matrix_timefiltered_topk(
            emb_s, ents_s, dates_s, **sim_params_s
        )
        lab_s = cluster_hdbscan(
            S_s, clust_params["min_cluster_size"],
            clust_params["min_samples"],
        )
        if split_by_time:
            lab_s = split_labels_by_time(
                lab_s, dates_s,
                split_gap_days=split_gap_days,
                min_segment_size=split_min_segment_size,
            )

        aris.append(adjusted_rand_score(base[idx], lab_s))

    return {
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris)),
        "aris": [float(a) for a in aris],
    }


def stability_eval_simple(
    emb, ents, dates, params,
    n_boot=25, frac=0.8, seed=123,
):
    """Convenience wrapper that unpacks a params dict."""
    sim_kwargs = dict(
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        gamma=float(params["gamma"]),
        tau_days=float(params["tau_days"]),
        top_k=int(params["top_k"]),
        time_window_days=int(params.get("time_window_days", 7)),
        require_dates_for_edges=bool(
            params.get("require_dates_for_edges", True)
        ),
        use_soft_gate=bool(params.get("use_soft_gate", True)),
        time_gate_eps=float(params.get("time_gate_eps", 0.01)),
    )
    clust_params = dict(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
    )
    do_split = bool(params.get("split_by_time", True))
    split_gap = int(
        params.get("split_gap_days", params.get("time_window_days", 7))
    )
    split_min = int(params.get("split_min_segment_size", 2))

    return stability_eval(
        emb, ents, dates,
        sim_params=sim_kwargs,
        clust_params=clust_params,
        split_by_time=do_split,
        split_gap_days=split_gap,
        split_min_segment_size=split_min,
        n_boot=n_boot, frac=frac, seed=seed,
    )


# =====================================================================
# 8. Data loading helper (shared by subcommands)
# =====================================================================

def load_corpus(input_path: str, st_model: str):
    """Load XLSX, build docs/dates/entities/embeddings. Returns dict."""
    df = pd.read_excel(input_path)
    docs = [
        build_doc(
            str(r.get("title", "") or ""),
            str(r.get("text", "") or ""),
        )
        for _, r in df.iterrows()
    ]
    dates = [
        parse_date_iso(x)
        for x in df.get("date_iso", [None] * len(df))
    ]
    n_dates = sum(1 for d in dates if d is not None)
    print(
        f"[INFO] date_iso coverage: {n_dates}/{len(dates)}"
        f" = {n_dates / len(dates):.3f}"
    )
    ents = [heuristic_entities(d) for d in docs]
    emb, rep = embed(docs, st_model)
    print(f"[INFO] Embeddings: {rep}")
    return dict(df=df, docs=docs, dates=dates, ents=ents, emb=emb)


def _sim_kwargs_from_params(p: dict) -> dict:
    return dict(
        alpha=float(p["alpha"]),
        beta=float(p["beta"]),
        gamma=float(p["gamma"]),
        tau_days=float(p["tau_days"]),
        top_k=int(p["top_k"]),
        time_window_days=int(p.get("time_window_days", 7)),
        require_dates_for_edges=bool(
            p.get("require_dates_for_edges", True)
        ),
        use_soft_gate=bool(p.get("use_soft_gate", True)),
        time_gate_eps=float(p.get("time_gate_eps", 0.01)),
    )


# =====================================================================
# 9. SUBCOMMAND: grid_search
# =====================================================================

def cmd_grid_search(args):
    """Run hyperparameter grid search with stability selection."""
    os.makedirs(args.outdir, exist_ok=True)

    corpus = load_corpus(args.input, args.st_model)
    df, dates, ents, emb = (
        corpus["df"], corpus["dates"], corpus["ents"], corpus["emb"],
    )

    split_gap = (
        int(args.split_gap_days) if int(args.split_gap_days) > 0
        else int(args.time_window_days)
    )

    all_rows = []
    filtered = []
    run_id = 0

    for a in args.alpha:
        for g in args.gamma:
            b = 1.0 - float(a) - float(g)
            if b < 0:
                continue
            for tau in args.tau_days:
                for tk in args.top_k:
                    for mcs in args.min_cluster_size:
                        for ms in args.min_samples:
                            run_id += 1
                            S = sim_matrix_timefiltered_topk(
                                emb, ents, dates,
                                alpha=float(a), beta=float(b),
                                gamma=float(g),
                                tau_days=float(tau), top_k=int(tk),
                                time_window_days=int(
                                    args.time_window_days
                                ),
                                require_dates_for_edges=bool(
                                    args.require_dates_for_edges
                                ),
                                use_soft_gate=bool(args.use_soft_gate),
                                time_gate_eps=float(args.time_gate_eps),
                            )
                            lab = cluster_hdbscan(S, int(mcs), int(ms))
                            lab_post = lab
                            if args.split_by_time:
                                lab_post = split_labels_by_time(
                                    lab_post, dates,
                                    split_gap_days=split_gap,
                                    min_segment_size=int(
                                        args.split_min_segment_size
                                    ),
                                )

                            st = cluster_stats(lab_post)
                            sp = temporal_spans(lab_post, dates)
                            sil = silhouette_cosine(emb, lab_post)

                            row = dict(
                                run_id=run_id,
                                alpha=float(a), beta=float(b),
                                gamma=float(g),
                                tau_days=float(tau), top_k=int(tk),
                                min_cluster_size=int(mcs),
                                min_samples=int(ms),
                                time_window_days=int(
                                    args.time_window_days
                                ),
                                require_dates_for_edges=bool(
                                    args.require_dates_for_edges
                                ),
                                use_soft_gate=bool(args.use_soft_gate),
                                time_gate_eps=float(args.time_gate_eps),
                                split_by_time=bool(args.split_by_time),
                                split_gap_days=int(split_gap),
                                split_min_segment_size=int(
                                    args.split_min_segment_size
                                ),
                                silhouette_cosine=float(sil),
                                noise_fraction=float(
                                    st["noise_fraction"]
                                ),
                                n_clusters=float(st["n_clusters"]),
                                cluster_size_max=float(
                                    st["cluster_size_max"]
                                ),
                                cluster_size_median=float(
                                    st["cluster_size_median"]
                                ),
                                median_span_days=float(
                                    sp["median_span_days"]
                                ),
                                p90_span_days=float(sp["p90_span_days"]),
                                max_span_days=float(sp["max_span_days"]),
                            )
                            all_rows.append(row)

                            # constraint filtering
                            ok = True
                            if st["n_clusters"] < args.min_clusters:
                                ok = False
                            if st["cluster_size_max"] > args.max_cluster_size:
                                ok = False
                            spans_nan = not (
                                sp["median_span_days"]
                                == sp["median_span_days"]
                                and sp["p90_span_days"]
                                == sp["p90_span_days"]
                            )
                            if spans_nan:
                                if not args.allow_missing_dates:
                                    ok = False
                            else:
                                if sp["median_span_days"] > args.max_median_span:
                                    ok = False
                                if sp["p90_span_days"] > args.max_p90_span:
                                    ok = False
                            if ok:
                                filtered.append(row)

    pd.DataFrame(all_rows).to_csv(
        os.path.join(args.outdir, "grid_results.csv"),
        index=False, encoding="utf-8",
    )

    if not filtered:
        print(
            "[WARN] No runs passed constraints. "
            "Try --split_by_time or relax span thresholds."
        )
        pd.DataFrame([]).to_csv(
            os.path.join(args.outdir, "grid_results_filtered.csv"),
            index=False, encoding="utf-8",
        )
        return

    df_f = (
        pd.DataFrame(filtered)
        .sort_values("silhouette_cosine", ascending=False)
        .reset_index(drop=True)
    )
    df_f.to_csv(
        os.path.join(args.outdir, "grid_results_filtered.csv"),
        index=False, encoding="utf-8",
    )

    # ── Stability selection among top-K ──
    K = min(int(args.stability_topk), len(df_f))
    best = None
    best_key = -1e18
    best_stab = None

    for cand in df_f.head(K).to_dict("records"):
        sim_params = _sim_kwargs_from_params(cand)
        clust_params = dict(
            min_cluster_size=int(cand["min_cluster_size"]),
            min_samples=int(cand["min_samples"]),
        )
        stab = stability_eval(
            emb, ents, dates,
            sim_params=sim_params,
            clust_params=clust_params,
            split_by_time=bool(args.split_by_time),
            split_gap_days=int(split_gap),
            split_min_segment_size=int(args.split_min_segment_size),
            n_boot=int(args.stability_boot),
            frac=float(args.stability_frac),
            seed=int(args.stability_seed),
        )
        key = (
            float(cand["silhouette_cosine"])
            + 0.02 * float(stab["mean_ari"])
        )
        print(
            f"  Run {int(cand['run_id'])} "
            f"ARI {stab['mean_ari']:.3f} ± {stab['std_ari']:.3f}"
        )
        if key > best_key:
            best_key = key
            best = cand
            best_stab = stab

    payload = {
        "best": best,
        "stability_mean": best_stab["mean_ari"],
        "stability_sd": best_stab["std_ari"],
    }
    params_path = os.path.join(args.outdir, "best_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {params_path}")
    print(f"     Selected: run_id={best['run_id']}")
    print(
        f"     ARI = {best_stab['mean_ari']:.3f}"
        f" ± {best_stab['std_ari']:.3f}"
    )


# =====================================================================
# 10. SUBCOMMAND: reconstruct
# =====================================================================

def compute_event_centroids(
    df: pd.DataFrame, labels: np.ndarray,
) -> pd.DataFrame:
    """Build per-event centroid table."""
    out = []
    df2 = df.copy()
    df2["event_id"] = labels
    df2["date_dt"] = pd.to_datetime(
        df2.get("date_iso", pd.Series([None] * len(df2))),
        errors="coerce",
    )

    for event_id, g in df2.groupby("event_id", dropna=False):
        event_id = int(event_id)
        if event_id == -1:
            continue
        g = g.sort_values(
            ["date_dt", "doc_id"], ascending=[True, True]
        )
        date_min = g["date_dt"].min()
        date_max = g["date_dt"].max()
        span_days = (
            (date_max - date_min).days
            if pd.notna(date_min) and pd.notna(date_max)
            else np.nan
        )
        rep_row = g.iloc[0]
        out.append({
            "event_id": event_id,
            "n_articles": int(len(g)),
            "date_min": (
                date_min.date().isoformat()
                if pd.notna(date_min) else ""
            ),
            "date_max": (
                date_max.date().isoformat()
                if pd.notna(date_max) else ""
            ),
            "span_days": (
                float(span_days) if pd.notna(span_days) else np.nan
            ),
            "representative_doc_id": int(rep_row["doc_id"]),
            "representative_title": (
                str(rep_row.get("title", ""))
                if pd.notna(rep_row.get("title", ""))
                else ""
            ),
        })

    return (
        pd.DataFrame(out)
        .sort_values(["n_articles", "event_id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def cmd_reconstruct(args, corpus=None, S_precomputed=None):
    """
    Apply best params to produce events.csv + event_centroids.csv.
    Optionally accepts pre-loaded corpus/similarity to avoid recomputation
    when called from run_all.
    Returns (events_df, centroids_df, S) for downstream use.
    """
    os.makedirs(args.outdir, exist_ok=True)

    # Load params
    with open(args.params, "r", encoding="utf-8") as f:
        payload = json.load(f)
    p = payload["best"]

    # Load or reuse corpus
    if corpus is None:
        corpus = load_corpus(args.input, args.st_model)
    df = corpus["df"].copy()
    dates, ents, emb = corpus["dates"], corpus["ents"], corpus["emb"]
    df["doc_id"] = np.arange(len(df), dtype=int)

    # Similarity
    if S_precomputed is not None:
        S = S_precomputed
    else:
        S = sim_matrix_timefiltered_topk(
            emb, ents, dates, **_sim_kwargs_from_params(p)
        )

    # HDBSCAN
    D = 1.0 - np.clip(S, 0.0, 1.0)
    D = np.ascontiguousarray(D, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(p["min_cluster_size"]),
        min_samples=int(p["min_samples"]),
    )
    labels_raw = clusterer.fit_predict(D).astype(int)

    # Post-splitting
    do_split = bool(p.get("split_by_time", True))
    split_gap = int(
        p.get("split_gap_days", p.get("time_window_days", 7))
    )
    split_min_seg = int(p.get("split_min_segment_size", 2))

    labels = labels_raw
    if do_split:
        labels = split_labels_by_time(
            labels, dates,
            split_gap_days=split_gap,
            min_segment_size=split_min_seg,
        ).astype(int)

    # Save
    df_out = df.copy()
    df_out["event_id_raw"] = labels_raw
    df_out["event_id"] = labels

    events_path = os.path.join(args.outdir, "events.csv")
    df_out.to_csv(events_path, index=False, encoding="utf-8")

    centroids = compute_event_centroids(df_out, labels)
    centroids_path = os.path.join(args.outdir, "event_centroids.csv")
    centroids.to_csv(centroids_path, index=False, encoding="utf-8")

    n = len(labels)
    noise = int((labels == -1).sum())
    k = len(set(labels.tolist()) - {-1})
    print(f"[OK] {events_path}")
    print(f"[OK] {centroids_path}")
    print(
        f"[SUMMARY] N={n}  K'={k}  "
        f"noise_fraction={noise / n:.3f}  "
        f"split_by_time={do_split} gap={split_gap}"
    )

    return df_out, centroids, S


# =====================================================================
# 11. SUBCOMMAND: figures
# =====================================================================

# ── Figure: Articles per year ────────────────────────────────────────

def plot_articles_per_year(events_df, outpath):
    dt = ensure_datetime_pd(events_df.get("date_iso"))
    tmp = pd.DataFrame({"dt": dt}).dropna(subset=["dt"])
    if tmp.empty:
        print("[WARN] No dated articles for articles-per-year.")
        return
    tmp["year"] = tmp["dt"].dt.year
    counts = tmp.groupby("year").size().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(int), counts.values,
           color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Number of Articles per Year")
    ax.set_xticks(counts.index.astype(int))
    ax.set_xticklabels(
        [str(int(y)) for y in counts.index], rotation=45, ha="right"
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Character length distribution ────────────────────────────

def plot_char_length_distribution(events_df, outpath, bins=30):
    texts = (
        events_df["title"].fillna("").astype(str)
        + " "
        + events_df["text"].fillna("").astype(str)
    )
    char_lens = texts.str.len()
    char_lens = char_lens[char_lens > 0]
    if char_lens.empty:
        print("[WARN] No text data for character length plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(char_lens, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Character Length")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Word count distribution ──────────────────────────────────

def plot_word_count_distribution(events_df, outpath, bins=30):
    texts = (
        events_df["title"].fillna("").astype(str)
        + " "
        + events_df["text"].fillna("").astype(str)
    )
    word_counts = texts.str.split().str.len()
    word_counts = word_counts[word_counts > 0]
    if word_counts.empty:
        print("[WARN] No text data for word count plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(word_counts, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Word Count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Event size CCDF ──────────────────────────────────────────

def plot_event_size_ccdf(cent, outpath):
    sizes = (
        cent.loc[cent["event_id"] != -1, "n_articles"]
        .dropna().astype(int).values
    )
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        print("[WARN] No event sizes for CCDF.")
        return

    x = np.sort(sizes)
    y = 1.0 - np.arange(len(x)) / float(len(x))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(x, y, where="post", linewidth=1.5)
    ax.set_xlabel("Event size (number of articles)")
    ax.set_ylabel("CCDF")
    ax.set_title("Event Size Distribution (CCDF)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Event span distribution ──────────────────────────────────

def plot_event_span_distribution(cent, outpath, bins=12):
    if "span_days" not in cent.columns:
        d0 = ensure_datetime_pd(cent.get("date_min"))
        d1 = ensure_datetime_pd(cent.get("date_max"))
        cent = cent.copy()
        cent["span_days"] = (d1 - d0).dt.days

    spans = (
        cent.loc[cent["event_id"] != -1, "span_days"]
        .dropna().astype(float).values
    )
    spans = spans[spans >= 0]
    if len(spans) == 0:
        print("[WARN] No span data for span distribution.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(spans, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Event span (days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Event Temporal Span")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Event timeline (events per year) ─────────────────────────

def plot_event_timeline_per_year(cent, outpath):
    d0 = ensure_datetime_pd(cent.get("date_min"))
    tmp = cent.copy()
    tmp["date_min_dt"] = d0
    tmp = tmp[(tmp["event_id"] != -1) & tmp["date_min_dt"].notna()]
    if tmp.empty:
        print("[WARN] No dated events for events-per-year.")
        return

    tmp["year"] = tmp["date_min_dt"].dt.year
    s = tmp.groupby("year").size().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s.index.astype(int), s.values, marker="o", linewidth=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of reconstructed events")
    ax.set_title("Event Timeline (Events per Year)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Bursts (articles vs events per month) ────────────────────

def plot_bursts_articles_vs_events(events_df, cent, outpath):
    dt = ensure_datetime_pd(events_df.get("date_iso"))
    a = pd.DataFrame({"dt": dt}).dropna(subset=["dt"])
    if a.empty:
        print("[WARN] No dated articles for burst plot.")
        return
    a["ym"] = a["dt"].dt.to_period("M").dt.to_timestamp()
    articles_pm = a.groupby("ym").size().sort_index()

    d0 = ensure_datetime_pd(cent.get("date_min"))
    e = cent.copy()
    e["date_min_dt"] = d0
    e = e[(e["event_id"] != -1) & e["date_min_dt"].notna()]
    if e.empty:
        print("[WARN] No dated events for burst plot.")
        return
    e["ym"] = e["date_min_dt"].dt.to_period("M").dt.to_timestamp()
    events_pm = e.groupby("ym").size().sort_index()

    idx = articles_pm.index.union(events_pm.index)
    articles_pm = articles_pm.reindex(idx, fill_value=0)
    events_pm = events_pm.reindex(idx, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, articles_pm.values, marker="o", markersize=3,
            label="Articles/month")
    ax.plot(idx, events_pm.values, marker="o", markersize=3,
            label="Events/month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_title("Bursts: Articles vs Reconstructed Events (per month)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Noise vs clustered ───────────────────────────────────────

def plot_noise_distribution(events_df, outpath):
    labels = events_df["event_id"].values
    n = len(labels)
    n_noise = int((labels == -1).sum())
    n_clustered = n - n_noise

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Noise", "Clustered"],
           [n_noise / n, n_clustered / n],
           color=["#1f77b4", "#1f77b4"], edgecolor="white")
    ax.set_ylabel("Proportion")
    ax.set_title("Noise vs Clustered Articles")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Cluster size CCDF (pre-segmentation) ────────────────────

def plot_cluster_size_ccdf_raw(events_df, outpath):
    raw_labels = events_df["event_id_raw"].values
    nonnoise = raw_labels[raw_labels != -1]
    if len(nonnoise) == 0:
        print("[WARN] No raw clusters for pre-split CCDF.")
        return

    _, counts = np.unique(nonnoise, return_counts=True)
    sizes = np.sort(counts)
    y = 1.0 - np.arange(len(sizes)) / float(len(sizes))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(sizes, y, where="post", linewidth=1.5)
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("CCDF")
    ax.set_title("Cluster Size Complementary CDF")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Degree distribution ──────────────────────────────────────

def plot_degree_distribution(S, outpath, bins=20):
    A = S.copy()
    np.fill_diagonal(A, 0.0)
    degrees = A.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(degrees, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Weighted degree")
    ax.set_ylabel("Count")
    ax.set_title("Similarity Graph Weighted Degree Distribution")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure: Laplacian spectrum ───────────────────────────────────────

def plot_spectrum_laplacian(S, outpath, n_eigenvalues=60):
    A = S.copy()
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)

    d_inv_sqrt = np.zeros_like(d)
    nonzero = d > 1e-12
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)

    n = A.shape[0]
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    eigvals = np.sort(np.linalg.eigvalsh(L))
    k = min(n_eigenvalues, len(eigvals))
    eigvals_plot = eigvals[:k]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, k + 1), eigvals_plot, linewidth=1.5)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue (normalized Laplacian)")
    ax.set_title(f"Similarity Graph Spectrum (first {k} eigenvalues)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)

    near_zero = int(np.sum(eigvals < 1e-6))
    first_nz = (
        eigvals[eigvals > 1e-6][0] if np.any(eigvals > 1e-6)
        else float("nan")
    )
    print(f"  [INFO] Near-zero eigenvalues: {near_zero}")
    print(f"  [INFO] Spectral gap (λ₂): {first_nz:.4f}")


# ── Figure: Bootstrap stability distribution ─────────────────────────

def plot_stability_distribution(
    emb, ents, dates, params, outpath,
    n_boot=25, frac=0.8, seed=123, bins=15,
):
    stab = stability_eval_simple(
        emb, ents, dates, params,
        n_boot=n_boot, frac=frac, seed=seed,
    )
    aris = np.array(stab["aris"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(aris, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Stability Distribution")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)
    print(
        f"  [INFO] ARI mean = {aris.mean():.3f} ± {aris.std():.3f}"
    )


def cmd_figures(
    args, events_df=None, cent=None, S=None, corpus=None,
):
    """
    Generate all paper figures.
    Optionally accepts pre-computed objects from run_all to avoid
    reloading / recomputing.
    """
    os.makedirs(args.outdir, exist_ok=True)

    # Load CSVs if not provided in memory
    if events_df is None:
        events_df = pd.read_csv(args.events)
    if cent is None:
        cent = pd.read_csv(args.centroids)

    with open(args.params, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload["best"]

    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("=" * 60)
    print("FIGURES — Phase 1: from CSV data")
    print("=" * 60)

    plot_articles_per_year(
        events_df, os.path.join(figdir, "no_articles_per_year.png"))
    plot_char_length_distribution(
        events_df, os.path.join(figdir, "char_length_distribution.png"),
        bins=getattr(args, "bins_char", 30))
    plot_word_count_distribution(
        events_df, os.path.join(figdir, "word_count_distribution.png"),
        bins=getattr(args, "bins_char", 30))
    plot_event_size_ccdf(
        cent, os.path.join(figdir, "event_size_ccdf.png"))
    plot_event_span_distribution(
        cent, os.path.join(figdir, "event_span_distribution.png"),
        bins=getattr(args, "bins_span", 12))
    plot_event_timeline_per_year(
        cent, os.path.join(figdir, "event_timeline_per_year.png"))
    plot_bursts_articles_vs_events(
        events_df, cent,
        os.path.join(figdir, "bursts_articles_vs_events_per_month.png"))
    plot_noise_distribution(
        events_df, os.path.join(figdir, "fig_noise_distribution.png"))
    plot_cluster_size_ccdf_raw(
        events_df, os.path.join(figdir, "fig_cluster_size_ccdf.png"))

    if getattr(args, "skip_recompute", False):
        print("\n[SKIP] Recomputation figures skipped (--skip_recompute).")
        return

    print()
    print("=" * 60)
    print("FIGURES — Phase 2: similarity graph diagnostics")
    print("=" * 60)

    # Recompute similarity if not provided
    if S is None:
        if corpus is None:
            corpus = load_corpus(
                args.input,
                getattr(args, "st_model",
                        "paraphrase-multilingual-MiniLM-L12-v2"),
            )
        S = sim_matrix_timefiltered_topk(
            corpus["emb"], corpus["ents"], corpus["dates"],
            **_sim_kwargs_from_params(params),
        )
    if corpus is None:
        corpus = load_corpus(
            args.input,
            getattr(args, "st_model",
                    "paraphrase-multilingual-MiniLM-L12-v2"),
        )

    plot_degree_distribution(
        S, os.path.join(figdir, "fig_degree_distribution.png"))
    plot_spectrum_laplacian(
        S, os.path.join(figdir, "fig_spectrum_laplacian.png"))

    print()
    print("=" * 60)
    print("FIGURES — Phase 3: bootstrap stability")
    print("=" * 60)

    plot_stability_distribution(
        corpus["emb"], corpus["ents"], corpus["dates"], params,
        os.path.join(figdir, "fig_stability_distribution.png"),
        n_boot=getattr(args, "stability_boot", 25),
        frac=getattr(args, "stability_frac", 0.8),
        seed=getattr(args, "stability_seed", 123),
    )


# =====================================================================
# 12. SUBCOMMAND: run_all (grid_search → reconstruct → figures)
# =====================================================================

def cmd_run_all(args):
    """Execute the full pipeline: grid_search → reconstruct → figures."""
    print("=" * 60)
    print("PHASE 1 / 3 : GRID SEARCH")
    print("=" * 60)
    cmd_grid_search(args)

    print()
    print("=" * 60)
    print("PHASE 2 / 3 : EVENT RECONSTRUCTION")
    print("=" * 60)
    # Point --params to the just-created best_params.json
    args.params = os.path.join(args.outdir, "best_params.json")
    events_df, centroids, S = cmd_reconstruct(args)

    print()
    print("=" * 60)
    print("PHASE 3 / 3 : FIGURES")
    print("=" * 60)
    # Point --events / --centroids to the just-created CSVs
    args.events = os.path.join(args.outdir, "events.csv")
    args.centroids = os.path.join(args.outdir, "event_centroids.csv")

    # Reload corpus for figure recomputation (embeddings needed)
    corpus = load_corpus(
        args.input,
        getattr(args, "st_model",
                "paraphrase-multilingual-MiniLM-L12-v2"),
    )
    cmd_figures(args, events_df=events_df, cent=centroids,
                S=S, corpus=corpus)

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# =====================================================================
# 13. Argument parsing and dispatch
# =====================================================================

def _add_common_args(p):
    """Arguments shared by all subcommands."""
    p.add_argument("--input", required=True,
                   help="Input XLSX (proteste.xlsx)")
    p.add_argument("--outdir", required=True,
                   help="Output directory")
    p.add_argument("--st_model",
                   default="paraphrase-multilingual-MiniLM-L12-v2",
                   help="SentenceTransformer model name")


def _add_grid_args(p):
    """Arguments for grid_search (also used by run_all)."""
    p.add_argument("--alpha", nargs="+", type=float, required=True)
    p.add_argument("--gamma", nargs="+", type=float, required=True)
    p.add_argument("--tau_days", nargs="+", type=float, required=True)
    p.add_argument("--top_k", nargs="+", type=int, required=True)
    p.add_argument("--min_cluster_size", nargs="+", type=int,
                   required=True)
    p.add_argument("--min_samples", nargs="+", type=int, required=True)

    p.add_argument("--time_window_days", type=int, default=7)
    p.add_argument("--require_dates_for_edges",
                   action="store_true", default=True)
    p.add_argument("--use_soft_gate",
                   action="store_true", default=True)
    p.add_argument("--time_gate_eps", type=float, default=0.01)

    p.add_argument("--split_by_time", action="store_true",
                   help="Enable temporal post-splitting (recommended)")
    p.add_argument("--split_gap_days", type=int, default=0,
                   help="Gap threshold; 0 = use time_window_days")
    p.add_argument("--split_min_segment_size", type=int, default=2)

    p.add_argument("--min_clusters", type=int, default=5)
    p.add_argument("--max_cluster_size", type=int, default=80)
    p.add_argument("--max_median_span", type=float, default=30.0)
    p.add_argument("--max_p90_span", type=float, default=90.0)
    p.add_argument("--allow_missing_dates", action="store_true")

    p.add_argument("--stability_topk", type=int, default=10)
    p.add_argument("--stability_boot", type=int, default=25)
    p.add_argument("--stability_frac", type=float, default=0.8)
    p.add_argument("--stability_seed", type=int, default=123)


def _add_figure_args(p):
    """Arguments specific to figure generation."""
    p.add_argument("--skip_recompute", action="store_true",
                   help="Skip figures requiring similarity recomputation")
    p.add_argument("--bins_char", type=int, default=30)
    p.add_argument("--bins_span", type=int, default=12)


def main():
    parser = argparse.ArgumentParser(
        description="Illegal Logging Event Reconstruction Pipeline v8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  grid_search   Run hyperparameter grid search with stability selection
  reconstruct   Apply best_params.json to produce events.csv + centroids
  figures       Generate all paper figures
  run_all       Execute full pipeline (grid_search → reconstruct → figures)

Example (full pipeline):
  python pipeline_v8.py run_all \\
      --input proteste.xlsx --outdir results/ \\
      --alpha 0.60 0.75 --gamma 0.05 0.10 0.15 \\
      --tau_days 2.0 5.0 7.0 --top_k 5 15 \\
      --min_cluster_size 3 4 --min_samples 2 \\
      --split_by_time --stability_boot 25
""",
    )
    sub = parser.add_subparsers(dest="command")

    # ── grid_search ──
    p_gs = sub.add_parser("grid_search",
                          help="Hyperparameter grid search")
    _add_common_args(p_gs)
    _add_grid_args(p_gs)

    # ── reconstruct ──
    p_rc = sub.add_parser("reconstruct",
                          help="Event reconstruction from best params")
    _add_common_args(p_rc)
    p_rc.add_argument("--params", required=True,
                      help="best_params.json from grid search")

    # ── figures ──
    p_fig = sub.add_parser("figures",
                           help="Generate all paper figures")
    _add_common_args(p_fig)
    p_fig.add_argument("--events", required=True,
                       help="Path to events.csv")
    p_fig.add_argument("--centroids", required=True,
                       help="Path to event_centroids.csv")
    p_fig.add_argument("--params", required=True,
                       help="Path to best_params.json")
    _add_figure_args(p_fig)
    p_fig.add_argument("--stability_boot", type=int, default=25)
    p_fig.add_argument("--stability_frac", type=float, default=0.8)
    p_fig.add_argument("--stability_seed", type=int, default=123)

    # ── run_all ──
    p_all = sub.add_parser("run_all",
                           help="Full pipeline: grid → reconstruct → figures")
    _add_common_args(p_all)
    _add_grid_args(p_all)
    _add_figure_args(p_all)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "grid_search": cmd_grid_search,
        "reconstruct": cmd_reconstruct,
        "figures": cmd_figures,
        "run_all": cmd_run_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()