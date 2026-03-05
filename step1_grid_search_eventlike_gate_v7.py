#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 (A) — Grid search with temporal gating + temporal post-splitting (v7)
============================================================================

What your v6 diagnostics show
----------------------------
Even with:
  - time_window_days=7
  - require_dates_for_edges=True
  - time-filtered top-k neighbor selection
you still see very large cluster spans (p90 in thousands of days).

This is not necessarily a "bug": a dense sequence of near-consecutive articles can form a
*temporal chain* (transitive closure) so that a single connected component spans years.

Event reconstruction (as used in IR / computational social science) usually applies a
*temporal segmentation* step after initial clustering to enforce that an "event instance"
is temporally compact.

v7 adds a deterministic post-processing step:

  Split each cluster into time-contiguous segments:
    - sort articles in the cluster by date
    - cut whenever the gap between consecutive dated articles exceeds split_gap_days
  Each segment becomes a new event cluster id.

This makes temporal-span diagnostics meaningful and allows you to grid-search the upstream
similarity/clustering hyperparameters while enforcing event compactness.

Key flags
---------
--split_by_time                  Enable temporal post-splitting (recommended)
--split_gap_days <int>           Gap threshold for splitting (default = time_window_days)
--split_min_segment_size <int>   Segments smaller than this are re-labeled as noise (-1)

Outputs
-------
- grid_results.csv
- grid_results_filtered.csv
- best_params.json
"""

import argparse, os, re, math, json
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except Exception:
    HAVE_ST = False

# -------------------- Utilities --------------------
_WS = re.compile(r"\s+")
def norm(s: str) -> str: return _WS.sub(" ", (s or "").strip())
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

# -------------------- Entities --------------------
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

# -------------------- Embeddings --------------------
def embed(texts: List[str], model: str):
    if HAVE_ST:
        st = SentenceTransformer(model)
        emb = st.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32), "st:" + model
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    return X, "tfidf"

# -------------------- Similarity --------------------
def jacc(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def time_kernel(d1: Optional[datetime], d2: Optional[datetime], tau_days: float) -> float:
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
      - candidates restricted to +/- time_window_days around date_i
      - then choose semantic top-k among candidates
    """
    n = len(ents)
    k = min(int(top_k), n)

    if hasattr(emb, "tocsr") or hasattr(emb, "tocsc"):
        sem = cosine_similarity(emb)
    else:
        sem = np.clip(emb @ emb.T, -1.0, 1.0)

    daynum = np.array([d.toordinal() if d is not None else -1 for d in dates], dtype=np.int64)
    S = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        if require_dates_for_edges and daynum[i] < 0:
            continue

        if time_window_days > 0 and daynum[i] >= 0:
            cand = np.where((daynum >= 0) & (np.abs(daynum - daynum[i]) <= int(time_window_days)))[0]
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
                S[i, j] = float(alpha) * T + (float(beta) * Sem + float(gamma) * E) * (float(time_gate_eps) + T)
            else:
                S[i, j] = float(alpha) * T + float(beta) * Sem + float(gamma) * E

    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S

# -------------------- Clustering + post-splitting --------------------
def cluster_hdbscan(S: np.ndarray, mcs: int, ms: int) -> np.ndarray:
    D = 1.0 - np.clip(S, 0.0, 1.0)
    D = np.ascontiguousarray(D, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    c = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=int(mcs), min_samples=int(ms))
    return c.fit_predict(D)

def split_labels_by_time(
    labels: np.ndarray,
    dates: List[Optional[datetime]],
    split_gap_days: int,
    min_segment_size: int = 2,
) -> np.ndarray:
    """
    Split each non-noise cluster into time-contiguous segments based on gaps in dated items.
    Undated items inside a cluster are assigned to the nearest dated segment (by absolute date distance),
    and if a cluster has no dated items, it is left unchanged.

    Segments smaller than min_segment_size are re-labeled as noise (-1).
    """
    n = len(labels)
    new_labels = np.full(n, -1, dtype=int)

    # group indices by old label
    by = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        by.setdefault(int(lab), []).append(i)

    next_lab = 0
    for lab, idxs in by.items():
        idxs = sorted(idxs, key=lambda i: (dates[i] is None, dates[i] or datetime.max))
        dated = [i for i in idxs if dates[i] is not None]
        undated = [i for i in idxs if dates[i] is None]

        if len(dated) == 0:
            # cannot split; keep as one segment
            if len(idxs) >= min_segment_size:
                for i in idxs:
                    new_labels[i] = next_lab
                next_lab += 1
            continue

        # split dated indices by gap
        segments = []
        cur = [dated[0]]
        for i in dated[1:]:
            if (dates[i] - dates[cur[-1]]).days > int(split_gap_days):
                segments.append(cur)
                cur = [i]
            else:
                cur.append(i)
        segments.append(cur)

        # assign labels to dated items
        seg_ids = []
        for seg in segments:
            if len(seg) < min_segment_size:
                # mark these dated items as noise
                for i in seg:
                    new_labels[i] = -1
                continue
            seg_id = next_lab
            next_lab += 1
            seg_ids.append(seg_id)
            for i in seg:
                new_labels[i] = seg_id

        # assign undated items to nearest segment (by distance to segment median date)
        if undated and seg_ids:
            seg_meds = []
            for seg, seg_id in zip([s for s in segments if len(s) >= min_segment_size], seg_ids):
                dts = [dates[i] for i in seg]
                # median date
                ords = sorted([d.toordinal() for d in dts])
                med_ord = ords[len(ords)//2]
                seg_meds.append((seg_id, med_ord))

            for u in undated:
                # cannot compute distance without date; keep as noise (conservative)
                new_labels[u] = -1

    return new_labels

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

def temporal_spans(labels: np.ndarray, dates: List[Optional[datetime]]) -> Dict[str, float]:
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
        return {"median_span_days": float("nan"), "p90_span_days": float("nan"), "max_span_days": float("nan")}

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

# -------------------- Stability --------------------
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
    base = cluster_hdbscan(S, clust_params["min_cluster_size"], clust_params["min_samples"])
    if split_by_time:
        base = split_labels_by_time(base, dates, split_gap_days=split_gap_days, min_segment_size=split_min_segment_size)

    aris = []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, int(frac * n), replace=False)
        idx.sort()

        emb_s = emb[idx] if not (hasattr(emb, "tocsr") or hasattr(emb, "tocsc")) else emb[idx]
        ents_s = [ents[i] for i in idx]
        dates_s = [dates[i] for i in idx]

        sim_params_s = dict(sim_params)
        sim_params_s["top_k"] = min(int(sim_params_s["top_k"]), len(idx))

        S_s = sim_matrix_timefiltered_topk(emb_s, ents_s, dates_s, **sim_params_s)
        lab_s = cluster_hdbscan(S_s, clust_params["min_cluster_size"], clust_params["min_samples"])
        if split_by_time:
            lab_s = split_labels_by_time(lab_s, dates_s, split_gap_days=split_gap_days, min_segment_size=split_min_segment_size)

        aris.append(adjusted_rand_score(base[idx], lab_s))

    return {"mean_ari": float(np.mean(aris)), "std_ari": float(np.std(aris))}

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--alpha", nargs="+", type=float, required=True)
    ap.add_argument("--gamma", nargs="+", type=float, required=True)
    ap.add_argument("--tau_days", nargs="+", type=float, required=True)
    ap.add_argument("--top_k", nargs="+", type=int, required=True)
    ap.add_argument("--min_cluster_size", nargs="+", type=int, required=True)
    ap.add_argument("--min_samples", nargs="+", type=int, required=True)

    ap.add_argument("--time_window_days", type=int, default=7)
    ap.add_argument("--require_dates_for_edges", action="store_true", default=True)

    ap.add_argument("--use_soft_gate", action="store_true", default=True)
    ap.add_argument("--time_gate_eps", type=float, default=0.01)

    # Post-splitting (NEW)
    ap.add_argument("--split_by_time", action="store_true", help="Split clusters into time-contiguous segments.")
    ap.add_argument("--split_gap_days", type=int, default=0, help="Gap threshold; default uses time_window_days.")
    ap.add_argument("--split_min_segment_size", type=int, default=2)

    # constraints (applied after optional splitting)
    ap.add_argument("--min_clusters", type=int, default=5)
    ap.add_argument("--max_cluster_size", type=int, default=80)
    ap.add_argument("--max_median_span", type=float, default=30.0)
    ap.add_argument("--max_p90_span", type=float, default=90.0)
    ap.add_argument("--allow_missing_dates", action="store_true")

    # stability
    ap.add_argument("--stability_topk", type=int, default=10)
    ap.add_argument("--stability_boot", type=int, default=25)
    ap.add_argument("--stability_frac", type=float, default=0.8)
    ap.add_argument("--stability_seed", type=int, default=123)

    ap.add_argument("--st_model", default="paraphrase-multilingual-MiniLM-L12-v2")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_excel(args.input)
    docs = [build_doc(str(r.get("title", "") or ""), str(r.get("text", "") or "")) for _, r in df.iterrows()]
    dates = [parse_date_iso(x) for x in df.get("date_iso", [None] * len(df))]
    n_dates = sum(1 for d in dates if d is not None)
    print(f"[INFO] date_iso coverage: {n_dates}/{len(dates)} = {n_dates/len(dates):.3f}")

    ents = [heuristic_entities(d) for d in docs]
    emb, rep = embed(docs, args.st_model)
    print("Embeddings:", rep)

    split_gap = int(args.split_gap_days) if int(args.split_gap_days) > 0 else int(args.time_window_days)

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
                                alpha=float(a), beta=float(b), gamma=float(g),
                                tau_days=float(tau), top_k=int(tk),
                                time_window_days=int(args.time_window_days),
                                require_dates_for_edges=bool(args.require_dates_for_edges),
                                use_soft_gate=bool(args.use_soft_gate),
                                time_gate_eps=float(args.time_gate_eps),
                            )
                            lab = cluster_hdbscan(S, int(mcs), int(ms))
                            lab_post = lab
                            if args.split_by_time:
                                lab_post = split_labels_by_time(
                                    lab_post, dates,
                                    split_gap_days=split_gap,
                                    min_segment_size=int(args.split_min_segment_size),
                                )

                            st = cluster_stats(lab_post)
                            sp = temporal_spans(lab_post, dates)
                            sil = silhouette_cosine(emb, lab_post)

                            row = dict(
                                run_id=run_id,
                                alpha=float(a), beta=float(b), gamma=float(g),
                                tau_days=float(tau), top_k=int(tk),
                                min_cluster_size=int(mcs), min_samples=int(ms),
                                time_window_days=int(args.time_window_days),
                                require_dates_for_edges=bool(args.require_dates_for_edges),
                                use_soft_gate=bool(args.use_soft_gate),
                                time_gate_eps=float(args.time_gate_eps),
                                split_by_time=bool(args.split_by_time),
                                split_gap_days=int(split_gap),
                                split_min_segment_size=int(args.split_min_segment_size),
                                silhouette_cosine=float(sil),
                                noise_fraction=float(st["noise_fraction"]),
                                n_clusters=float(st["n_clusters"]),
                                cluster_size_max=float(st["cluster_size_max"]),
                                cluster_size_median=float(st["cluster_size_median"]),
                                median_span_days=float(sp["median_span_days"]),
                                p90_span_days=float(sp["p90_span_days"]),
                                max_span_days=float(sp["max_span_days"]),
                            )
                            all_rows.append(row)

                            ok = True
                            if st["n_clusters"] < args.min_clusters:
                                ok = False
                            if st["cluster_size_max"] > args.max_cluster_size:
                                ok = False

                            spans_nan = not (sp["median_span_days"] == sp["median_span_days"] and sp["p90_span_days"] == sp["p90_span_days"])
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

    pd.DataFrame(all_rows).to_csv(os.path.join(args.outdir, "grid_results.csv"), index=False, encoding="utf-8")

    if not filtered:
        print("[WARN] No runs passed constraints. Try enabling --split_by_time (recommended) or relax spans.")
        pd.DataFrame([]).to_csv(os.path.join(args.outdir, "grid_results_filtered.csv"), index=False, encoding="utf-8")
        return

    df_f = pd.DataFrame(filtered).sort_values("silhouette_cosine", ascending=False).reset_index(drop=True)
    df_f.to_csv(os.path.join(args.outdir, "grid_results_filtered.csv"), index=False, encoding="utf-8")

    # Stability selection among top-K
    K = min(int(args.stability_topk), len(df_f))
    best = None
    best_key = -1e18
    best_stab = None

    for cand in df_f.head(K).to_dict("records"):
        sim_params = dict(
            alpha=float(cand["alpha"]), beta=float(cand["beta"]), gamma=float(cand["gamma"]),
            tau_days=float(cand["tau_days"]), top_k=int(cand["top_k"]),
            time_window_days=int(cand["time_window_days"]),
            require_dates_for_edges=bool(cand["require_dates_for_edges"]),
            use_soft_gate=bool(cand["use_soft_gate"]),
            time_gate_eps=float(cand["time_gate_eps"]),
        )
        clust_params = dict(min_cluster_size=int(cand["min_cluster_size"]), min_samples=int(cand["min_samples"]))

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

        key = float(cand["silhouette_cosine"]) + 0.02 * float(stab["mean_ari"])
        print(f"Run {int(cand['run_id'])} ARI {stab['mean_ari']} ± {stab['std_ari']}")

        if key > best_key:
            best_key = key
            best = cand
            best_stab = stab

    payload = {"best": best, "stability_mean": best_stab["mean_ari"], "stability_sd": best_stab["std_ari"]}
    with open(os.path.join(args.outdir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nSelected best:", best)
    print("Stability mean±sd:", (best_stab["mean_ari"], best_stab["std_ari"]))


if __name__ == "__main__":
    main()
