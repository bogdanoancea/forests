#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_all_figures.py — Generate all figures for the illegal logging paper.
==========================================================================

Generates 11 figures from pipeline outputs + recomputation where needed.

Figures from CSV data alone (no recomputation):
  1. no_articles_per_year.png
  2. char_length_distribution.png
  3. event_size_ccdf.png
  4. event_span_distribution.png
  5. event_timeline_per_year.png
  6. bursts_articles_vs_events_per_month.png
  7. fig_noise_distribution.png
  8. fig_cluster_size_ccdf.png  (from event_id_raw in events.csv)

Figures requiring similarity matrix recomputation:
  9. fig_degree_distribution.png
 10. fig_spectrum_laplacian.png

Figures requiring bootstrap recomputation:
 11. fig_stability_distribution.png

Usage
-----
  python make_all_figures.py \
      --input   proteste.xlsx \
      --events  events.csv \
      --centroids event_centroids.csv \
      --params  best_params.json \
      --outdir  figures/ \
      --st_model paraphrase-multilingual-MiniLM-L12-v2 \
      --stability_boot 25 \
      --stability_frac 0.8 \
      --stability_seed 123

If --skip_recompute is passed, only the CSV-based figures (1–8) are generated.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Utilities ────────────────────────────────────────────────────────────

def ensure_datetime(s):
    """Robust datetime parsing."""
    return pd.to_datetime(s, errors="coerce", utc=False)


# ── Figure 1: Articles per year ──────────────────────────────────────────

def plot_articles_per_year(events_df, outpath):
    """Bar chart of number of articles per calendar year."""
    dt = ensure_datetime(events_df.get("date_iso"))
    df = pd.DataFrame({"dt": dt}).dropna(subset=["dt"])
    if df.empty:
        print("[WARN] No dated articles for articles-per-year plot.")
        return

    df["year"] = df["dt"].dt.year
    counts = df.groupby("year").size().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(float), counts.values, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Number of Articles per Year")
    ax.set_xticks(counts.index.astype(float))
    ax.set_xticklabels([f"{int(y)}" for y in counts.index], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure 2: Character length distribution ──────────────────────────────

def plot_char_length_distribution(events_df, outpath, bins=30):
    """Histogram of character lengths (title + text concatenated)."""
    texts = events_df["title"].fillna("").astype(str) + " " + events_df["text"].fillna("").astype(str)
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


# ── Figure 3: Word count distribution (bonus, useful for paper) ──────────

def plot_word_count_distribution(events_df, outpath, bins=30):
    """Histogram of word counts."""
    texts = events_df["title"].fillna("").astype(str) + " " + events_df["text"].fillna("").astype(str)
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


# ── Figure 4: Event size CCDF ────────────────────────────────────────────

def plot_event_size_ccdf(cent, outpath):
    """Complementary CDF of event sizes (post-splitting)."""
    sizes = cent.loc[cent["event_id"] != -1, "n_articles"].dropna().astype(int).values
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


# ── Figure 5: Event span distribution ────────────────────────────────────

def plot_event_span_distribution(cent, outpath, bins=12):
    """Histogram of event temporal spans in days."""
    if "span_days" not in cent.columns:
        d0 = ensure_datetime(cent.get("date_min"))
        d1 = ensure_datetime(cent.get("date_max"))
        cent = cent.copy()
        cent["span_days"] = (d1 - d0).dt.days

    spans = cent.loc[cent["event_id"] != -1, "span_days"].dropna().astype(float).values
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


# ── Figure 6: Event timeline (events per year) ──────────────────────────

def plot_event_timeline_per_year(cent, outpath):
    """Line chart of number of reconstructed events per year."""
    d0 = ensure_datetime(cent.get("date_min"))
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


# ── Figure 7: Articles vs events per month (burst plot) ──────────────────

def plot_bursts_articles_vs_events(events_df, cent, outpath):
    """Dual time series: articles/month and events/month."""
    # Articles per month
    dt = ensure_datetime(events_df.get("date_iso"))
    a = pd.DataFrame({"dt": dt}).dropna(subset=["dt"])
    if a.empty:
        print("[WARN] No dated articles for burst plot.")
        return
    a["ym"] = a["dt"].dt.to_period("M").dt.to_timestamp()
    articles_pm = a.groupby("ym").size().sort_index()

    # Events per month (using event start date)
    d0 = ensure_datetime(cent.get("date_min"))
    e = cent.copy()
    e["date_min_dt"] = d0
    e = e[(e["event_id"] != -1) & e["date_min_dt"].notna()]
    if e.empty:
        print("[WARN] No dated events for burst plot.")
        return
    e["ym"] = e["date_min_dt"].dt.to_period("M").dt.to_timestamp()
    events_pm = e.groupby("ym").size().sort_index()

    # Align
    idx = articles_pm.index.union(events_pm.index)
    articles_pm = articles_pm.reindex(idx, fill_value=0)
    events_pm = events_pm.reindex(idx, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, articles_pm.values, marker="o", markersize=3, label="Articles/month")
    ax.plot(idx, events_pm.values, marker="o", markersize=3, label="Events/month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_title("Bursts: Articles vs Reconstructed Events (per month)")
    ax.legend()
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)


# ── Figure 8: Noise vs clustered proportion ──────────────────────────────

def plot_noise_distribution(events_df, outpath):
    """Bar chart showing proportion of noise vs clustered articles."""
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


# ── Figure 9: Cluster size CCDF (pre-segmentation) ──────────────────────

def plot_cluster_size_ccdf_raw(events_df, outpath):
    """CCDF of cluster sizes from raw HDBSCAN (before temporal splitting)."""
    raw_labels = events_df["event_id_raw"].values
    nonnoise = raw_labels[raw_labels != -1]
    if len(nonnoise) == 0:
        print("[WARN] No raw clusters for pre-split CCDF.")
        return

    unique, counts = np.unique(nonnoise, return_counts=True)
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


# =====================================================================
# Figures requiring similarity matrix recomputation (9, 10, 11)
# =====================================================================

def recompute_similarity_matrix(input_path, params):
    """
    Recompute the similarity matrix S for the selected configuration.
    Returns S (n×n numpy array), embeddings, entities, dates.
    """
    from step1_grid_search_eventlike_gate_v7 import (
        build_doc, heuristic_entities, parse_date_iso,
        sim_matrix_timefiltered_topk,
    )
    from sentence_transformers import SentenceTransformer

    df = pd.read_excel(input_path)
    docs = [
        build_doc(str(r.get("title", "") or ""), str(r.get("text", "") or ""))
        for _, r in df.iterrows()
    ]
    dates = [parse_date_iso(x) for x in df.get("date_iso", [None] * len(df))]
    ents = [heuristic_entities(d) for d in docs]

    st_model = params.get("_st_model", "paraphrase-multilingual-MiniLM-L12-v2")
    model = SentenceTransformer(st_model)
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    sim_kwargs = dict(
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        gamma=float(params["gamma"]),
        tau_days=float(params["tau_days"]),
        top_k=int(params["top_k"]),
        time_window_days=int(params.get("time_window_days", 7)),
        require_dates_for_edges=bool(params.get("require_dates_for_edges", True)),
        use_soft_gate=bool(params.get("use_soft_gate", True)),
        time_gate_eps=float(params.get("time_gate_eps", 0.01)),
    )

    S = sim_matrix_timefiltered_topk(emb, ents, dates, **sim_kwargs)
    return S, emb, ents, dates


# ── Figure 10: Degree distribution ───────────────────────────────────────

def plot_degree_distribution(S, outpath, bins=20):
    """Histogram of weighted degree of the similarity graph."""
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


# ── Figure 11: Laplacian spectrum ────────────────────────────────────────

def plot_spectrum_laplacian(S, outpath, n_eigenvalues=60):
    """Spectrum of the normalized graph Laplacian (first n eigenvalues)."""
    A = S.copy()
    np.fill_diagonal(A, 0.0)

    d = A.sum(axis=1)
    # Avoid division by zero for isolated nodes
    d_inv_sqrt = np.zeros_like(d)
    nonzero = d > 1e-12
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)

    n = A.shape[0]
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Compute eigenvalues (symmetric → real)
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)

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

    # Report spectral diagnostics
    near_zero = np.sum(eigvals < 1e-6)
    first_nonzero = eigvals[eigvals > 1e-6][0] if np.any(eigvals > 1e-6) else float("nan")
    print(f"  [INFO] Near-zero eigenvalues: {near_zero}")
    print(f"  [INFO] First non-zero eigenvalue (spectral gap): {first_nonzero:.4f}")


# ── Figure 12: Bootstrap stability distribution ──────────────────────────

def plot_stability_distribution(
    emb, ents, dates, params, outpath,
    n_boot=25, frac=0.8, seed=123, bins=15,
):
    """
    Recompute bootstrap ARI values and plot their distribution.
    """
    from step1_grid_search_eventlike_gate_v7 import (
        sim_matrix_timefiltered_topk, cluster_hdbscan, split_labels_by_time,
    )
    from sklearn.metrics import adjusted_rand_score

    sim_kwargs = dict(
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        gamma=float(params["gamma"]),
        tau_days=float(params["tau_days"]),
        top_k=int(params["top_k"]),
        time_window_days=int(params.get("time_window_days", 7)),
        require_dates_for_edges=bool(params.get("require_dates_for_edges", True)),
        use_soft_gate=bool(params.get("use_soft_gate", True)),
        time_gate_eps=float(params.get("time_gate_eps", 0.01)),
    )
    mcs = int(params["min_cluster_size"])
    ms = int(params["min_samples"])
    do_split = bool(params.get("split_by_time", True))
    split_gap = int(params.get("split_gap_days", params.get("time_window_days", 7)))
    split_min = int(params.get("split_min_segment_size", 2))

    n = len(ents)

    # Full-data labels (baseline)
    S_full = sim_matrix_timefiltered_topk(emb, ents, dates, **sim_kwargs)
    base = cluster_hdbscan(S_full, mcs, ms)
    if do_split:
        base = split_labels_by_time(base, dates, split_gap_days=split_gap, min_segment_size=split_min)

    rng = np.random.default_rng(int(seed))
    aris = []

    for b in range(int(n_boot)):
        idx = rng.choice(n, int(frac * n), replace=False)
        idx.sort()

        emb_s = emb[idx]
        ents_s = [ents[i] for i in idx]
        dates_s = [dates[i] for i in idx]

        sim_kwargs_s = dict(sim_kwargs)
        sim_kwargs_s["top_k"] = min(int(sim_kwargs_s["top_k"]), len(idx))

        S_s = sim_matrix_timefiltered_topk(emb_s, ents_s, dates_s, **sim_kwargs_s)
        lab_s = cluster_hdbscan(S_s, mcs, ms)
        if do_split:
            lab_s = split_labels_by_time(lab_s, dates_s, split_gap_days=split_gap, min_segment_size=split_min)

        ari = adjusted_rand_score(base[idx], lab_s)
        aris.append(ari)
        print(f"  [BOOT {b+1:3d}/{n_boot}] ARI = {ari:.3f}")

    aris = np.array(aris)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(aris, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap Stability Distribution")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", outpath)
    print(f"  [INFO] ARI mean = {aris.mean():.3f} ± {aris.std():.3f}")


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate all figures for the illegal logging event reconstruction paper."
    )
    ap.add_argument("--input", required=True,
                    help="Input XLSX (proteste.xlsx)")
    ap.add_argument("--events", required=True,
                    help="Path to events.csv")
    ap.add_argument("--centroids", required=True,
                    help="Path to event_centroids.csv")
    ap.add_argument("--params", required=True,
                    help="Path to best_params.json")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for figures")
    ap.add_argument("--st_model", default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="SentenceTransformer model name")
    ap.add_argument("--stability_boot", type=int, default=25,
                    help="Number of bootstrap replicates for stability figure")
    ap.add_argument("--stability_frac", type=float, default=0.8,
                    help="Subsampling fraction for bootstrap")
    ap.add_argument("--stability_seed", type=int, default=123,
                    help="Random seed for bootstrap")
    ap.add_argument("--skip_recompute", action="store_true",
                    help="Skip figures that require similarity matrix recomputation "
                         "(degree distribution, spectrum, stability)")
    ap.add_argument("--bins_char", type=int, default=30,
                    help="Bins for character length histogram")
    ap.add_argument("--bins_span", type=int, default=12,
                    help="Bins for event span histogram")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    events_df = pd.read_csv(args.events)
    cent = pd.read_csv(args.centroids)

    with open(args.params, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload["best"]
    params["_st_model"] = args.st_model

    print("=" * 60)
    print("PHASE 1: Figures from CSV data (no recomputation)")
    print("=" * 60)

    # 1. Articles per year
    plot_articles_per_year(
        events_df,
        os.path.join(args.outdir, "no_articles_per_year.png"),
    )

    # 2. Character length distribution
    plot_char_length_distribution(
        events_df,
        os.path.join(args.outdir, "char_length_distribution.png"),
        bins=args.bins_char,
    )

    # 3. Word count distribution (bonus)
    plot_word_count_distribution(
        events_df,
        os.path.join(args.outdir, "word_count_distribution.png"),
        bins=args.bins_char,
    )

    # 4. Event size CCDF
    plot_event_size_ccdf(
        cent,
        os.path.join(args.outdir, "event_size_ccdf.png"),
    )

    # 5. Event span distribution
    plot_event_span_distribution(
        cent,
        os.path.join(args.outdir, "event_span_distribution.png"),
        bins=args.bins_span,
    )

    # 6. Event timeline per year
    plot_event_timeline_per_year(
        cent,
        os.path.join(args.outdir, "event_timeline_per_year.png"),
    )

    # 7. Bursts: articles vs events per month
    plot_bursts_articles_vs_events(
        events_df, cent,
        os.path.join(args.outdir, "bursts_articles_vs_events_per_month.png"),
    )

    # 8. Noise vs clustered
    plot_noise_distribution(
        events_df,
        os.path.join(args.outdir, "fig_noise_distribution.png"),
    )

    # 9. Cluster size CCDF (pre-segmentation, from event_id_raw)
    plot_cluster_size_ccdf_raw(
        events_df,
        os.path.join(args.outdir, "fig_cluster_size_ccdf.png"),
    )

    if args.skip_recompute:
        print("\n[SKIP] Recomputation figures skipped (--skip_recompute).")
        print("       Skipped: fig_degree_distribution, fig_spectrum_laplacian,")
        print("                fig_stability_distribution")
        return

    print()
    print("=" * 60)
    print("PHASE 2: Recomputing similarity matrix")
    print("=" * 60)

    S, emb, ents, dates = recompute_similarity_matrix(args.input, params)

    # 10. Degree distribution
    plot_degree_distribution(
        S,
        os.path.join(args.outdir, "fig_degree_distribution.png"),
    )

    # 11. Laplacian spectrum
    plot_spectrum_laplacian(
        S,
        os.path.join(args.outdir, "fig_spectrum_laplacian.png"),
    )

    print()
    print("=" * 60)
    print("PHASE 3: Bootstrap stability recomputation")
    print("=" * 60)

    # 12. Stability distribution
    plot_stability_distribution(
        emb, ents, dates, params,
        os.path.join(args.outdir, "fig_stability_distribution.png"),
        n_boot=args.stability_boot,
        frac=args.stability_frac,
        seed=args.stability_seed,
    )

    print()
    print("=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()


#python make_all_figures.py \
#    --input proteste.xlsx \
#    --events events.csv \
#    --centroids event_centroids.csv \
#    --params best_params.json \
#    --outdir figures/ \
#    --stability_boot 25