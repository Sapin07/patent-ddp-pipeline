"""
Pipeline de Caminho de Desenvolvimento Disruptivo (DDP)
Wang et al. (2024), Journal of Informetrics 18, 101493

Executa o pipeline em dois modos:
  1. Dados reais do BigQuery (forward-citations only — limitação documentada)
  2. Dados sintéticos (demonstração completa do método DDP)

Figuras salvas em: relatorio/img/
"""

import sys, os, json, pathlib, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sem display (salva arquivos)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict, Counter
from typing import Optional, Dict, List, Tuple, Set

WORKSPACE = pathlib.Path("/home/vcunha/pen/SI/ic-esteban")
IMG_DIR   = WORKSPACE / "relatorio" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = str(WORKSPACE / "bq-results-20260518-131545-1779110232837.json")
NET_PATH  = str(WORKSPACE / "bq-results-20260518-131903-1779110375248.json")

# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1 — CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────────────────────────────────────

def load_bigquery_json(json_metadata_path: str,
                       json_network_path: str) -> Tuple[pd.DataFrame, dict]:
    print(f"[BigQuery] Carregando metadados de {pathlib.Path(json_metadata_path).name}...")
    df_meta = pd.read_json(json_metadata_path, lines=True, dtype={"priority_date": str})
    df_meta["year"] = pd.to_datetime(
        df_meta["priority_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.year
    year_map = dict(zip(df_meta["publication_number"], df_meta["year"]))

    print(f"[BigQuery] Carregando rede de {pathlib.Path(json_network_path).name}...")
    df_net = pd.read_json(json_network_path, lines=True, dtype={"citing_date": str})
    df_net["citing_year"] = pd.to_datetime(
        df_net["citing_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.year
    new_years = dict(zip(df_net["citing_patent"], df_net["citing_year"]))
    year_map = {**new_years, **year_map}

    df_pairs = df_net[["citing_patent", "cited_patent"]].copy()
    df_pairs.columns = ["citing", "cited"]
    df_pairs["year_citing"] = df_pairs["citing"].map(year_map)
    df_pairs["year_cited"]  = df_pairs["cited"].map(year_map)

    n_before = len(df_pairs)
    df_pairs = df_pairs.dropna(subset=["year_citing", "year_cited"])

    focal_set = set(df_meta["publication_number"])
    backward = df_pairs[df_pairs["citing"].isin(focal_set)]
    forward  = df_pairs[df_pairs["cited"].isin(focal_set)]
    print(f"  {len(df_pairs):,} pares válidos | "
          f"Backward: {len(backward)} | Forward: {len(forward)}")
    if len(backward) == 0:
        print("  ⚠ Sem backward citations — D' requer re-query no BigQuery.")
    return df_pairs, year_map


def create_synthetic_example() -> Tuple[pd.DataFrame, dict]:
    patents = {
        "P1995": 1995, "P1996a": 1996, "P1996b": 1996,
        "P1998": 1998, "P1999": 1999,
        "P2001": 2001, "P2002a": 2002, "P2002b": 2002,
        "P2004": 2004, "P2005a": 2005, "P2005b": 2005,
        "P2007": 2007, "P2008a": 2008, "P2008b": 2008, "P2008c": 2008,
        "P2010": 2010, "P2011a": 2011, "P2011b": 2011,
        "P2012": 2012, "P2013a": 2013, "P2013b": 2013, "P2013c": 2013,
        "P2015a": 2015, "P2015b": 2015, "P2015c": 2015,
        "P2016": 2016, "P2017a": 2017, "P2017b": 2017,
        "P2018": 2018, "P2019": 2019,
    }
    citation_pairs = [
        ("P1996a","P1995"), ("P1998","P1996a"), ("P1998","P1996b"),
        ("P1999","P1998"), ("P2001","P1999"), ("P2002a","P2001"),
        ("P2004","P2002a"), ("P2005a","P2004"), ("P2007","P2005a"),
        ("P2008a","P2007"), ("P2010","P2008a"), ("P2012","P2010"),
        ("P2016","P2012"), ("P2018","P2016"), ("P2019","P2018"),
        ("P1996b","P1995"),
        ("P2002b","P1999"), ("P2002b","P1998"),
        ("P2005b","P2002a"), ("P2005b","P2002b"),
        ("P2008b","P2005a"), ("P2008b","P2005b"),
        ("P2008c","P2004"), ("P2008c","P2005a"),
        ("P2011a","P2008a"), ("P2011a","P2007"),
        ("P2011b","P2008b"),
        ("P2013a","P2011a"), ("P2013a","P2010"),
        ("P2013b","P2011b"),
        ("P2013c","P2010"),
        ("P2015a","P2013a"),
        ("P2015b","P2013b"), ("P2015b","P2013a"),
        ("P2015c","P2012"), ("P2015c","P2010"),
        ("P2017a","P2015a"), ("P2017a","P2016"),
        ("P2017b","P2015b"),
    ]
    df = pd.DataFrame(citation_pairs, columns=["citing","cited"])
    df["year_citing"] = df["citing"].map(patents)
    df["year_cited"]  = df["cited"].map(patents)
    print(f"[Sintético] {len(df)} pares | {len(patents)} patentes")
    return df, patents


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — CONSTRUÇÃO DA REDE 2-012U
# ─────────────────────────────────────────────────────────────────────────────

def build_citation_network(df_pairs: pd.DataFrame, year_map: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in df_pairs.iterrows():
        citing, cited = row["citing"], row["cited"]
        yc = year_map.get(citing, 0)
        yd = year_map.get(cited, 0)
        if yc <= yd:
            continue
        G.add_node(citing, year=yc)
        G.add_node(cited,  year=yd)
        G.add_edge(citing, cited)
    print(f"[2-012U] Nós: {G.number_of_nodes()} | "
          f"Arestas: {G.number_of_edges()} | "
          f"DAG: {nx.is_directed_acyclic_graph(G)}")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3 — TRIPLETS E DISRUPÇÃO D'
# ─────────────────────────────────────────────────────────────────────────────

def classify_triplets_and_disruption(G: nx.DiGraph, year_map: dict,
                                     focal_patents=None) -> pd.DataFrame:
    if focal_patents is None:
        focal_patents = [n for n in G.nodes()
                         if G.in_degree(n) > 0 and G.out_degree(n) > 0]
    results = []
    for focal in focal_patents:
        focal_year = year_map.get(focal, 0)
        backward: Set = set(G.successors(focal))
        if not backward:
            continue
        forward: Set = {n for n in G.predecessors(focal)
                        if year_map.get(n, 0) > focal_year}
        if not forward:
            continue
        ni = nj = 0
        for fwd in forward:
            fwd_cites = set(G.successors(fwd))
            if fwd_cites & backward:
                nj += 1
            else:
                ni += 1
        all_back_citers: Set = set()
        for back in backward:
            all_back_citers.update(
                n for n in G.predecessors(back)
                if year_map.get(n, 0) > focal_year
            )
        nk_tilde = len(all_back_citers)
        nk = nk_tilde - nj
        d_prime    = ni / (ni + nk_tilde) if (ni + nk_tilde) > 0 else np.nan
        d_original = (ni - nj) / (ni + nj + nk) if (ni + nj + nk) > 0 else np.nan
        results.append({
            "patent": focal, "year": focal_year,
            "ni": ni, "nj": nj, "nk": nk, "nk_tilde": nk_tilde,
            "n_forward": len(forward), "n_backward": len(backward),
            "d_prime": d_prime, "d_original": d_original,
        })
    df = pd.DataFrame(results).sort_values("year").reset_index(drop=True)
    valid = df["d_prime"].dropna()
    print(f"[D'] {len(df)} patentes focais | "
          f"Média: {valid.mean():.3f} | Mediana: {valid.median():.3f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 4 — ANÁLISE TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────

def analyze_annual_disruption(df: pd.DataFrame) -> pd.DataFrame:
    return (df.dropna(subset=["d_prime"])
            .groupby("year")
            .agg(count=("patent","count"),
                 mean_d=("d_prime","mean"),
                 median_d=("d_prime","median"),
                 std_d=("d_prime","std"),
                 max_d=("d_prime","max"))
            .reset_index())


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 5 — CONTRAÇÃO DA REDE
# ─────────────────────────────────────────────────────────────────────────────

def contract_citation_network(G, df_disruption, disruption_threshold=0.3,
                               min_citations=2):
    df_valid = df_disruption[
        (df_disruption["d_prime"] >= disruption_threshold) &
        (df_disruption["n_forward"] >= min_citations)
    ].copy()
    valid_patents = set(df_valid["patent"])
    G_c = G.subgraph(valid_patents).copy()
    if not nx.is_directed_acyclic_graph(G_c):
        G_c = _remove_cycles(G_c)
    print(f"[Contração θ≥{disruption_threshold}] "
          f"{G.number_of_nodes()} → {G_c.number_of_nodes()} nós | "
          f"{G.number_of_edges()} → {G_c.number_of_edges()} arestas")
    return G_c, df_valid


def _remove_cycles(G):
    G2 = G.copy()
    while not nx.is_directed_acyclic_graph(G2):
        cycles = list(nx.simple_cycles(G2))
        if not cycles:
            break
        G2.remove_edge(cycles[0][-1], cycles[0][0])
    return G2


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 6 — SPLC + CAMINHO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_splc_weights(G):
    topo = list(nx.topological_sort(G))
    paths_to   = {n: 0 for n in G.nodes()}
    paths_from = {n: 0 for n in G.nodes()}
    for n in topo:
        paths_to[n] = 1 if G.in_degree(n) == 0 else sum(paths_to[p] for p in G.predecessors(n))
    for n in reversed(topo):
        paths_from[n] = 1 if G.out_degree(n) == 0 else sum(paths_from[s] for s in G.successors(n))
    edge_w = {(u,v): paths_to[u]*paths_from[v] for u,v in G.edges()}
    return edge_w, paths_to, paths_from


def extract_main_path(G, edge_weights, year_map, disruption_df=None):
    if G.number_of_nodes() == 0:
        return []
    topo = list(nx.topological_sort(G))
    max_w = {n: 0.0 for n in G.nodes()}
    pred  = {n: None for n in G.nodes()}
    for v in topo:
        for u in G.predecessors(v):
            w = edge_weights.get((u,v), 0)
            if max_w[u] + w > max_w[v]:
                max_w[v] = max_w[u] + w
                pred[v]  = u
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0] or list(G.nodes())
    best  = max(sinks, key=lambda s: max_w[s])
    path, node = [], best
    while node is not None:
        path.append(node)
        node = pred[node]
    path = list(reversed(path))
    d_map = {} if disruption_df is None else dict(zip(disruption_df["patent"], disruption_df["d_prime"]))
    print(f"[Caminho Principal] {len(path)} nós: " +
          " → ".join(f"{n}({year_map.get(n,'?')})" for n in path))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 7 — CPM
# ─────────────────────────────────────────────────────────────────────────────

def extract_critical_path(G, edge_weights, year_map):
    if G.number_of_nodes() == 0:
        return [], nx.DiGraph()
    topo = list(nx.topological_sort(G))
    es = {n: 0 for n in G.nodes()}
    for n in topo:
        for s in G.successors(n):
            w = edge_weights.get((n,s), 1)
            es[s] = max(es[s], es[n]+w)
    end = max(es, key=es.get)
    ls = {n: es[end] for n in G.nodes()}
    for n in reversed(topo):
        for p in G.predecessors(n):
            w = edge_weights.get((p,n), 1)
            ls[p] = min(ls[p], ls[n]-w)
    crit_nodes, crit_edges = set(), []
    for u,v in G.edges():
        w = edge_weights.get((u,v), 1)
        if abs(ls[v] - es[u] - w) < 1e-9:
            crit_nodes |= {u,v}
            crit_edges.append((u,v))
    G_c = G.subgraph(crit_nodes).copy()
    crit_path = [n for n in topo if n in crit_nodes]
    print(f"[CPM] {len(crit_nodes)} nós críticos | {len(crit_edges)} arestas críticas")
    return crit_path, G_c


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def _hier_layout(G, year_map):
    years = sorted({year_map.get(n, 0) for n in G.nodes()})
    year_nodes = defaultdict(list)
    for n in G.nodes():
        year_nodes[year_map.get(n, 0)].append(n)
    pos = {}
    for yi, y in enumerate(years):
        ns = year_nodes[y]
        for xi, n in enumerate(ns):
            pos[n] = ((xi - (len(ns)-1)/2) * 1.5, -yi)
    return pos


def plot_disruption_annual(df_annual, title="Grau de Disrupção Anual (dados sintéticos)",
                           save_path=None):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(df_annual["year"], df_annual["count"], color="#4C72B0", alpha=0.7,
            label="Qtd. patentes focais")
    ax1.set_xlabel("Ano"); ax1.set_ylabel("Quantidade", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(df_annual["year"], df_annual["mean_d"], color="#DD8452",
             lw=2.5, marker="o", ms=5, label="D' médio")
    ax2.fill_between(df_annual["year"],
                     df_annual["mean_d"] - df_annual["std_d"].fillna(0),
                     df_annual["mean_d"] + df_annual["std_d"].fillna(0),
                     color="#DD8452", alpha=0.15)
    ax2.axhline(0.5, color="red", ls="--", alpha=0.5, lw=1)
    ax2.set_ylabel("D' médio", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    ax2.set_ylim(0, 1.1)
    for _, row in df_annual[df_annual["mean_d"] > 0.5].iterrows():
        ax2.annotate(f"{row['mean_d']:.2f}", xy=(row["year"], row["mean_d"]),
                     xytext=(0, 8), textcoords="offset points", ha="center",
                     fontsize=8, color="#DD8452")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc="upper left")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fig] → {save_path}")
    plt.close()


def plot_main_path(G, main_path, year_map, disruption_df=None,
                   title="Caminho Principal (SPLC)", save_path=None):
    if G.number_of_nodes() == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 12))
    pos = _hier_layout(G, year_map)
    mp_set = set(main_path)
    node_colors = ["#E8623C" if n in mp_set else "#7FB3D3" for n in G.nodes()]
    node_sizes  = [400 if n in mp_set else 150 for n in G.nodes()]
    edge_colors, edge_widths = [], []
    for u,v in G.edges():
        if u in mp_set and v in mp_set:
            edge_colors.append("#C0392B"); edge_widths.append(2.5)
        else:
            edge_colors.append("#AAAAAA"); edge_widths.append(0.8)
    nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_colors,
                     node_size=node_sizes, edge_color=edge_colors,
                     width=edge_widths, labels={n: n for n in G.nodes()},
                     font_size=6, arrows=True, arrowsize=10)
    ax.legend(handles=[
        mpatches.Patch(color="#E8623C", label="Caminho principal (SPLC)"),
        mpatches.Patch(color="#7FB3D3", label="Outros nós"),
    ], loc="upper right")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fig] → {save_path}")
    plt.close()


def plot_disruption_distribution(df_disruption, save_path=None):
    df_v = df_disruption.dropna(subset=["d_prime"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_v["d_prime"], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(0.5, color="red", ls="--", alpha=0.7)
    axes[0].set_xlabel("D' (Fórmula 3)"); axes[0].set_ylabel("Frequência")
    axes[0].set_title("Distribuição do Índice D'")
    df_b = df_disruption.dropna(subset=["d_prime","d_original"])
    axes[1].scatter(df_b["d_original"], df_b["d_prime"], alpha=0.4, s=20, color="#DD8452")
    axes[1].plot([-1,1],[-1,1], "r--", alpha=0.4)
    axes[1].set_xlabel("D (Fórmula 1 — Wu et al.)"); axes[1].set_ylabel("D' (Fórmula 3)")
    axes[1].set_title("Comparação D vs D'")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fig] → {save_path}")
    plt.close()


def plot_citation_heatmap_real(net_rows, meta_rows, save_path=None):
    """Figura extra com dados reais: distribuição temporal de citações forward."""
    from collections import Counter
    # Dados reais: citar por ano do citante e contagem
    focal = {r["publication_number"]: int(str(r["priority_date"])[:4]) for r in meta_rows}
    fwd   = [r for r in net_rows if r["cited_patent"] in focal]

    # Anos dos citantes
    citing_years = [int(str(r["citing_date"])[:4]) for r in fwd]
    cited_years  = [focal[r["cited_patent"]] for r in fwd]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Histograma anos dos citantes
    axes[0].hist(citing_years, bins=range(2004, 2027), color="#4C72B0",
                 edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Ano da patente citante (internacional)")
    axes[0].set_ylabel("Nº de citações")
    axes[0].set_title("Citações forward para patentes BR\npor ano do citante")
    axes[0].tick_params(axis="x", rotation=45)

    # Top patentes por contagem de citações
    top = Counter(r["cited_patent"] for r in fwd).most_common(10)
    labs = [p[:25] + "…" if len(p) > 25 else p for p, _ in top]
    vals = [c for _, c in top]
    axes[1].barh(labs[::-1], vals[::-1], color="#E8623C", alpha=0.85)
    axes[1].set_xlabel("Nº de citações forward")
    axes[1].set_title("Top 10 patentes BR mais citadas\n(corpus C12N + lamininas)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fig] → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_synthetic():
    print("\n" + "="*60)
    print("PIPELINE SINTÉTICO — Demonstração DDP Completo")
    print("="*60)

    df_pairs, year_map = create_synthetic_example()
    G = build_citation_network(df_pairs, year_map)
    df_dis = classify_triplets_and_disruption(G, year_map)
    df_ann = analyze_annual_disruption(df_dis)

    print("\nAnálise Anual:")
    print(df_ann.to_string(index=False))

    G_c, df_kept = contract_citation_network(G, df_dis, 0.25, 1)
    if G_c.number_of_nodes() == 0:
        G_c, df_kept = contract_citation_network(G, df_dis, 0.1, 1)

    ew, pt, pf = compute_splc_weights(G_c)
    main_path   = extract_main_path(G_c, ew, year_map, df_dis)
    crit_path, G_crit = extract_critical_path(G_c, ew, year_map)

    # Figuras
    plot_disruption_annual(
        df_ann,
        title="Grau de Disrupção Anual — Dados Sintéticos (Wang et al., 2024)",
        save_path=str(IMG_DIR / "fig_disruption_annual.pdf"),
    )
    plot_main_path(
        G_c, main_path, year_map, df_dis,
        title="Caminho Principal (SPLC) — Dados Sintéticos",
        save_path=str(IMG_DIR / "fig_main_path.pdf"),
    )
    plot_disruption_distribution(
        df_dis,
        save_path=str(IMG_DIR / "fig_distribution.pdf"),
    )

    # Resumo
    mp_df = df_dis[df_dis["patent"].isin(main_path)][
        ["patent","year","d_prime","ni","nj","nk"]
    ].sort_values("year")
    print("\nNós do caminho principal:")
    print(mp_df.to_string(index=False))

    # Excel
    with pd.ExcelWriter(str(IMG_DIR / "resultados_sinteticos.xlsx"), engine="openpyxl") as w:
        df_dis.to_excel(w, sheet_name="Disruption_Degree", index=False)
        df_ann.to_excel(w, sheet_name="Annual_Analysis",   index=False)
        mp_df.to_excel(w, sheet_name="Main_Path",          index=False)
    print("[Export] resultados_sinteticos.xlsx salvo.")
    return df_dis, df_ann, main_path, G_c


def run_pipeline_real_partial():
    print("\n" + "="*60)
    print("ANÁLISE REAL — Dados BigQuery (Forward citations BR corpus)")
    print("="*60)

    with open(META_PATH) as f:
        meta_rows = [json.loads(l) for l in f if l.strip()]
    with open(NET_PATH) as f:
        net_rows  = [json.loads(l) for l in f if l.strip()]

    focal_set = {r["publication_number"] for r in meta_rows}
    focal_meta = {r["publication_number"]: r for r in meta_rows}

    # Apenas pares forward (internacional → BR focal)
    fwd = [r for r in net_rows if r["cited_patent"] in focal_set]
    all_cited  = {r["cited_patent"]  for r in fwd}
    all_citing = {r["citing_patent"] for r in fwd}

    print(f"  Patentes focais BR: {len(focal_set):,}")
    print(f"  Pares forward:      {len(fwd):,}")
    print(f"  BR citadas (≥1):    {len(all_cited)}")
    print(f"  Citantes únicos:    {len(all_citing)}")

    # ── Polilaminina ──────────────────────────────────────────────────────────
    poly = [r for r in fwd if "BR-PI0805852" in r["cited_patent"]]
    print(f"\n  Polilaminina (BR-PI0805852): {len(poly)} citações forward")
    for r in poly:
        print(f"    {r['citing_patent']} ({str(r['citing_date'])[:4]}) → {r['cited_patent']}")

    # ── DataFrames ────────────────────────────────────────────────────────────
    df_fwd = pd.DataFrame(fwd)
    df_fwd["citing_year"] = pd.to_datetime(
        df_fwd["citing_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.year
    df_fwd["cited_year"] = df_fwd["cited_patent"].map(
        lambda p: int(str(focal_meta.get(p, {}).get("priority_date", "0"))[:4])
        if focal_meta.get(p) else None
    )
    df_fwd["country_citing"] = df_fwd["citing_patent"].str.split("-").str[0]

    df_meta = pd.DataFrame(meta_rows)
    df_meta["year"] = pd.to_datetime(
        df_meta["priority_date"].astype(str), format="%Y%m%d", errors="coerce"
    ).dt.year

    # ── Métricas por patente BR citada ────────────────────────────────────────
    citation_counts = Counter(r["cited_patent"] for r in fwd)
    df_impact = pd.DataFrame(
        list(citation_counts.items()), columns=["publication_number", "forward_citations"]
    )
    df_impact = df_impact.merge(
        df_meta[["publication_number","family_id","title","priority_date","year"]],
        on="publication_number", how="left"
    ).sort_values("forward_citations", ascending=False).reset_index(drop=True)

    # primeira e última citação por patente
    first_last = df_fwd.groupby("cited_patent")["citing_year"].agg(["min","max"])
    first_last.columns = ["first_citation_year","last_citation_year"]
    df_impact = df_impact.merge(first_last, left_on="publication_number",
                                right_index=True, how="left")
    df_impact["citation_span_years"] = (
        df_impact["last_citation_year"] - df_impact["first_citation_year"]
    ).fillna(0).astype(int)

    print(f"\n  Top 5 mais citadas:")
    for _, row in df_impact.head(5).iterrows():
        title_short = (row["title"] or "")[:60]
        print(f"    {row['publication_number']}: {row['forward_citations']} cit. | {title_short}")

    # ── Construção da rede bipartida (DAG real) ───────────────────────────────
    G_real = nx.DiGraph()
    for r in fwd:
        cy = int(str(r["citing_date"])[:4]) if r.get("citing_date") else 0
        cited_y = int(str(focal_meta.get(r["cited_patent"],{}).get("priority_date","0"))[:4])
        if cy > 0 and cited_y > 0 and cy >= cited_y:
            G_real.add_node(r["citing_patent"],  year=cy,      kind="intl")
            G_real.add_node(r["cited_patent"],   year=cited_y, kind="focal_br")
            G_real.add_edge(r["citing_patent"], r["cited_patent"])

    print(f"\n  Rede real: {G_real.number_of_nodes()} nós | "
          f"{G_real.number_of_edges()} arestas | "
          f"DAG: {nx.is_directed_acyclic_graph(G_real)}")

    # ── Estatísticas de rede ──────────────────────────────────────────────────
    in_deg  = dict(G_real.in_degree())   # BR: recebe; intl: recebe = 0
    out_deg = dict(G_real.out_degree())  # intl: emite; BR: emite = 0
    print(f"  In-degree  (BR recebe): max={max(in_deg.values())}  "
          f"mean={np.mean(list(in_deg.values())):.2f}")
    print(f"  Out-degree (intl emite): max={max(out_deg.values())}  "
          f"mean={np.mean(list(out_deg.values())):.2f}")
    # Componentes fracamente conectadas
    wcc = list(nx.weakly_connected_components(G_real))
    print(f"  Componentes WCC: {len(wcc)} | "
          f"maior: {max(len(c) for c in wcc)} nós")

    # ── FIGURA 1: timeline + top 10 (existente, aprimorada) ──────────────────
    plot_citation_heatmap_real(
        net_rows, meta_rows,
        save_path=str(IMG_DIR / "fig_real_forward_citations.pdf"),
    )

    # ── FIGURA 2: rede bipartida (top 20 BR mais citadas) ────────────────────
    _plot_real_network(G_real, df_impact, df_fwd,
                       save_path=str(IMG_DIR / "fig_real_network.pdf"))

    # ── FIGURA 3: evolução temporal por família principal ─────────────────────
    _plot_real_temporal_families(df_fwd, df_impact,
                                  save_path=str(IMG_DIR / "fig_real_temporal.pdf"))

    # ── Exportação Excel ──────────────────────────────────────────────────────
    country_dist = df_fwd["country_citing"].value_counts().reset_index()
    country_dist.columns = ["country", "citations"]
    annual_fwd = df_fwd.groupby("citing_year").size().reset_index(name="citations")

    with pd.ExcelWriter(str(IMG_DIR / "resultados_reais.xlsx"),
                        engine="openpyxl") as w:
        df_impact.to_excel(w, sheet_name="Top_Citadas_BR", index=False)
        df_fwd.to_excel(w, sheet_name="Pares_Forward", index=False)
        country_dist.to_excel(w, sheet_name="Paises_Citantes", index=False)
        annual_fwd.to_excel(w, sheet_name="Citacoes_por_Ano", index=False)

    # CSV de compatibilidade
    df_impact.to_csv(str(IMG_DIR / "top_cited_br_patents.csv"), index=False)
    print(f"[Export] resultados_reais.xlsx + top_cited_br_patents.csv salvos.")
    return df_impact, G_real


def _plot_real_network(G_real, df_impact, df_fwd, save_path=None, top_n=20):
    """Rede bipartida: top N BR mais citadas + seus citantes internacionais."""
    top_br = set(df_impact.head(top_n)["publication_number"])
    # Subgrafo dos top_n
    nodes_sub = top_br | {u for u,v in G_real.edges() if v in top_br}
    G_sub = G_real.subgraph(nodes_sub).copy()

    country_citing = df_fwd.set_index("citing_patent")["country_citing"].to_dict()
    country_colors = {"US": "#4C72B0", "CN": "#DD8452", "WO": "#55A868",
                      "EP": "#C44E52"}

    br_nodes   = [n for n in G_sub.nodes() if G_sub.nodes[n].get("kind") == "focal_br"]
    intl_nodes = [n for n in G_sub.nodes() if G_sub.nodes[n].get("kind") == "intl"]

    # Layout bipartido manual
    pos = {}
    br_sorted = sorted(br_nodes, key=lambda n: G_sub.in_degree(n), reverse=True)
    for i, n in enumerate(br_sorted):
        pos[n] = (0, -i * 1.2)
    for j, n in enumerate(sorted(intl_nodes,
                                  key=lambda n: G_sub.out_degree(n), reverse=True)):
        pos[n] = (3, -j * (len(br_sorted) * 1.2 / max(len(intl_nodes), 1)))

    fig, ax = plt.subplots(figsize=(12, max(8, len(br_sorted) * 0.55)))

    nc_intl = [country_colors.get(country_citing.get(n, ""), "#AAAAAA")
               for n in intl_nodes]
    nx.draw_networkx_nodes(G_sub, pos, nodelist=br_nodes, ax=ax,
                           node_color="#E8623C", node_size=350, alpha=0.9)
    nx.draw_networkx_nodes(G_sub, pos, nodelist=intl_nodes, ax=ax,
                           node_color=nc_intl, node_size=80, alpha=0.6)

    # Highlight polilaminina
    poly_nodes = [n for n in br_nodes if "BR-PI0805852" in n]
    if poly_nodes:
        nx.draw_networkx_nodes(G_sub, pos, nodelist=poly_nodes, ax=ax,
                               node_color="#FFD700", node_size=500,
                               edgecolors="black", linewidths=1.5)

    nx.draw_networkx_edges(G_sub, pos, ax=ax, alpha=0.25,
                           edge_color="#888888", arrows=True,
                           arrowsize=6, width=0.6)

    # Labels apenas para BR nodes (abreviado)
    br_labels = {n: n[:22] + "…" if len(n) > 22 else n for n in br_nodes}
    nx.draw_networkx_labels(G_sub, pos, labels=br_labels, ax=ax,
                            font_size=6.5, font_color="#222222")

    # Legenda
    legend_h = [
        mpatches.Patch(color="#E8623C", label=f"BR focal (top {top_n} citadas)"),
        mpatches.Patch(color="#FFD700", label="Polilaminina UFRJ"),
        mpatches.Patch(color="#4C72B0", label="Citante US"),
        mpatches.Patch(color="#DD8452", label="Citante CN"),
        mpatches.Patch(color="#55A868", label="Citante WO/PCT"),
        mpatches.Patch(color="#AAAAAA", label="Citante outros"),
    ]
    ax.legend(handles=legend_h, loc="upper right", fontsize=8)
    ax.set_title(f"Rede de Citações Forward — Top {top_n} Patentes BR (dados reais BigQuery)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fig] → {save_path}")
    plt.close()


def _plot_real_temporal_families(df_fwd, df_impact, save_path=None, top_n=8):
    """Evolução temporal de citações para as top N patentes BR."""
    top_pats = list(df_impact.head(top_n)["publication_number"])
    df_sub = df_fwd[df_fwd["cited_patent"].isin(top_pats)].copy()
    df_sub = df_sub.dropna(subset=["citing_year"])

    annual = (df_sub.groupby(["cited_patent","citing_year"])
              .size().reset_index(name="count"))

    title_map = dict(zip(df_impact["publication_number"], df_impact["title"]))

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10.colors
    for i, pat in enumerate(top_pats):
        sub = annual[annual["cited_patent"] == pat].sort_values("citing_year")
        if sub.empty:
            continue
        short_title = (title_map.get(pat) or pat)[:40]
        ax.plot(sub["citing_year"], sub["count"], marker="o", ms=4,
                color=colors[i % 10], lw=1.8, label=short_title)

    # Highlight polilaminina com linha especial
    poly_rows = df_fwd[df_fwd["cited_patent"].str.contains("BR-PI0805852", na=False)]
    if not poly_rows.empty:
        poly_ann = (poly_rows.groupby("citing_year").size()
                    .reset_index(name="count").sort_values("citing_year"))
        ax.plot(poly_ann["citing_year"], poly_ann["count"],
                marker="*", ms=12, color="gold", lw=2.5, linestyle="--",
                label="Polilaminina UFRJ", zorder=5,
                markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Ano da patente citante")
    ax.set_ylabel("Nº de citações forward")
    ax.set_title(f"Evolução Temporal de Citações — Top {top_n} Patentes BR (dados reais)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Fig] → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Dados reais — análise da rede forward (BigQuery)
    df_impact, G_real = run_pipeline_real_partial()

    # 2. Pipeline DDP completo com dados sintéticos (demonstração do método)
    df_dis, df_ann, main_path, G_c = run_pipeline_synthetic()

    print("\n" + "="*60)
    print(f"[CONCLUÍDO] Figuras salvas em: {IMG_DIR}")
    print("="*60)
