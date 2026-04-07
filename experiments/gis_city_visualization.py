"""
GIS City Road Network Visualization Script
===========================================
Professional GIS visualizations for HHRoad (HRNR_Hyperbolic) model paper.
Generates publication-quality figures for 5 cities: Beijing, Chengdu, Xi'an, Porto, San Francisco.

Usage:
    python gis_city_visualization.py

Output:
    /home/user/hperroad/experiments/figures/gis/
        - city_overview.pdf
        - roadnet_{city}.pdf  (5 files)
        - dataset_stats.pdf
        - topology_analysis.pdf
        - spatial_stats.pdf
"""

import os
import sys
import time
import warnings
import traceback

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

# osmnx configuration
import osmnx as ox
ox.settings.use_cache = True
ox.settings.cache_folder = "/tmp/osmnx_cache"
ox.settings.timeout = 180
ox.settings.log_console = False

# ── Constants ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = "/home/user/hperroad/experiments/figures/gis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# City definitions: (label, code, bbox_lon_min, lat_min, lon_max, lat_max)
# osmnx v2.x bbox = (left, bottom, right, top) = (lon_min, lat_min, lon_max, lat_max)
CITIES = [
    {
        "name": "Beijing",
        "code": "bj",
        "bbox": (116.2, 39.8, 116.5, 40.0),   # (left, bottom, right, top)
        "center": (116.35, 39.9),
    },
    {
        "name": "Chengdu",
        "code": "cd",
        "bbox": (103.9, 30.5, 104.1, 30.7),
        "center": (104.0, 30.6),
    },
    {
        "name": "Xi'an",
        "code": "xa",
        "bbox": (108.8, 34.2, 109.05, 34.4),
        "center": (108.925, 34.3),
    },
    {
        "name": "Porto",
        "code": "prt",
        "bbox": (-8.7, 41.1, -8.55, 41.2),
        "center": (-8.625, 41.15),
    },
    {
        "name": "San Francisco",
        "code": "sf",
        "bbox": (-122.5, 37.7, -122.35, 37.82),
        "center": (-122.425, 37.76),
    },
]

# Color scheme
ROAD_COLORS = {
    "motorway": "#FF4136",
    "trunk": "#FF851B",
    "primary": "#FFDC00",
    "secondary": "#2ECC40",
    "tertiary": "#0074D9",
    "residential": "#AAAAAA",
    "living_street": "#888888",
    "service": "#666666",
    "unclassified": "#555555",
    "other": "#444444",
}
ROAD_WIDTHS = {
    "motorway": 2.0,
    "trunk": 1.8,
    "primary": 1.5,
    "secondary": 1.2,
    "tertiary": 1.0,
    "residential": 0.7,
    "living_street": 0.6,
    "service": 0.5,
    "unclassified": 0.5,
    "other": 0.4,
}
BG_COLOR = "#0d1117"
EDGE_COLOR_DEFAULT = "#4a9eff"
TEXT_COLOR = "#e6edf3"
ACCENT_COLOR = "#58a6ff"

# ── Helpers ────────────────────────────────────────────────────────────────────

def download_city_graph(city, retries=3, wait=10):
    """Download road network for a city using osmnx v2.x API."""
    lon_min, lat_min, lon_max, lat_max = city["bbox"]
    # osmnx v2.x: bbox = (left, bottom, right, top)
    bbox = (lon_min, lat_min, lon_max, lat_max)
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading {city['name']} (attempt {attempt}/{retries})...")
            G = ox.graph_from_bbox(bbox, network_type="drive", retain_all=False)
            print(f"  OK: {len(G.nodes)} nodes, {len(G.edges)} edges")
            return G
        except Exception as exc:
            print(f"  Attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(wait)
    print(f"  ERROR: Could not download {city['name']} after {retries} attempts.")
    return None


def get_edge_highway_type(edge_data):
    """Extract standardised highway type from edge attributes dict."""
    hw = edge_data.get("highway", "other")
    if isinstance(hw, list):
        hw = hw[0]
    if not isinstance(hw, str):
        hw = "other"
    for key in ROAD_COLORS:
        if hw.startswith(key):
            return key
    return "other"


def edges_to_segments(gdf_edges):
    """Return list of (x_arr, y_arr, hw_type) for each edge geometry."""
    segments = []
    for _, row in gdf_edges.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        hw = row.get("highway", "other")
        if isinstance(hw, list):
            hw = hw[0]
        if not isinstance(hw, str):
            hw = "other"
        matched = "other"
        for key in ROAD_COLORS:
            if hw.startswith(key):
                matched = key
                break
        xs, ys = geom.xy
        segments.append((np.array(xs), np.array(ys), matched))
    return segments


def compute_road_stats(G, gdf_edges, city):
    """Compute stats dict for a city's road network."""
    lon_min, lat_min, lon_max, lat_max = city["bbox"]
    bbox_area_km2 = (
        (lon_max - lon_min) * 111.0 * np.cos(np.radians((lat_min + lat_max) / 2))
        * (lat_max - lat_min) * 111.0
    )
    type_counts = {}
    total_length_m = 0.0
    for _, row in gdf_edges.iterrows():
        hw = row.get("highway", "other")
        if isinstance(hw, list):
            hw = hw[0]
        if not isinstance(hw, str):
            hw = "other"
        matched = "other"
        for key in ROAD_COLORS:
            if hw.startswith(key):
                matched = key
                break
        type_counts[matched] = type_counts.get(matched, 0) + 1
        length = row.get("length", 0) or 0
        total_length_m += float(length)

    density = (total_length_m / 1000.0) / bbox_area_km2 if bbox_area_km2 > 0 else 0
    avg_length = (total_length_m / len(gdf_edges)) if len(gdf_edges) > 0 else 0
    return {
        "n_nodes": len(G.nodes),
        "n_edges": len(G.edges),
        "n_segments": len(gdf_edges),
        "bbox_area_km2": bbox_area_km2,
        "total_length_km": total_length_m / 1000.0,
        "density_km_per_km2": density,
        "avg_length_m": avg_length,
        "type_counts": type_counts,
    }


def apply_dark_style(ax, title="", xlabel="Longitude", ylabel="Latitude"):
    """Apply professional dark GIS style to axes."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontsize=9, pad=4, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=7)
    ax.grid(True, color="#21262d", linewidth=0.4, linestyle="--", alpha=0.6)


def draw_road_segments(ax, segments):
    """Draw road segments on ax, grouped by type for efficiency."""
    grouped = {}
    for xs, ys, hw in segments:
        grouped.setdefault(hw, []).append((xs, ys))
    # Draw lower-priority roads first
    order = [
        "other", "unclassified", "service", "living_street",
        "residential", "tertiary", "secondary", "primary", "trunk", "motorway",
    ]
    for hw in order:
        if hw not in grouped:
            continue
        color = ROAD_COLORS.get(hw, ROAD_COLORS["other"])
        lw = ROAD_WIDTHS.get(hw, 0.4)
        for xs, ys in grouped[hw]:
            ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", alpha=0.85)


def add_north_arrow(ax, x=0.96, y=0.12, size=0.06):
    """Add a north arrow to axes (axes fraction coords)."""
    ax.annotate(
        "", xy=(x, y + size), xytext=(x, y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
    )
    ax.text(x, y - 0.03, "N", transform=ax.transAxes,
            ha="center", va="top", color="white", fontsize=8, fontweight="bold")


def add_scale_bar(ax, city_bbox, x=0.05, y=0.06, length_deg=0.05):
    """Add a simple scale bar. length_deg is approximate degrees of longitude."""
    lon_min, lat_min, lon_max, lat_max = city_bbox
    lat_mid = (lat_min + lat_max) / 2
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_mid))
    length_km = length_deg * km_per_deg_lon
    # Scale bar in axes fraction space
    ax_width_deg = lon_max - lon_min
    bar_frac = length_deg / ax_width_deg
    bar_x0 = x
    bar_x1 = x + bar_frac
    bar_y = y
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], transform=ax.transAxes,
            color="white", linewidth=2.0, solid_capstyle="butt")
    ax.plot([bar_x0, bar_x0], [bar_y - 0.01, bar_y + 0.01], transform=ax.transAxes,
            color="white", linewidth=1.5)
    ax.plot([bar_x1, bar_x1], [bar_y - 0.01, bar_y + 0.01], transform=ax.transAxes,
            color="white", linewidth=1.5)
    ax.text((bar_x0 + bar_x1) / 2, bar_y + 0.025, f"{length_km:.1f} km",
            transform=ax.transAxes, ha="center", va="bottom",
            color="white", fontsize=6.5)


def make_road_legend_handles():
    """Return legend handles for road types."""
    handles = []
    labels = []
    for hw, color in ROAD_COLORS.items():
        if hw == "other":
            continue
        handles.append(Line2D([0], [0], color=color, linewidth=2))
        labels.append(hw.replace("_", " ").title())
    return handles, labels


# ── Figure 1: City Overview ────────────────────────────────────────────────────

def figure_city_overview(city_data_list):
    """5-city overview in 2x3 grid."""
    print("\n[Figure 1] City Overview Map...")
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG_COLOR)

    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    axes = []
    for row, col in positions:
        ax = fig.add_subplot(2, 3, row * 3 + col + 1)
        axes.append(ax)

    for idx, (city, data) in enumerate(city_data_list):
        ax = axes[idx]
        G, gdf_nodes, gdf_edges, stats = data
        lon_min, lat_min, lon_max, lat_max = city["bbox"]

        if gdf_edges is not None and len(gdf_edges) > 0:
            segments = edges_to_segments(gdf_edges)
            draw_road_segments(ax, segments)
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
        else:
            ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_COLOR, fontsize=10)

        title = (
            f"{city['name']}\n"
            f"Segments: {stats['n_segments']:,}  |  Area: {stats['bbox_area_km2']:.1f} km²"
        )
        apply_dark_style(ax, title=title)

        # Grid lines every 0.1 degree
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.tick_params(axis="both", labelsize=6)

        add_north_arrow(ax)
        add_scale_bar(ax, city["bbox"])

    # Last panel: legend + summary stats
    ax_legend = fig.add_subplot(2, 3, 6)
    ax_legend.set_facecolor(BG_COLOR)
    ax_legend.axis("off")
    handles, labels = make_road_legend_handles()
    leg = ax_legend.legend(
        handles, labels,
        title="Road Type",
        title_fontsize=9,
        fontsize=8,
        loc="upper left",
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor=TEXT_COLOR,
    )
    leg.get_title().set_color(ACCENT_COLOR)
    ax_legend.set_title("Road Network Legend", color=TEXT_COLOR, fontsize=10,
                         fontweight="bold", pad=8)

    # Summary table
    y_pos = 0.45
    ax_legend.text(0.05, y_pos + 0.07, "Dataset Summary", color=ACCENT_COLOR,
                   fontsize=9, fontweight="bold", transform=ax_legend.transAxes)
    for city, data in city_data_list:
        _, _, _, stats = data
        ax_legend.text(
            0.05, y_pos,
            f"{city['name']:14s}: {stats['n_segments']:>5,} segs",
            color=TEXT_COLOR, fontsize=7.5, fontfamily="monospace",
            transform=ax_legend.transAxes,
        )
        y_pos -= 0.065

    fig.suptitle(
        "Road Networks — HHRoad Dataset Cities",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUTPUT_DIR, "city_overview.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Per-city detailed road network ───────────────────────────────────

def figure_roadnet_city(city, G, gdf_nodes, gdf_edges, stats):
    """Detailed road network map for a single city."""
    print(f"  Rendering {city['name']}...")
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(BG_COLOR)

    lon_min, lat_min, lon_max, lat_max = city["bbox"]

    if gdf_edges is not None and len(gdf_edges) > 0:
        segments = edges_to_segments(gdf_edges)
        draw_road_segments(ax, segments)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    apply_dark_style(ax, title=f"{city['name']} — Road Network")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.tick_params(axis="both", labelsize=8)

    add_north_arrow(ax, x=0.97, y=0.10, size=0.07)
    add_scale_bar(ax, city["bbox"], length_deg=0.04)

    # Legend
    handles, labels = make_road_legend_handles()
    leg = ax.legend(
        handles, labels,
        title="Road Type",
        title_fontsize=8,
        fontsize=7,
        loc="lower right",
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor=TEXT_COLOR,
    )
    leg.get_title().set_color(ACCENT_COLOR)

    # Stats annotation box
    type_lines = "\n".join(
        f"  {k}: {v:,}" for k, v in sorted(stats["type_counts"].items(),
                                             key=lambda x: -x[1])[:6]
    )
    info_text = (
        f"Nodes: {stats['n_nodes']:,}\n"
        f"Edges: {stats['n_edges']:,}\n"
        f"Segments: {stats['n_segments']:,}\n"
        f"Total length: {stats['total_length_km']:.1f} km\n"
        f"Density: {stats['density_km_per_km2']:.2f} km/km²\n"
        f"Avg seg length: {stats['avg_length_m']:.0f} m\n"
        f"Area: {stats['bbox_area_km2']:.1f} km²\n"
        f"Road types:\n{type_lines}"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                 edgecolor="#30363d", alpha=0.9)
    ax.text(
        0.015, 0.985, info_text,
        transform=ax.transAxes, fontsize=7, verticalalignment="top",
        color=TEXT_COLOR, bbox=props, fontfamily="monospace",
    )

    bbox_label = (
        f"BBox: ({lon_min:.2f},{lat_min:.2f}) – ({lon_max:.2f},{lat_max:.2f})"
    )
    ax.text(
        0.5, 0.005, bbox_label,
        transform=ax.transAxes, fontsize=6.5, ha="center", va="bottom",
        color="#8b949e",
    )

    fig.suptitle(
        f"{city['name']} Road Network  |  HHRoad Dataset",
        color=TEXT_COLOR, fontsize=12, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    out_path = os.path.join(OUTPUT_DIR, f"roadnet_{city['code']}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ── Figure 3: Dataset Statistics Comparison ────────────────────────────────────

def figure_dataset_stats(city_data_list):
    """Bar charts comparing dataset statistics across cities."""
    print("\n[Figure 3] Dataset Statistics Comparison...")
    cities_labels = [c["name"] for c, _ in city_data_list]
    stats_list = [data[3] for _, data in city_data_list]

    n_segments = [s["n_segments"] for s in stats_list]
    densities = [s["density_km_per_km2"] for s in stats_list]
    avg_lengths = [s["avg_length_m"] for s in stats_list]

    all_types = list(ROAD_COLORS.keys())
    type_matrix = np.zeros((len(cities_labels), len(all_types)), dtype=int)
    for i, s in enumerate(stats_list):
        for j, t in enumerate(all_types):
            type_matrix[i, j] = s["type_counts"].get(t, 0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor(BG_COLOR)
    bar_colors = [ROAD_COLORS["primary"], ROAD_COLORS["secondary"],
                  ROAD_COLORS["tertiary"], ROAD_COLORS["residential"],
                  ROAD_COLORS["trunk"]]

    # (a) Segments per city
    ax = axes[0, 0]
    ax.set_facecolor(BG_COLOR)
    x = np.arange(len(cities_labels))
    bars = ax.bar(x, n_segments, color=bar_colors, width=0.6, edgecolor="#30363d")
    ax.set_xticks(x)
    ax.set_xticklabels(cities_labels, color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Number of Road Segments", color=TEXT_COLOR, fontsize=9)
    ax.set_title("(a) Road Segments per City", color=TEXT_COLOR, fontsize=10,
                 fontweight="bold")
    for bar, val in zip(bars, n_segments):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(n_segments) * 0.01,
                f"{val:,}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)
    apply_dark_style(ax, xlabel="", ylabel="")
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.set_ylabel("Number of Road Segments", color=TEXT_COLOR, fontsize=9)

    # (b) Road network density
    ax = axes[0, 1]
    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(x, densities, color=bar_colors, width=0.6, edgecolor="#30363d")
    ax.set_xticks(x)
    ax.set_xticklabels(cities_labels, color=TEXT_COLOR, fontsize=9)
    ax.set_title("(b) Road Network Density", color=TEXT_COLOR, fontsize=10,
                 fontweight="bold")
    for bar, val in zip(bars, densities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(densities) * 0.01,
                f"{val:.2f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)
    apply_dark_style(ax, xlabel="", ylabel="")
    ax.set_ylabel("km of road / km²", color=TEXT_COLOR, fontsize=9)

    # (c) Road type distribution (stacked bar)
    ax = axes[1, 0]
    ax.set_facecolor(BG_COLOR)
    bottoms = np.zeros(len(cities_labels))
    for j, hw_type in enumerate(all_types):
        vals = type_matrix[:, j]
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottoms, label=hw_type.replace("_", " ").title(),
               color=ROAD_COLORS[hw_type], width=0.6, edgecolor="#30363d", alpha=0.9)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(cities_labels, color=TEXT_COLOR, fontsize=9)
    ax.set_title("(c) Road Type Distribution", color=TEXT_COLOR, fontsize=10,
                 fontweight="bold")
    leg = ax.legend(fontsize=7, loc="upper right", facecolor="#161b22",
                    edgecolor="#30363d", labelcolor=TEXT_COLOR, ncol=2)
    apply_dark_style(ax, xlabel="", ylabel="")
    ax.set_ylabel("Segment Count", color=TEXT_COLOR, fontsize=9)

    # (d) Average segment length
    ax = axes[1, 1]
    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(x, avg_lengths, color=bar_colors, width=0.6, edgecolor="#30363d")
    ax.set_xticks(x)
    ax.set_xticklabels(cities_labels, color=TEXT_COLOR, fontsize=9)
    ax.set_title("(d) Average Road Segment Length", color=TEXT_COLOR, fontsize=10,
                 fontweight="bold")
    for bar, val in zip(bars, avg_lengths):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_lengths) * 0.01,
                f"{val:.0f}m", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)
    apply_dark_style(ax, xlabel="", ylabel="")
    ax.set_ylabel("Average Length (m)", color=TEXT_COLOR, fontsize=9)

    for ax_row in axes:
        for ax_ in ax_row:
            ax_.tick_params(colors=TEXT_COLOR)
            ax_.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
            for spine in ax_.spines.values():
                spine.set_edgecolor("#30363d")
            ax_.grid(axis="y", color="#21262d", linewidth=0.5, linestyle="--", alpha=0.6)

    fig.suptitle(
        "HHRoad Dataset — Road Network Statistics Comparison",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(OUTPUT_DIR, "dataset_stats.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 4: Topology Analysis ────────────────────────────────────────────────

def figure_topology_analysis(city_data_list):
    """Degree distribution histograms + network stats per city."""
    print("\n[Figure 4] Topology Analysis...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.patch.set_facecolor(BG_COLOR)

    for idx, (city, data) in enumerate(city_data_list):
        ax = axes[idx]
        G, gdf_nodes, gdf_edges, stats = data
        ax.set_facecolor(BG_COLOR)

        if G is not None and len(G.nodes) > 0:
            degrees = [d for _, d in G.degree()]
            in_degrees = [d for _, d in G.in_degree()] if G.is_directed() else degrees
            out_degrees = [d for _, d in G.out_degree()] if G.is_directed() else degrees

            max_deg = max(degrees) if degrees else 1
            bins = min(max_deg + 1, 20)
            ax.hist(degrees, bins=bins, color=EDGE_COLOR_DEFAULT, edgecolor="#30363d",
                    alpha=0.85, density=True)
            avg_deg = np.mean(degrees) if degrees else 0
            max_d = max(degrees) if degrees else 0

            # avg clustering (for undirected approximation)
            try:
                import networkx as nx
                ug = G.to_undirected()
                avg_clust = nx.average_clustering(ug)
            except Exception:
                avg_clust = float("nan")

            info = (
                f"Nodes: {stats['n_nodes']:,}\n"
                f"Edges: {stats['n_edges']:,}\n"
                f"Avg degree: {avg_deg:.2f}\n"
                f"Max degree: {max_d}\n"
                f"Avg clustering: {avg_clust:.3f}"
            )
            props = dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                         edgecolor="#30363d", alpha=0.85)
            ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=6.5,
                    verticalalignment="top", horizontalalignment="right",
                    color=TEXT_COLOR, bbox=props, fontfamily="monospace")
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center",
                    color=TEXT_COLOR, fontsize=12)

        apply_dark_style(ax, title=city["name"],
                         xlabel="Node Degree", ylabel="Density" if idx == 0 else "")
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        if idx > 0:
            ax.set_ylabel("")

    fig.suptitle(
        "Road Network Topology — Degree Distribution per City",
        color=TEXT_COLOR, fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "topology_analysis.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 5: Spatial Stats World Map ─────────────────────────────────────────

def figure_spatial_stats(city_data_list):
    """World map showing city locations, bubble-sized by segment count."""
    print("\n[Figure 5] Spatial Statistics Map...")
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw a simple world outline using a low-res polygon approximation
    # We use a manually crafted simple continental outline approach with
    # approximate continent rectangles for background context
    continent_rects = [
        # (x, y, width, height, label)
        (-10, 35, 40, 35, "Europe"),
        (-20, -40, 60, 75, "Africa"),
        (25, 10, 130, 70, "Asia"),
        (-170, 15, 130, 65, "N.America"),
        (-85, -60, 60, 70, "S.America"),
        (110, -45, 55, 45, "Australia"),
    ]
    for (x, y, w, h, label) in continent_rects:
        rect = Rectangle((x, y), w, h, linewidth=0.5, edgecolor="#30363d",
                          facecolor="#161b22", alpha=0.7)
        ax.add_patch(rect)

    # Try to draw world boundaries using Natural Earth data via geopandas
    try:
        import geopandas as gpd
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        world.plot(ax=ax, color="#1c2333", edgecolor="#30363d", linewidth=0.3)
    except Exception:
        # Fallback: simple rectangular approximation
        pass

    # City bubbles
    segment_counts = [data[3]["n_segments"] for _, data in city_data_list]
    max_segs = max(segment_counts) if segment_counts else 1
    bubble_colors = [
        ROAD_COLORS["primary"], ROAD_COLORS["secondary"],
        ROAD_COLORS["tertiary"], ROAD_COLORS["trunk"], ROAD_COLORS["motorway"],
    ]

    for i, (city, data) in enumerate(city_data_list):
        lon, lat = city["center"]
        n_segs = data[3]["n_segments"]
        bubble_size = 200 + 1500 * (n_segs / max_segs)
        ax.scatter(lon, lat, s=bubble_size, color=bubble_colors[i],
                   alpha=0.85, edgecolors="white", linewidths=1.2,
                   zorder=5)
        offset_y = 3.5 if i % 2 == 0 else -5.0
        ax.annotate(
            f"{city['name']}\n{n_segs:,} segs",
            xy=(lon, lat),
            xytext=(lon + 2, lat + offset_y),
            textcoords="data",
            color=TEXT_COLOR,
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            arrowprops=dict(arrowstyle="-", color="#8b949e", lw=0.8),
            zorder=6,
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 80)
    ax.set_xlabel("Longitude", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Latitude", color=TEXT_COLOR, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.4, linestyle="--", alpha=0.5)

    # Bubble size legend
    legend_sizes = [1000, 3000, 5000]
    legend_handles = [
        plt.scatter([], [], s=200 + 1500 * (s / max_segs),
                    color="#888888", alpha=0.7, edgecolors="white", linewidths=0.8)
        for s in legend_sizes
    ]
    leg = ax.legend(
        legend_handles,
        [f"{s:,} segs" for s in legend_sizes],
        title="Segment Count",
        title_fontsize=8,
        fontsize=7.5,
        loc="lower left",
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor=TEXT_COLOR,
        scatterpoints=1,
    )
    leg.get_title().set_color(ACCENT_COLOR)

    fig.suptitle(
        "HHRoad Dataset — Geographic Distribution of Study Cities",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(OUTPUT_DIR, "spatial_stats.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HHRoad GIS City Visualization")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"osmnx version: {ox.__version__}")

    # Step 1: Download all city data
    print("\n[Step 1] Downloading road network data from OpenStreetMap...")
    city_data_list = []
    for city in CITIES:
        print(f"\n--- {city['name']} ({city['code']}) ---")
        G = download_city_graph(city)
        if G is not None:
            try:
                gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
                stats = compute_road_stats(G, gdf_edges, city)
                print(
                    f"  Stats: {stats['n_segments']} segments, "
                    f"{stats['total_length_km']:.1f} km, "
                    f"{stats['density_km_per_km2']:.2f} km/km²"
                )
            except Exception as e:
                print(f"  ERROR converting to GDF: {e}")
                gdf_nodes, gdf_edges = None, None
                stats = {
                    "n_nodes": 0, "n_edges": 0, "n_segments": 0,
                    "bbox_area_km2": 0, "total_length_km": 0,
                    "density_km_per_km2": 0, "avg_length_m": 0,
                    "type_counts": {},
                }
        else:
            gdf_nodes, gdf_edges = None, None
            stats = {
                "n_nodes": 0, "n_edges": 0, "n_segments": 0,
                "bbox_area_km2": 0, "total_length_km": 0,
                "density_km_per_km2": 0, "avg_length_m": 0,
                "type_counts": {},
            }
        city_data_list.append((city, (G, gdf_nodes, gdf_edges, stats)))

    # Step 2: Generate figures
    print("\n[Step 2] Generating figures...")

    # Figure 1: City Overview
    try:
        figure_city_overview(city_data_list)
    except Exception as e:
        print(f"  ERROR in city_overview: {e}")
        traceback.print_exc()

    # Figure 2: Per-city detailed maps
    print("\n[Figure 2] Per-city detailed road network maps...")
    for city, data in city_data_list:
        G, gdf_nodes, gdf_edges, stats = data
        try:
            figure_roadnet_city(city, G, gdf_nodes, gdf_edges, stats)
        except Exception as e:
            print(f"  ERROR in roadnet_{city['code']}: {e}")
            traceback.print_exc()

    # Figure 3: Dataset Statistics
    try:
        figure_dataset_stats(city_data_list)
    except Exception as e:
        print(f"  ERROR in dataset_stats: {e}")
        traceback.print_exc()

    # Figure 4: Topology Analysis
    try:
        figure_topology_analysis(city_data_list)
    except Exception as e:
        print(f"  ERROR in topology_analysis: {e}")
        traceback.print_exc()

    # Figure 5: Spatial Stats World Map
    try:
        figure_spatial_stats(city_data_list)
    except Exception as e:
        print(f"  ERROR in spatial_stats: {e}")
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    generated_files = []
    expected_files = (
        ["city_overview.pdf"]
        + [f"roadnet_{c['code']}.pdf" for c in CITIES]
        + ["dataset_stats.pdf", "topology_analysis.pdf", "spatial_stats.pdf"]
    )
    for fname in expected_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  [OK] {fname:35s}  ({size_kb:.1f} KB)")
            generated_files.append(fpath)
        else:
            print(f"  [MISSING] {fname}")

    print(f"\nTotal files generated: {len(generated_files)}/{len(expected_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nCity statistics:")
    for city, data in city_data_list:
        _, _, _, stats = data
        print(
            f"  {city['name']:15s}: "
            f"{stats['n_nodes']:>6,} nodes, "
            f"{stats['n_edges']:>7,} edges, "
            f"{stats['n_segments']:>6,} segments, "
            f"{stats['total_length_km']:>8.1f} km"
        )


if __name__ == "__main__":
    main()
