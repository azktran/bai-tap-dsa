# bai-tap-dsa
bai tap nhom dsa city routing

"""
"""
"""
=============================================================================
CITY LOGISTICS & ROUTING SYSTEM — PART 1 OPTIMIZED
=============================================================================
Area    : Dong Da District, Hanoi, Vietnam
Goal    : Build a practical routing graph for shortest path, VRP, shipper
          allocation, congestion simulation, and logistics optimization.

Key improvements vs the original Part 1:
  - Uses simplify=True for routing performance while preserving edge geometry.
  - Filters roads that are unsuitable for motorcycle/vehicle logistics.
  - Keeps the largest weakly connected component to reduce unreachable pairs.
  - Separates routing graph concerns from GIS rendering/export.
  - Normalizes OSM attributes and numeric edge weights before saving.
  - Exports compact GeoJSON with only useful routing attributes.
=============================================================================
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd

try:
    import folium
except ImportError:
    folium = None

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PLACE = "Đống Đa, Hà Nội, Việt Nam"
OUTPUT_DIR = Path("./dong_da_output")
CACHE_DIR = Path("./osm_cache")

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.cache_folder = str(CACHE_DIR)
ox.settings.timeout = 300


# Speeds are tuned for inner-city Hanoi motorcycle/logistics routing.
# Unit: km/h.
HEURISTIC_SPEEDS = {
    "motorway": 70,
    "trunk": 55,
    "primary": 40,
    "secondary": 32,
    "tertiary": 26,
    "residential": 22,
    "service": 14,
    "unclassified": 18,
    "living_street": 10,
    "road": 20,
}

# Roads that should not be used for vehicle/motorcycle logistics.
BLOCKED_HIGHWAYS = {
    "footway",
    "steps",
    "pedestrian",
    "cycleway",
    "bridleway",
    "corridor",
    "elevator",
    "escalator",
    "platform",
    "proposed",
    "construction",
}

NO_ACCESS_VALUES = {"no", "private", "customers", "permit"}

EXPORT_EDGE_COLUMNS = [
    "u",
    "v",
    "key",
    "geometry",
    "osmid",
    "highway",
    "name",
    "length",
    "oneway",
    "lanes",
    "maxspeed",
    "speed_kph",
    "travel_time",
    "weight_distance",
    "weight_time",
    "weight_congestion",
    "weight_logistics",
    "congestion_factor",
    "capacity",
]

STATIC_ROAD_STYLE = {
    "motorway": ("#e67e22", "#f39c12", 3.2, 2.0, 9, 1.0),
    "trunk": ("#e67e22", "#f39c12", 3.0, 1.9, 9, 1.0),
    "primary": ("#d97706", "#fbbf24", 2.6, 1.6, 8, 1.0),
    "secondary": ("#9ca3af", "#ffffff", 2.0, 1.2, 7, 1.0),
    "tertiary": ("#9ca3af", "#ffffff", 1.5, 0.9, 6, 0.95),
    "residential": ("#a3a3a3", "#ffffff", 1.1, 0.7, 5, 0.9),
    "service": ("#bdbdbd", "#eeeeee", 0.8, 0.45, 4, 0.82),
    "living_street": ("#bdbdbd", "#f3f4f6", 0.8, 0.45, 4, 0.82),
    "unclassified": ("#bdbdbd", "#eeeeee", 0.8, 0.45, 4, 0.82),
    "road": ("#a3a3a3", "#ffffff", 1.0, 0.65, 5, 0.9),
    "_default": ("#bdbdbd", "#dddddd", 0.7, 0.4, 3, 0.75),
}

INTERACTIVE_ROAD_STYLE = {
    "motorway": ("#e67e22", 5),
    "trunk": ("#e67e22", 5),
    "primary": ("#f59e0b", 4),
    "secondary": ("#facc15", 3),
    "tertiary": ("#2563eb", 3),
    "residential": ("#6b7280", 2),
    "service": ("#9ca3af", 2),
    "living_street": ("#9ca3af", 2),
    "unclassified": ("#9ca3af", 2),
    "road": ("#6b7280", 2),
}


# =============================================================================
# HELPERS
# =============================================================================

def first_value(value: Any, default: Any = None) -> Any:
    """Return the first scalar value from common OSM list-like attributes."""
    if isinstance(value, (list, tuple, set)):
        return next(iter(value), default)
    return value if value is not None else default


def to_float(value: Any, default: float) -> float:
    """Convert OSM string/list values to float safely."""
    value = first_value(value, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int) -> int:
    """Convert OSM string/list values to int safely."""
    value = first_value(value, default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_highway(value: Any) -> str:
    """Normalize an OSM highway attribute to a single lowercase string."""
    value = first_value(value, "unclassified")
    return str(value).lower() if value is not None else "unclassified"


def is_blocked_for_logistics(edge_data: dict[str, Any]) -> bool:
    """
    Decide whether an OSM edge is unsuitable for logistics routing.

    This is conservative: it removes explicit pedestrian/cycle-only facilities
    and edges with clear vehicle or motorcycle access restrictions.
    """
    highway = normalize_highway(edge_data.get("highway"))
    if highway in BLOCKED_HIGHWAYS:
        return True

    access = str(first_value(edge_data.get("access"), "")).lower()
    vehicle = str(first_value(edge_data.get("vehicle"), "")).lower()
    motor_vehicle = str(first_value(edge_data.get("motor_vehicle"), "")).lower()
    motorcycle = str(first_value(edge_data.get("motorcycle"), "")).lower()

    if access in NO_ACCESS_VALUES:
        return True
    if vehicle == "no" or motor_vehicle == "no" or motorcycle == "no":
        return True

    return False


def keep_largest_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Keep the largest weakly connected component for directed road graphs."""
    if G.number_of_nodes() == 0:
        return G
    if nx.is_weakly_connected(G):
        return G

    largest = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(largest).copy()


def clean_object_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make GeoJSON export robust by converting unsupported object values.

    Geometry is left untouched. Lists/dicts are stringified; normal scalar values
    are kept as-is where possible.
    """
    cleaned = gdf.copy()
    for col in cleaned.columns:
        if col == cleaned.geometry.name:
            continue
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].map(
                lambda x: ", ".join(map(str, x))
                if isinstance(x, (list, tuple, set))
                else str(x)
                if isinstance(x, dict)
                else x
            )
    return cleaned


def display_value(value: Any, default: str = "Unnamed road") -> str:
    """Convert OSM scalar/list values to a readable string for map labels."""
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    if isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value if item is not None and not pd.isna(item)]
        return ", ".join(values) if values else default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    return text


def add_logistics_weights(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add routing weights needed by later shortest path and VRP stages."""
    G = ox.add_edge_speeds(G, hwy_speeds=HEURISTIC_SPEEDS)
    G = ox.add_edge_travel_times(G)

    congestion_by_highway = {
        "motorway": 1.15,
        "trunk": 1.30,
        "primary": 1.75,
        "secondary": 1.55,
        "tertiary": 1.30,
        "residential": 1.12,
        "service": 1.05,
        "living_street": 1.05,
        "unclassified": 1.12,
        "road": 1.15,
    }

    fuel_cost_per_km = 2000.0
    time_value_per_min = 500.0

    for _, _, _, data in G.edges(keys=True, data=True):
        highway = normalize_highway(data.get("highway"))
        length_m = to_float(data.get("length"), 50.0)
        speed_kph = to_float(data.get("speed_kph"), HEURISTIC_SPEEDS.get(highway, 20))
        travel_time_s = to_float(data.get("travel_time"), length_m / max(speed_kph, 1) * 3.6)
        lanes = max(to_int(data.get("lanes"), 1), 1)

        congestion = congestion_by_highway.get(highway, 1.12)
        distance_km = length_m / 1000.0
        time_min = travel_time_s / 60.0

        data["highway"] = highway
        data["length"] = length_m
        data["speed_kph"] = speed_kph
        data["travel_time"] = travel_time_s
        data["lanes"] = lanes
        data["weight_distance"] = distance_km
        data["weight_time"] = time_min
        data["congestion_factor"] = congestion
        data["weight_congestion"] = time_min * congestion
        data["weight_logistics"] = (
            time_min * time_value_per_min * congestion
            + distance_km * fuel_cost_per_km
        )
        data["capacity"] = lanes * speed_kph

    return G


def build_routing_graph() -> nx.MultiDiGraph:
    """
    Download and prepare a practical logistics routing graph.

    simplify=True is intentional: OSMnx preserves road geometry on simplified
    edges, but the graph has far fewer routing states than simplify=False.
    """
    print("\n[1/5] Downloading routing graph...")
    print(f"      Place: {PLACE}")
    print("      network_type=all, simplify=True, retain_all=False")

    start = time.time()
    G = ox.graph_from_place(
        PLACE,
        network_type="all",
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    raw_nodes = G.number_of_nodes()
    raw_edges = G.number_of_edges()

    blocked_edges = [
        (u, v, k)
        for u, v, k, data in G.edges(keys=True, data=True)
        if is_blocked_for_logistics(data)
    ]
    G.remove_edges_from(blocked_edges)
    G.remove_nodes_from(list(nx.isolates(G)))
    G = keep_largest_component(G)

    elapsed = time.time() - start
    print(f"  Raw graph:       {raw_nodes:,} nodes | {raw_edges:,} edges")
    print(f"  Removed edges:   {len(blocked_edges):,}")
    print(f"  Routing graph:   {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    print(f"  Download/clean:  {elapsed:.1f}s")

    return G


def download_feature_layer(tags: dict[str, Any], layer_name: str) -> gpd.GeoDataFrame | None:
    """Download polygon GIS features for optional map/dashboard use."""
    try:
        gdf = ox.features_from_place(PLACE, tags=tags)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        print(f"  {layer_name}: {len(gdf):,} polygons")
        return gdf
    except Exception as exc:
        print(f"  {layer_name}: skipped ({exc})")
        return None


def export_outputs(G: nx.MultiDiGraph) -> None:
    """Save GraphML, nodes GeoJSON, edges GeoJSON, and compact GIS layers."""
    print("\n[4/5] Converting graph to GeoDataFrames...")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

    print(f"  Nodes GDF: {len(nodes_gdf):,} rows | CRS: {nodes_gdf.crs}")
    print(f"  Edges GDF: {len(edges_gdf):,} rows | CRS: {edges_gdf.crs}")

    print("\n[5/5] Saving outputs...")

    graphml_path = OUTPUT_DIR / "dong_da_graph.graphml"
    ox.save_graphml(G, filepath=graphml_path)
    print(f"  GraphML:       {graphml_path}")

    nodes_path = OUTPUT_DIR / "dong_da_nodes.geojson"
    nodes_export = clean_object_columns(nodes_gdf)
    nodes_export.to_file(nodes_path, driver="GeoJSON")
    print(f"  Nodes GeoJSON: {nodes_path}")

    edges_path = OUTPUT_DIR / "dong_da_edges.geojson"
    edges_export = clean_object_columns(edges_gdf.reset_index())
    keep_cols = [col for col in EXPORT_EDGE_COLUMNS if col in edges_export.columns]
    edges_export = edges_export[keep_cols]
    edges_export.to_file(edges_path, driver="GeoJSON")
    print(f"  Edges GeoJSON: {edges_path}")


def load_saved_layer(name: str) -> gpd.GeoDataFrame | None:
    """Load an optional saved GeoJSON layer if it exists."""
    path = OUTPUT_DIR / f"dong_da_{name}.geojson"
    if not path.exists():
        return None
    try:
        return gpd.read_file(path)
    except Exception:
        return None


def static_road_style(highway: Any) -> tuple[str, str, float, float, int, float]:
    highway = normalize_highway(highway)
    return STATIC_ROAD_STYLE.get(highway, STATIC_ROAD_STYLE["_default"])


def iter_latlon_paths(geometry: Any):
    """Yield Folium-ready [(lat, lon), ...] paths from line geometries."""
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield [(lat, lon) for lon, lat in geometry.coords]
    elif geometry.geom_type == "MultiLineString":
        for part in geometry.geoms:
            yield [(lat, lon) for lon, lat in part.coords]


def save_static_map(G: nx.MultiDiGraph) -> None:
    """Save a static, report-friendly PNG map of the optimized graph."""
    print("\n[MAP] Rendering static PNG map...")

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    buildings = load_saved_layer("buildings")
    parks = load_saved_layer("parks")
    water = load_saved_layer("water")

    fig, ax = plt.subplots(1, 1, figsize=(18, 20), dpi=220, facecolor="#f4efe7")
    ax.set_facecolor("#f4efe7")

    bounds = nodes_gdf.total_bounds
    margin = 0.004
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

    if parks is not None and len(parks) > 0:
        parks.plot(
            ax=ax,
            color="#c8e6c9",
            edgecolor="#a5d6a7",
            linewidth=0.25,
            alpha=0.85,
            zorder=1,
        )

    if water is not None and len(water) > 0:
        water.plot(
            ax=ax,
            color="#b3d9ff",
            edgecolor="#90c4e8",
            linewidth=0.35,
            alpha=0.92,
            zorder=2,
        )

    if buildings is not None and len(buildings) > 0:
        buildings.plot(
            ax=ax,
            color="#d6cfc8",
            edgecolor="#bdb5ad",
            linewidth=0.08,
            alpha=0.72,
            zorder=3,
        )

    highway_col = edges_gdf["highway"].map(normalize_highway)
    render_order = [
        "service",
        "living_street",
        "unclassified",
        "road",
        "residential",
        "tertiary",
        "secondary",
        "primary",
        "trunk",
        "motorway",
    ]

    other_types = [t for t in highway_col.unique() if t not in render_order]
    for road_type in other_types + render_order:
        subset = edges_gdf[highway_col == road_type]
        if len(subset) == 0:
            continue

        outline, fill, outline_width, fill_width, zorder, alpha = static_road_style(road_type)
        subset.plot(
            ax=ax,
            color=outline,
            linewidth=outline_width,
            alpha=alpha * 0.65,
            zorder=zorder,
            capstyle="round",
            joinstyle="round",
        )
        subset.plot(
            ax=ax,
            color=fill,
            linewidth=fill_width,
            alpha=alpha,
            zorder=zorder + 0.1,
            capstyle="round",
            joinstyle="round",
        )

    ax.set_title(
        "Dong Da District - Optimized Logistics Road Graph",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=12,
        color="#111827",
    )
    ax.text(
        1.0,
        1.01,
        "OpenStreetMap | OSMnx + GeoPandas",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#6b7280",
    )
    ax.set_axis_off()

    out_path = OUTPUT_DIR / "dong_da_static_map.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Static map PNG: {out_path}")


def save_interactive_map(G: nx.MultiDiGraph) -> None:
    """Save an interactive HTML map with road tooltips and optional layers."""
    print("\n[MAP] Rendering interactive HTML map...")

    if folium is None:
        print("  Folium is not installed. Run: pip install folium")
        return

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    center_lat = float(nodes_gdf.geometry.y.mean())
    center_lon = float(nodes_gdf.geometry.x.mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="cartodbpositron",
        control_scale=True,
    )

    optional_layers = [
        ("Parks", "parks", "#7cb342", "#c8e6c9"),
        ("Water", "water", "#42a5f5", "#b3d9ff"),
        ("Buildings", "buildings", "#9e9e9e", "#d6cfc8"),
    ]
    for label, name, border_color, fill_color in optional_layers:
        layer = load_saved_layer(name)
        if layer is None or len(layer) == 0:
            continue
        folium.GeoJson(
            layer,
            name=label,
            style_function=lambda _feature, bc=border_color, fc=fill_color: {
                "color": bc,
                "weight": 0.5,
                "fillColor": fc,
                "fillOpacity": 0.45,
            },
        ).add_to(m)

    road_group = folium.FeatureGroup(name="Routing roads", show=True)
    for _, row in edges_gdf.iterrows():
        highway = normalize_highway(row.get("highway"))
        color, width = INTERACTIVE_ROAD_STYLE.get(highway, ("#6b7280", 2))
        road_name = display_value(row.get("name"))
        length = to_float(row.get("length"), 0.0)
        time_min = to_float(row.get("weight_time"), 0.0)

        tooltip = (
            f"{road_name}"
            f" | {highway}"
            f" | {length:.0f} m"
            f" | {time_min:.1f} min"
        )

        for path in iter_latlon_paths(row.geometry):
            folium.PolyLine(
                path,
                color=color,
                weight=width,
                opacity=0.78,
                tooltip=tooltip,
            ).add_to(road_group)

    road_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    out_path = OUTPUT_DIR / "dong_da_interactive_map.html"
    m.save(out_path)
    print(f"  Interactive map HTML: {out_path}")


def save_feature_layers() -> None:
    """
    Download useful GIS layers once and save them for dashboards/rendering.

    The routing graph does not depend on these layers, so failures here should
    not block shortest path or VRP work.
    """
    print("\n[3/5] Downloading optional GIS feature layers...")

    layers = {
        "buildings": download_feature_layer({"building": True}, "Buildings"),
        "parks": download_feature_layer(
            {
                "leisure": ["park", "garden", "recreation_ground"],
                "landuse": ["grass", "meadow", "recreation_ground", "village_green"],
            },
            "Parks/Green",
        ),
        "water": download_feature_layer(
            {
                "natural": ["water", "wetland"],
                "waterway": ["riverbank"],
                "landuse": ["reservoir", "basin"],
            },
            "Water",
        ),
    }

    for name, gdf in layers.items():
        if gdf is None or len(gdf) == 0:
            continue
        out_path = OUTPUT_DIR / f"dong_da_{name}.geojson"
        clean_object_columns(gdf[["geometry"]].copy()).to_file(out_path, driver="GeoJSON")
        print(f"  {name.title()} GeoJSON: {out_path}")


def print_summary(G: nx.MultiDiGraph) -> None:
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    highway_counts = edges["highway"].value_counts().head(12)
    total_km = edges["length"].sum() / 1000.0 if "length" in edges.columns else 0.0

    print("\n" + "=" * 70)
    print("  PART 1 COMPLETE — Optimized Routing Graph Built")
    print("=" * 70)
    print(f"""
GRAPH SUMMARY:
  Type        : {type(G).__name__}
  Nodes       : {G.number_of_nodes():,}
  Edges       : {G.number_of_edges():,}
  simplify    : True  (faster routing, edge geometry retained)
  component   : Largest weakly connected component
  CRS         : {G.graph.get("crs", "unknown")}
  Length      : {total_km:.1f} km

EDGE WEIGHTS:
  weight_distance   — distance in km
  weight_time       — travel time in minutes
  weight_congestion — time × congestion factor
  weight_logistics  — composite VND-like cost

OUTPUT FILES:
  {OUTPUT_DIR / "dong_da_graph.graphml"}
  {OUTPUT_DIR / "dong_da_nodes.geojson"}
  {OUTPUT_DIR / "dong_da_edges.geojson"}
  {OUTPUT_DIR / "dong_da_buildings.geojson"}
  {OUTPUT_DIR / "dong_da_parks.geojson"}
  {OUTPUT_DIR / "dong_da_water.geojson"}
""")

    print("TOP ROAD TYPES:")
    for road_type, count in highway_counts.items():
        bar = "#" * min(int(count / max(highway_counts.max(), 1) * 30), 30)
        print(f"  {str(road_type):18s}: {count:5,}  {bar}")

    print("\nREADY FOR PART 2:")
    print('  G = ox.load_graphml("./dong_da_output/dong_da_graph.graphml")')
    print('  path = nx.shortest_path(G, source=u, target=v, weight="weight_time")')
    print("=" * 70)


def main() -> None:
    print("=" * 70)
    print("  CITY LOGISTICS & ROUTING — PART 1 OPTIMIZED")
    print("  Dong Da District, Hanoi, Vietnam")
    print("=" * 70)

    G = build_routing_graph()

    print("\n[2/5] Adding logistics weights...")
    G = add_logistics_weights(G)
    print("  Added: distance, time, congestion, logistics cost, capacity")

    save_feature_layers()
    export_outputs(G)
    save_static_map(G)
    save_interactive_map(G)
    print_summary(G)


if __name__ == "__main__":
    main()


"""
=============================================================================
CITY LOGISTICS & ROUTING SYSTEM — PARTS 2 TO 5
=============================================================================
Requires Part 1 output:
  ./dong_da_output/dong_da_graph.graphml

Pipeline:
  Part 2 — shortest routes + distance/time matrices
  Part 3 — zone division
  Part 4 — VRP route optimization
  Part 5 — simulation + dashboard
=============================================================================
"""

import heapq
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt

try:
    import folium
except ImportError:
    folium = None


OUTPUT_DIR = Path("./dong_da_output")
GRAPH_PATH = OUTPUT_DIR / "dong_da_graph.graphml"

N_DEPOTS = 3
N_DELIVERIES = 100
N_SHIPPERS = 8
RANDOM_SEED = 42

SHIPPER_CAPACITY_KG = 45.0
SHIPPER_MAX_TIME_MIN = 300.0
SERVICE_TIME_MIN = 3.0
WORKING_START_HOUR = 8

UNREACHABLE = 9999.0

MAP_COLORS = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#34495e",
    "#ff6b6b",
    "#4ecdc4",
]


# =============================================================================
# COMMON HELPERS
# =============================================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def to_float(value: Any, default: float) -> float:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_graph_numeric_attrs(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    numeric_attrs = [
        "length",
        "speed_kph",
        "travel_time",
        "weight_distance",
        "weight_time",
        "weight_congestion",
        "weight_logistics",
        "congestion_factor",
        "capacity",
    ]
    for _, _, _, data in G.edges(keys=True, data=True):
        for attr in numeric_attrs:
            if attr in data:
                data[attr] = to_float(data[attr], 0.0)
    return G


def load_graph() -> nx.MultiDiGraph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {GRAPH_PATH}. Run part1_optimized.py first."
        )
    G = ox.load_graphml(GRAPH_PATH)
    G = normalize_graph_numeric_attrs(G)
    if not nx.is_weakly_connected(G):
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    return G


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def matrix_index_to_order_idx(matrix_idx: int) -> int:
    return matrix_idx - N_DEPOTS


def order_idx_to_matrix_index(order_idx: int) -> int:
    return order_idx + N_DEPOTS


def graph_node_from_matrix_index(matrix_idx: int) -> str:
    matrix_nodes = load_json(OUTPUT_DIR / "matrix_nodes.json")["nodes"]
    return str(matrix_nodes[matrix_idx])


def node_latlon(G: nx.MultiDiGraph, node: Any) -> tuple[float, float]:
    data = G.nodes[str(node)]
    return float(data["y"]), float(data["x"])


def edge_latlon_path(G: nx.MultiDiGraph, u: Any, v: Any, weight: str = "weight_time") -> list[tuple[float, float]]:
    u = str(u)
    v = str(v)
    edge_dict = G.get_edge_data(u, v)
    if not edge_dict:
        return [node_latlon(G, u), node_latlon(G, v)]

    _, data = min(
        edge_dict.items(),
        key=lambda item: to_float(item[1].get(weight), to_float(item[1].get("length"), 1.0)),
    )
    geometry = data.get("geometry")
    if geometry is not None and hasattr(geometry, "coords"):
        return [(lat, lon) for lon, lat in geometry.coords]
    return [node_latlon(G, u), node_latlon(G, v)]


def route_latlon_path(G: nx.MultiDiGraph, route_nodes: list[Any]) -> list[tuple[float, float]]:
    full_path: list[tuple[float, float]] = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        segment = edge_latlon_path(G, u, v)
        if full_path and segment:
            segment = segment[1:]
        full_path.extend(segment)
    return full_path


def shortest_node_route(G: nx.MultiDiGraph, src: Any, dst: Any, weight: str = "weight_time") -> list[str]:
    try:
        return nx.shortest_path(G, str(src), str(dst), weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def make_base_folium_map(G: nx.MultiDiGraph, name: str) -> Any:
    if folium is None:
        return None

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    center = [float(nodes_gdf.geometry.y.mean()), float(nodes_gdf.geometry.x.mean())]
    fmap = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron", control_scale=True)

    for layer_name, file_name, stroke, fill, opacity in [
        ("Parks", "dong_da_parks.geojson", "#7cb342", "#c8e6c9", 0.45),
        ("Water", "dong_da_water.geojson", "#42a5f5", "#b3d9ff", 0.65),
        ("Buildings", "dong_da_buildings.geojson", "#9e9e9e", "#d6cfc8", 0.32),
    ]:
        path = OUTPUT_DIR / file_name
        if not path.exists():
            continue
        try:
            layer = path.read_text(encoding="utf-8")
            folium.GeoJson(
                layer,
                name=layer_name,
                show=True,
                style_function=lambda _feature, s=stroke, f=fill, o=opacity: {
                    "color": s,
                    "weight": 0.4,
                    "fillColor": f,
                    "fillOpacity": o,
                },
            ).add_to(fmap)
        except Exception:
            pass

    road_style = {
        "motorway": ("#f39c12", 4),
        "trunk": ("#f39c12", 4),
        "primary": ("#f59e0b", 4),
        "secondary": ("#ffffff", 3),
        "tertiary": ("#ffffff", 2),
        "residential": ("#ffffff", 2),
        "service": ("#d1d5db", 1),
        "living_street": ("#d1d5db", 1),
        "unclassified": ("#d1d5db", 1),
        "road": ("#ffffff", 2),
    }

    road_group = folium.FeatureGroup(name="Part 1 detailed road graph", show=True)
    for _, row in edges_gdf.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        highway = str(row.get("highway", "unclassified")).lower()
        color, width = road_style.get(highway, ("#d1d5db", 1))
        coords = [(lat, lon) for lon, lat in geometry.coords]
        folium.PolyLine(coords, color=color, weight=width, opacity=0.45).add_to(road_group)
    road_group.add_to(fmap)

    title_html = f"""
    <div style="position: fixed; top: 12px; left: 50px; z-index: 9999;
                background: white; padding: 8px 12px; border: 1px solid #ddd;
                border-radius: 6px; font-family: Arial; font-size: 14px;">
      <b>{name}</b>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    return fmap


def save_folium_map(fmap: Any, filename: str) -> None:
    if fmap is None:
        print(f"  Skipped {filename}: folium is not installed")
        return
    folium.LayerControl(collapsed=False).add_to(fmap)
    path = OUTPUT_DIR / filename
    fmap.save(path)
    print(f"  Interactive map: {path}")


def load_layer_geojson(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    try:
        return ox.io.gpd.read_file(path)
    except Exception:
        try:
            import geopandas as gpd

            return gpd.read_file(path)
        except Exception:
            return None


def plot_part1_detailed_base(
    ax: Any,
    G: nx.MultiDiGraph,
    title: str,
    show_title: bool = True,
) -> None:
    """Draw the same detailed base map style as Part 1, then overlays can be added."""
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

    ax.set_facecolor("#f4efe7")
    bounds = nodes_gdf.total_bounds
    margin = 0.004
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

    for filename, color, edge, alpha, zorder in [
        ("dong_da_parks.geojson", "#c8e6c9", "#a5d6a7", 0.85, 1),
        ("dong_da_water.geojson", "#b3d9ff", "#90c4e8", 0.92, 2),
        ("dong_da_buildings.geojson", "#d6cfc8", "#bdb5ad", 0.72, 3),
    ]:
        layer = load_layer_geojson(filename)
        if layer is not None and len(layer) > 0:
            layer.plot(
                ax=ax,
                color=color,
                edgecolor=edge,
                linewidth=0.08 if "buildings" in filename else 0.25,
                alpha=alpha,
                zorder=zorder,
            )

    style = {
        "motorway": ("#e67e22", "#f39c12", 3.2, 2.0, 9, 1.0),
        "trunk": ("#e67e22", "#f39c12", 3.0, 1.9, 9, 1.0),
        "primary": ("#d97706", "#fbbf24", 2.6, 1.6, 8, 1.0),
        "secondary": ("#9ca3af", "#ffffff", 2.0, 1.2, 7, 1.0),
        "tertiary": ("#9ca3af", "#ffffff", 1.5, 0.9, 6, 0.95),
        "residential": ("#a3a3a3", "#ffffff", 1.1, 0.7, 5, 0.9),
        "service": ("#bdbdbd", "#eeeeee", 0.8, 0.45, 4, 0.82),
        "living_street": ("#bdbdbd", "#f3f4f6", 0.8, 0.45, 4, 0.82),
        "unclassified": ("#bdbdbd", "#eeeeee", 0.8, 0.45, 4, 0.82),
        "road": ("#a3a3a3", "#ffffff", 1.0, 0.65, 5, 0.9),
    }
    highway_col = edges_gdf["highway"].map(lambda value: str(value).lower())
    render_order = [
        "service",
        "living_street",
        "unclassified",
        "road",
        "residential",
        "tertiary",
        "secondary",
        "primary",
        "trunk",
        "motorway",
    ]
    other_types = [t for t in highway_col.unique() if t not in render_order]
    for road_type in other_types + render_order:
        subset = edges_gdf[highway_col == road_type]
        if len(subset) == 0:
            continue
        outline, fill, outline_width, fill_width, zorder, alpha = style.get(
            road_type,
            ("#bdbdbd", "#dddddd", 0.7, 0.4, 3, 0.75),
        )
        subset.plot(
            ax=ax,
            color=outline,
            linewidth=outline_width,
            alpha=alpha * 0.6,
            zorder=zorder,
            capstyle="round",
            joinstyle="round",
        )
        subset.plot(
            ax=ax,
            color=fill,
            linewidth=fill_width,
            alpha=alpha,
            zorder=zorder + 0.1,
            capstyle="round",
            joinstyle="round",
        )

    if show_title:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10, color="#111827")
    ax.set_axis_off()


# =============================================================================
# PART 2 — SHORTEST ROUTE CALCULATION
# =============================================================================

def generate_delivery_points(G: nx.MultiDiGraph) -> tuple[list[int], list[int], list[int]]:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    nodes = list(G.nodes())
    if len(nodes) < N_DEPOTS + N_DELIVERIES:
        raise ValueError("Graph has fewer nodes than requested depots + deliveries.")

    sampled = random.sample(nodes, N_DEPOTS + N_DELIVERIES)
    depots = sampled[:N_DEPOTS]
    deliveries = sampled[N_DEPOTS:]
    return depots, deliveries, depots + deliveries


def build_point_metadata(G: nx.MultiDiGraph, depots: list[int], deliveries: list[int]) -> dict[str, Any]:
    rng = random.Random(RANDOM_SEED)
    return {
        "depots": [
            {
                "node_id": str(node),
                "matrix_index": i,
                "lat": float(G.nodes[node]["y"]),
                "lon": float(G.nodes[node]["x"]),
                "name": f"Kho {i + 1}",
            }
            for i, node in enumerate(depots)
        ],
        "deliveries": [
            {
                "node_id": str(node),
                "matrix_index": order_idx_to_matrix_index(i),
                "order_index": i,
                "order_id": f"ORD-{1000 + i}",
                "lat": float(G.nodes[node]["y"]),
                "lon": float(G.nodes[node]["x"]),
                "weight_kg": round(rng.uniform(0.5, 8.0), 1),
                "time_window": [
                    f"{rng.randint(8, 11):02d}:00",
                    f"{rng.randint(14, 18):02d}:00",
                ],
            }
            for i, node in enumerate(deliveries)
        ],
    }


def build_matrix(
    G: nx.MultiDiGraph,
    points: list[int],
    weight: str,
    label: str,
) -> np.ndarray:
    n = len(points)
    matrix = np.full((n, n), UNREACHABLE, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)

    point_set = set(points)
    point_to_idx = {node: idx for idx, node in enumerate(points)}

    start = time.time()
    for i, source in enumerate(points):
        lengths = nx.single_source_dijkstra_path_length(G, source, weight=weight)
        for target, value in lengths.items():
            if target in point_set:
                matrix[i, point_to_idx[target]] = float(value)

        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {label}: {i + 1}/{n} | elapsed={elapsed:.1f}s | ETA={eta:.0f}s")

    return matrix


def part2_shortest_routes(G: nx.MultiDiGraph) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("PART 2 — SHORTEST ROUTE CALCULATION")
    print("=" * 70)

    depots, deliveries, all_points = generate_delivery_points(G)
    metadata = build_point_metadata(G, depots, deliveries)
    save_json(OUTPUT_DIR / "delivery_nodes.json", metadata)
    save_json(
        OUTPUT_DIR / "matrix_nodes.json",
        {"nodes": [str(n) for n in all_points], "n_depots": N_DEPOTS},
    )

    print(f"Generated {N_DEPOTS} depots and {N_DELIVERIES} delivery points")

    time_matrix = build_matrix(G, all_points, "weight_time", "time_matrix")
    dist_matrix = build_matrix(G, all_points, "weight_distance", "dist_matrix")

    np.save(OUTPUT_DIR / "time_matrix.npy", time_matrix)
    np.save(OUTPUT_DIR / "dist_matrix.npy", dist_matrix)

    finite = time_matrix < UNREACHABLE
    summary = {
        "n_points": len(all_points),
        "reachable_pairs": int(finite.sum()),
        "total_pairs": int(time_matrix.size),
        "reachable_pct": round(float(finite.mean() * 100), 2),
        "mean_time_min": round(float(time_matrix[(time_matrix > 0) & finite].mean()), 2),
    }
    save_json(OUTPUT_DIR / "part2_summary.json", summary)
    visualize_part2_shortest_routes(G, metadata)
    print(f"Reachable pairs: {summary['reachable_pct']}%")
    return metadata


def visualize_part2_shortest_routes(G: nx.MultiDiGraph, metadata: dict[str, Any]) -> None:
    """Visualize 100 orders and shortest routes from the nearest depot."""
    print("\n[VIS] Part 2 shortest routes...")

    depots = metadata["depots"]
    deliveries = metadata["deliveries"]
    time_matrix = np.load(OUTPUT_DIR / "time_matrix.npy")

    # Assign each order to the depot with the lowest travel time.
    route_specs = []
    for delivery in deliveries:
        dst_idx = int(delivery["matrix_index"])
        best_depot = min(depots, key=lambda depot: float(time_matrix[int(depot["matrix_index"]), dst_idx]))
        src_node = best_depot["node_id"]
        dst_node = delivery["node_id"]
        route_nodes = shortest_node_route(G, src_node, dst_node)
        if route_nodes:
            route_specs.append((best_depot, delivery, route_nodes))

    fig, ax = plt.subplots(figsize=(14, 14), dpi=180, facecolor="#f4efe7")
    plot_part1_detailed_base(
        ax,
        G,
        "Part 2 - 100 Orders and Shortest Routes on Part 1 Map",
    )

    for depot_idx, depot in enumerate(depots):
        color = MAP_COLORS[depot_idx % len(MAP_COLORS)]
        depot_routes = [spec for spec in route_specs if spec[0]["name"] == depot["name"]]
        for _, _, route_nodes in depot_routes:
            path = route_latlon_path(G, route_nodes)
            if len(path) < 2:
                continue
            lats = [p[0] for p in path]
            lons = [p[1] for p in path]
            ax.plot(lons, lats, color=color, linewidth=0.9, alpha=0.42, zorder=20)

        ax.scatter(depot["lon"], depot["lat"], s=130, marker="s", color=color, edgecolor="white", zorder=31)
        ax.text(depot["lon"], depot["lat"], depot["name"], fontsize=8, color="#111827", zorder=32)

    ax.scatter(
        [d["lon"] for d in deliveries],
        [d["lat"] for d in deliveries],
        s=18,
        color="#f97316",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.9,
        zorder=30,
    )
    out_path = OUTPUT_DIR / "part2_shortest_routes.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Static map: {out_path}")

    fmap = make_base_folium_map(G, "Part 2 - Shortest Routes for 100 Orders")
    if fmap is not None:
        for depot_idx, depot in enumerate(depots):
            color = MAP_COLORS[depot_idx % len(MAP_COLORS)]
            group = folium.FeatureGroup(name=f"{depot['name']} shortest routes", show=True)
            for _, delivery, route_nodes in [spec for spec in route_specs if spec[0]["name"] == depot["name"]]:
                path = route_latlon_path(G, route_nodes)
                if len(path) >= 2:
                    folium.PolyLine(
                        path,
                        color=color,
                        weight=2,
                        opacity=0.45,
                        tooltip=f"{depot['name']} -> {delivery['order_id']}",
                    ).add_to(group)
                folium.CircleMarker(
                    [delivery["lat"], delivery["lon"]],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=f"{delivery['order_id']} | {delivery['weight_kg']}kg",
                ).add_to(group)
            folium.Marker(
                [depot["lat"], depot["lon"]],
                tooltip=depot["name"],
                icon=folium.Icon(color="red", icon="home"),
            ).add_to(group)
            group.add_to(fmap)
    save_folium_map(fmap, "part2_shortest_routes.html")


# =============================================================================
# PART 3 — ZONE DIVISION
# =============================================================================

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = 6) -> str:
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    is_lon = True
    bits = []

    for _ in range(precision * 5):
        if is_lon:
            mid = sum(lon_range) / 2
            bit = lon >= mid
            lon_range[0 if bit else 1] = mid
        else:
            mid = sum(lat_range) / 2
            bit = lat >= mid
            lat_range[0 if bit else 1] = mid
        bits.append(1 if bit else 0)
        is_lon = not is_lon

    chars = []
    for i in range(0, len(bits), 5):
        value = 0
        for bit in bits[i : i + 5]:
            value = value * 2 + bit
        chars.append(BASE32[value])
    return "".join(chars)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def sets(self) -> dict[int, list[int]]:
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(h))


def part3_zone_division(G: nx.MultiDiGraph | None = None) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("PART 3 — ZONE DIVISION")
    print("=" * 70)

    data = load_json(OUTPUT_DIR / "delivery_nodes.json")
    deliveries = data["deliveries"]
    time_matrix = np.load(OUTPUT_DIR / "time_matrix.npy")

    coords = [(d["lat"], d["lon"]) for d in deliveries]
    weights = [float(d["weight_kg"]) for d in deliveries]

    uf = UnionFind(len(deliveries))
    merge_radius_m = 650.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if haversine_m(coords[i], coords[j]) <= merge_radius_m:
                uf.union(i, j)

    components = sorted(uf.sets().values(), key=len, reverse=True)

    # Greedy bin packing by component size, then refine by nearest zone centroid.
    zones = {i: [] for i in range(N_SHIPPERS)}
    zone_loads = [(0, i) for i in range(N_SHIPPERS)]
    heapq.heapify(zone_loads)

    for comp in components:
        load, zone_id = heapq.heappop(zone_loads)
        zones[zone_id].extend(comp)
        heapq.heappush(zone_loads, (load + len(comp), zone_id))

    # If a zone is empty, move the farthest item from the largest zone into it.
    for zone_id in range(N_SHIPPERS):
        if zones[zone_id]:
            continue
        largest_zone = max(zones, key=lambda z: len(zones[z]))
        zones[zone_id].append(zones[largest_zone].pop())

    depots = data["depots"]
    zone_summary = {}
    for zone_id, members in zones.items():
        lat = float(np.mean([coords[i][0] for i in members]))
        lon = float(np.mean([coords[i][1] for i in members]))

        depot_times = []
        for depot in depots:
            depot_idx = int(depot["matrix_index"])
            member_matrix_indices = [order_idx_to_matrix_index(i) for i in members]
            mean_time = float(np.mean(time_matrix[depot_idx, member_matrix_indices]))
            depot_times.append((mean_time, depot["name"]))

        nearest_depot = min(depot_times, key=lambda x: x[0])[1]
        zone_summary[str(zone_id)] = {
            "zone_id": zone_id,
            "n_deliveries": len(members),
            "delivery_indices": sorted(members),
            "delivery_ids": [deliveries[i]["order_id"] for i in sorted(members)],
            "centroid_lat": lat,
            "centroid_lon": lon,
            "total_weight_kg": round(sum(weights[i] for i in members), 1),
            "geohash_prefix": encode_geohash(lat, lon, precision=5),
            "nearest_depot": nearest_depot,
        }

    save_json(OUTPUT_DIR / "zone_assignments.json", zone_summary)
    visualize_part3_zones(zone_summary, data, G)
    print(f"Saved {len(zone_summary)} zones")
    return zone_summary


def visualize_part3_zones(
    zone_summary: dict[str, Any],
    delivery_data: dict[str, Any],
    G: nx.MultiDiGraph | None = None,
) -> None:
    """Visualize zone division as a static scatter map and an interactive map."""
    print("\n[VIS] Part 3 zones...")

    deliveries = delivery_data["deliveries"]
    depots = delivery_data["depots"]

    fig, ax = plt.subplots(figsize=(14, 14), dpi=180, facecolor="#f4efe7")
    if G is not None:
        plot_part1_detailed_base(
            ax,
            G,
            "Part 3 - Zone Division on Part 1 Map",
        )
    else:
        ax.set_facecolor("#f4efe7")

    for zone_id_str, zone in zone_summary.items():
        zone_id = int(zone_id_str)
        color = MAP_COLORS[zone_id % len(MAP_COLORS)]
        members = zone["delivery_indices"]
        lats = [deliveries[i]["lat"] for i in members]
        lons = [deliveries[i]["lon"] for i in members]
        ax.scatter(lons, lats, s=42, color=color, alpha=0.9, edgecolor="white", linewidth=0.35, zorder=30)
        ax.text(
            zone["centroid_lon"],
            zone["centroid_lat"],
            f"Z{zone_id}\n{len(members)} orders",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "alpha": 0.75, "edgecolor": "none"},
            zorder=35,
        )

    for depot in depots:
        ax.scatter(depot["lon"], depot["lat"], s=180, marker="*", color="#facc15", edgecolor="white", linewidth=1.0, zorder=40)
        ax.text(depot["lon"], depot["lat"], depot["name"], color="#92400e", fontsize=9, fontweight="bold", zorder=41)

    if G is None:
        ax.set_title("Part 3 - Zone Division for Shipper Territories", color="#111827", fontsize=13, fontweight="bold")
        ax.tick_params(colors="#6b7280", labelsize=7)
        ax.grid(True, color="#cbd5e1", alpha=0.35, linewidth=0.4)
    out_path = OUTPUT_DIR / "part3_zones.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Static map: {out_path}")

    if folium is None:
        print("  Skipped part3_zones.html: folium is not installed")
        return

    center = [
        float(np.mean([d["lat"] for d in deliveries])),
        float(np.mean([d["lon"] for d in deliveries])),
    ]
    fmap = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron", control_scale=True)

    for zone_id_str, zone in zone_summary.items():
        zone_id = int(zone_id_str)
        color = MAP_COLORS[zone_id % len(MAP_COLORS)]
        group = folium.FeatureGroup(name=f"Zone {zone_id} ({zone['n_deliveries']} orders)", show=True)

        members = zone["delivery_indices"]
        points = [[deliveries[i]["lat"], deliveries[i]["lon"]] for i in members]
        if len(points) >= 3:
            try:
                from shapely.geometry import MultiPoint

                hull = MultiPoint([(lon, lat) for lat, lon in points]).convex_hull
                if hull.geom_type == "Polygon":
                    hull_points = [(lat, lon) for lon, lat in hull.exterior.coords]
                    folium.Polygon(
                        hull_points,
                        color=color,
                        weight=2,
                        fill=True,
                        fill_opacity=0.13,
                        tooltip=f"Zone {zone_id}",
                    ).add_to(group)
            except Exception:
                pass

        folium.CircleMarker(
            [zone["centroid_lat"], zone["centroid_lon"]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Zone {zone_id}: {zone['n_deliveries']} orders | {zone['total_weight_kg']}kg",
        ).add_to(group)

        for idx in members:
            delivery = deliveries[idx]
            folium.CircleMarker(
                [delivery["lat"], delivery["lon"]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.8,
                tooltip=f"{delivery['order_id']} | {delivery['weight_kg']}kg | Zone {zone_id}",
            ).add_to(group)
        group.add_to(fmap)

    for depot in depots:
        folium.Marker(
            [depot["lat"], depot["lon"]],
            tooltip=depot["name"],
            icon=folium.Icon(color="red", icon="home"),
        ).add_to(fmap)

    save_folium_map(fmap, "part3_zones.html")


# =============================================================================
# PART 4 — VRP OPTIMIZATION
# =============================================================================

@dataclass
class Route:
    shipper_id: int
    depot_idx: int
    stops: list[int]
    load_kg: float

    def full_path(self) -> list[int]:
        return [self.depot_idx] + self.stops + [self.depot_idx]


def route_travel_time(path: list[int], time_matrix: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    travel = sum(float(time_matrix[path[i], path[i + 1]]) for i in range(len(path) - 1))
    service = max(0, len(path) - 2) * SERVICE_TIME_MIN
    return travel + service


def nearest_neighbor_route(
    depot_idx: int,
    candidates: list[int],
    order_weights: dict[int, float],
    time_matrix: np.ndarray,
) -> tuple[list[int], list[int], float]:
    remaining = set(candidates)
    route = []
    load = 0.0
    current = depot_idx

    while remaining:
        feasible = [
            stop for stop in remaining
            if load + order_weights[stop] <= SHIPPER_CAPACITY_KG
        ]
        if not feasible:
            break

        best = min(feasible, key=lambda stop: float(time_matrix[current, stop]))
        trial_path = [depot_idx] + route + [best] + [depot_idx]
        if route_travel_time(trial_path, time_matrix) > SHIPPER_MAX_TIME_MIN:
            break

        route.append(best)
        remaining.remove(best)
        load += order_weights[best]
        current = best

    return route, sorted(remaining), load


def two_opt(stops: list[int], depot_idx: int, time_matrix: np.ndarray) -> list[int]:
    if len(stops) < 4:
        return stops

    best = stops[:]
    best_cost = route_travel_time([depot_idx] + best + [depot_idx], time_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cost = route_travel_time([depot_idx] + candidate + [depot_idx], time_matrix)
                if cost + 1e-6 < best_cost:
                    best = candidate
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break
    return best


def part4_vrp_optimization(G: nx.MultiDiGraph | None = None) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("PART 4 — VRP OPTIMIZATION")
    print("=" * 70)

    delivery_data = load_json(OUTPUT_DIR / "delivery_nodes.json")
    zone_data = load_json(OUTPUT_DIR / "zone_assignments.json")
    time_matrix = np.load(OUTPUT_DIR / "time_matrix.npy")
    dist_matrix = np.load(OUTPUT_DIR / "dist_matrix.npy")

    deliveries = delivery_data["deliveries"]
    depots = delivery_data["depots"]

    order_weights = {
        order_idx_to_matrix_index(i): float(deliveries[i]["weight_kg"])
        for i in range(len(deliveries))
    }

    unassigned = set(order_idx_to_matrix_index(i) for i in range(len(deliveries)))
    routes: list[Route] = []

    for zone_id_str, zone in sorted(zone_data.items(), key=lambda item: int(item[0])):
        zone_id = int(zone_id_str)
        depot_name = zone["nearest_depot"]
        depot_idx = next(
            int(d["matrix_index"]) for d in depots if d["name"] == depot_name
        )
        candidates = [
            order_idx_to_matrix_index(i)
            for i in zone["delivery_indices"]
            if order_idx_to_matrix_index(i) in unassigned
        ]

        stops, leftover, load = nearest_neighbor_route(
            depot_idx,
            candidates,
            order_weights,
            time_matrix,
        )
        stops = two_opt(stops, depot_idx, time_matrix)
        for stop in stops:
            unassigned.discard(stop)

        routes.append(Route(zone_id, depot_idx, stops, load))

    # Try to insert unassigned orders into the cheapest feasible route.
    for stop in list(unassigned):
        weight = order_weights[stop]
        best_move = None
        for route_idx, route in enumerate(routes):
            if route.load_kg + weight > SHIPPER_CAPACITY_KG:
                continue
            for pos in range(len(route.stops) + 1):
                candidate_stops = route.stops[:pos] + [stop] + route.stops[pos:]
                candidate_path = [route.depot_idx] + candidate_stops + [route.depot_idx]
                candidate_time = route_travel_time(candidate_path, time_matrix)
                if candidate_time > SHIPPER_MAX_TIME_MIN:
                    continue
                old_time = route_travel_time(route.full_path(), time_matrix)
                delta = candidate_time - old_time
                if best_move is None or delta < best_move[0]:
                    best_move = (delta, route_idx, pos)
        if best_move:
            _, route_idx, pos = best_move
            routes[route_idx].stops.insert(pos, stop)
            routes[route_idx].load_kg += weight
            unassigned.discard(stop)

    schedule = []
    for offset, route in enumerate(sorted(routes, key=lambda r: len(r.stops), reverse=True)):
        path = route.full_path()
        travel_time = route_travel_time(path, time_matrix)
        travel_dist = sum(float(dist_matrix[path[i], path[i + 1]]) for i in range(len(path) - 1))
        schedule.append(
            {
                "shipper_id": route.shipper_id,
                "depot_idx": route.depot_idx,
                "depot_name": depots[route.depot_idx]["name"],
                "departure": f"{WORKING_START_HOUR:02d}:{offset * 5:02d}",
                "n_stops": len(route.stops),
                "load_kg": round(route.load_kg, 1),
                "total_time_min": round(travel_time, 1),
                "total_distance_km": round(travel_dist, 2),
                "feasible": route.load_kg <= SHIPPER_CAPACITY_KG and travel_time <= SHIPPER_MAX_TIME_MIN,
                "route_stops": path,
                "order_ids": [
                    deliveries[matrix_index_to_order_idx(stop)]["order_id"]
                    for stop in route.stops
                ],
            }
        )

    solution = {
        "metadata": {
            "algorithm": "Zone assignment + nearest neighbor + 2-opt + cheapest insertion",
            "n_depots": N_DEPOTS,
            "n_shippers": len(routes),
            "n_orders": len(deliveries),
            "assigned_orders": len(deliveries) - len(unassigned),
            "unassigned_orders": len(unassigned),
            "total_time_min": round(sum(s["total_time_min"] for s in schedule), 1),
            "total_distance_km": round(sum(s["total_distance_km"] for s in schedule), 2),
            "shipper_capacity_kg": SHIPPER_CAPACITY_KG,
            "shipper_max_time_min": SHIPPER_MAX_TIME_MIN,
        },
        "schedule": schedule,
        "unassigned_orders": [
            {
                "order_index": matrix_index_to_order_idx(stop),
                "order_id": deliveries[matrix_index_to_order_idx(stop)]["order_id"],
            }
            for stop in sorted(unassigned)
        ],
    }
    save_json(OUTPUT_DIR / "vrp_solution.json", solution)
    if G is not None:
        visualize_part4_vrp_routes(G, solution, delivery_data)
    print(
        f"Assigned {solution['metadata']['assigned_orders']}/{len(deliveries)} orders | "
        f"unassigned={len(unassigned)}"
    )
    return solution


def matrix_route_to_graph_route(G: nx.MultiDiGraph, matrix_route: list[int]) -> list[str]:
    """Convert a depot/order matrix route into a continuous graph-node route."""
    graph_route: list[str] = []
    for src_idx, dst_idx in zip(matrix_route[:-1], matrix_route[1:]):
        src = graph_node_from_matrix_index(src_idx)
        dst = graph_node_from_matrix_index(dst_idx)
        segment = shortest_node_route(G, src, dst)
        if not segment:
            continue
        if graph_route:
            segment = segment[1:]
        graph_route.extend(segment)
    return graph_route


def visualize_part4_vrp_routes(
    G: nx.MultiDiGraph,
    solution: dict[str, Any],
    delivery_data: dict[str, Any],
) -> None:
    """Visualize final optimized VRP routes as static and interactive maps."""
    print("\n[VIS] Part 4 VRP routes...")

    depots = delivery_data["depots"]
    deliveries = delivery_data["deliveries"]
    route_paths = []
    for route in solution["schedule"]:
        graph_route = matrix_route_to_graph_route(G, route["route_stops"])
        latlon = route_latlon_path(G, graph_route) if graph_route else []
        route_paths.append((route, graph_route, latlon))

    fig, ax = plt.subplots(figsize=(15, 15), dpi=180, facecolor="#f4efe7")
    plot_part1_detailed_base(
        ax,
        G,
        "Part 4 - Optimized VRP Routes on Part 1 Map",
    )

    for route, _, latlon in route_paths:
        color = MAP_COLORS[int(route["shipper_id"]) % len(MAP_COLORS)]
        if len(latlon) >= 2:
            lats = [p[0] for p in latlon]
            lons = [p[1] for p in latlon]
            ax.plot(lons, lats, color=color, linewidth=2.0, alpha=0.82, zorder=25)

        stop_orders = [idx for idx in route["route_stops"][1:-1] if idx >= N_DEPOTS]
        ax.scatter(
            [deliveries[matrix_index_to_order_idx(idx)]["lon"] for idx in stop_orders],
            [deliveries[matrix_index_to_order_idx(idx)]["lat"] for idx in stop_orders],
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=30,
        )

    for depot in depots:
        ax.scatter(depot["lon"], depot["lat"], s=160, marker="s", color="#111827", edgecolor="white", zorder=40)
        ax.text(depot["lon"], depot["lat"], depot["name"], fontsize=8, color="#111827", fontweight="bold", zorder=41)

    out_path = OUTPUT_DIR / "part4_vrp_routes.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Static map: {out_path}")

    fmap = make_base_folium_map(G, "Part 4 - Optimized VRP Routes")
    if fmap is not None:
        for route, _, latlon in route_paths:
            shipper_id = int(route["shipper_id"])
            color = MAP_COLORS[shipper_id % len(MAP_COLORS)]
            group = folium.FeatureGroup(
                name=f"Shipper {shipper_id}: {route['n_stops']} stops, {route['total_time_min']} min",
                show=True,
            )
            if len(latlon) >= 2:
                folium.PolyLine(
                    latlon,
                    color=color,
                    weight=4,
                    opacity=0.82,
                    tooltip=(
                        f"Shipper {shipper_id} | {route['n_stops']} stops | "
                        f"{route['total_time_min']} min | {route['total_distance_km']} km"
                    ),
                ).add_to(group)

            depot = depots[int(route["depot_idx"])]
            folium.Marker(
                [depot["lat"], depot["lon"]],
                tooltip=f"Start/End: {depot['name']}",
                icon=folium.Icon(color="red", icon="home"),
            ).add_to(group)

            for seq, matrix_idx in enumerate(route["route_stops"][1:-1], start=1):
                if matrix_idx < N_DEPOTS:
                    continue
                order = deliveries[matrix_index_to_order_idx(matrix_idx)]
                folium.CircleMarker(
                    [order["lat"], order["lon"]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_opacity=0.88,
                    tooltip=f"#{seq} {order['order_id']} | {order['weight_kg']}kg | S{shipper_id}",
                ).add_to(group)
            group.add_to(fmap)

        unassigned_group = folium.FeatureGroup(name="Unassigned orders", show=True)
        for item in solution["unassigned_orders"]:
            order = deliveries[int(item["order_index"])]
            folium.CircleMarker(
                [order["lat"], order["lon"]],
                radius=5,
                color="#111827",
                fill=True,
                fill_color="#ef4444",
                fill_opacity=0.95,
                tooltip=f"Unassigned: {order['order_id']}",
            ).add_to(unassigned_group)
        unassigned_group.add_to(fmap)

    save_folium_map(fmap, "part4_vrp_routes.html")


# =============================================================================
# PART 5 — SIMULATION & DASHBOARD
# =============================================================================

def parse_departure_minutes(value: str) -> float:
    hour, minute = value.split(":")
    return (int(hour) - WORKING_START_HOUR) * 60 + int(minute)


def part5_simulation_dashboard() -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("PART 5 — SIMULATION & DASHBOARD")
    print("=" * 70)

    solution = load_json(OUTPUT_DIR / "vrp_solution.json")
    delivery_data = load_json(OUTPUT_DIR / "delivery_nodes.json")
    time_matrix = np.load(OUTPUT_DIR / "time_matrix.npy")
    deliveries = delivery_data["deliveries"]

    events = []
    delivered = 0

    for route in solution["schedule"]:
        shipper_id = route["shipper_id"]
        path = route["route_stops"]
        current_time = parse_departure_minutes(route["departure"])

        events.append(
            {
                "time_min": round(current_time, 1),
                "clock": minutes_to_clock(current_time),
                "type": "DEPART",
                "shipper_id": shipper_id,
                "location": route["depot_name"],
            }
        )

        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            travel = float(time_matrix[src, dst])
            current_time += travel

            if dst >= N_DEPOTS:
                order_idx = matrix_index_to_order_idx(dst)
                delivered += 1
                events.append(
                    {
                        "time_min": round(current_time, 1),
                        "clock": minutes_to_clock(current_time),
                        "type": "ARRIVE_ORDER",
                        "shipper_id": shipper_id,
                        "order_id": deliveries[order_idx]["order_id"],
                    }
                )
                current_time += SERVICE_TIME_MIN
                events.append(
                    {
                        "time_min": round(current_time, 1),
                        "clock": minutes_to_clock(current_time),
                        "type": "DELIVERED",
                        "shipper_id": shipper_id,
                        "order_id": deliveries[order_idx]["order_id"],
                    }
                )
            else:
                events.append(
                    {
                        "time_min": round(current_time, 1),
                        "clock": minutes_to_clock(current_time),
                        "type": "RETURN_DEPOT",
                        "shipper_id": shipper_id,
                        "location": route["depot_name"],
                    }
                )

    events.sort(key=lambda event: event["time_min"])
    summary = {
        "total_events": len(events),
        "delivered_orders": delivered,
        "unassigned_orders": len(solution["unassigned_orders"]),
        "total_time_min": solution["metadata"]["total_time_min"],
        "total_distance_km": solution["metadata"]["total_distance_km"],
        "active_shippers": len(solution["schedule"]),
    }

    save_json(OUTPUT_DIR / "simulation_events.json", events)
    save_json(OUTPUT_DIR / "simulation_summary.json", summary)
    save_dashboard_html(solution, summary, events)
    print(f"Saved simulation with {len(events)} events")
    return summary


def minutes_to_clock(minutes_from_start: float) -> str:
    total = WORKING_START_HOUR * 60 + int(round(minutes_from_start))
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def save_dashboard_html(solution: dict[str, Any], summary: dict[str, Any], events: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"""
        <tr>
          <td>S{r['shipper_id']}</td>
          <td>{r['depot_name']}</td>
          <td>{r['departure']}</td>
          <td>{r['n_stops']}</td>
          <td>{r['load_kg']}</td>
          <td>{r['total_time_min']}</td>
          <td>{r['total_distance_km']}</td>
          <td>{'OK' if r['feasible'] else 'Check'}</td>
        </tr>
        """
        for r in solution["schedule"]
    )
    event_rows = "\n".join(
        f"<tr><td>{e['clock']}</td><td>{e['type']}</td><td>S{e['shipper_id']}</td><td>{e.get('order_id', e.get('location', ''))}</td></tr>"
        for e in events[:300]
    )

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dong Da Logistics Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #111827; }}
    header {{ padding: 18px 24px; background: #111827; color: white; }}
    main {{ padding: 20px 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 20px; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }}
    th {{ background: #f3f4f6; }}
    iframe {{ width: 100%; height: 620px; border: 1px solid #d1d5db; border-radius: 8px; background: white; }}
    h2 {{ margin-top: 26px; }}
  </style>
</head>
<body>
  <header>
    <h1>Dong Da Logistics Dashboard</h1>
  </header>
  <main>
    <section class="grid">
      <div class="card">Delivered<div class="metric">{summary['delivered_orders']}</div></div>
      <div class="card">Unassigned<div class="metric">{summary['unassigned_orders']}</div></div>
      <div class="card">Shippers<div class="metric">{summary['active_shippers']}</div></div>
      <div class="card">Total Time<div class="metric">{summary['total_time_min']} min</div></div>
      <div class="card">Distance<div class="metric">{summary['total_distance_km']} km</div></div>
    </section>

    <h2>Interactive Map</h2>
    <iframe src="part4_vrp_routes.html"></iframe>

    <h2>Shipper Schedule</h2>
    <table>
      <thead>
        <tr>
          <th>Shipper</th><th>Depot</th><th>Depart</th><th>Stops</th>
          <th>Load kg</th><th>Time min</th><th>Distance km</th><th>Status</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>

    <h2>Simulation Events</h2>
    <table>
      <thead><tr><th>Time</th><th>Event</th><th>Shipper</th><th>Target</th></tr></thead>
      <tbody>{event_rows}</tbody>
    </table>
  </main>
</body>
</html>
"""
    (OUTPUT_DIR / "dashboard.html").write_text(html, encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_output_dir()

    print("=" * 70)
    print("CITY LOGISTICS & ROUTING SYSTEM — COMPLETE PIPELINE")
    print("Dong Da District, Hanoi, Vietnam")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PART 1 — BUILD OPTIMIZED ROAD GRAPH + BASE MAP")
    print("=" * 70)
    G = build_routing_graph()
    G = add_logistics_weights(G)
    save_feature_layers()
    export_outputs(G)
    save_static_map(G)
    save_interactive_map(G)
    print_summary(G)

    print(f"\nGraph ready: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    part2_shortest_routes(G)
    part3_zone_division(G)
    part4_vrp_optimization(G)
    part5_simulation_dashboard()

    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE FINISHED")
    print("=" * 70)
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
    print("Open dashboard: dong_da_output/dashboard.html")


if __name__ == "__main__":
    main()
