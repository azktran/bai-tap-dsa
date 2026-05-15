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

