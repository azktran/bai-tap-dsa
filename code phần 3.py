
import json
import math
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any
import math
import folium
from sklearn.preprocessing import StandardScaler

import networkx as nx
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt

try:
    import folium
except ImportError:
    folium = None


BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

OUTPUT_DIR = Path("./dong_da_output")
GRAPH_PATH = OUTPUT_DIR / "dong_da_graph.graphml"

N_DEPOTS = 3
N_DELIVERIES = 100
N_SHIPPERS = 8
RANDOM_SEED = 42

SHIPPER_CAPACITY_KG = 60.0
SHIPPER_MAX_TIME_MIN = 300.0
SERVICE_TIME_MIN = 3.0
WORKING_START_HOUR = 8
DEPOT_CAPACITY = N_DELIVERIES // N_DEPOTS + 1
TW_LATE_PENALTY_FACTOR = 3.0
N_WORKERS = min(cpu_count(), 8)

UNREACHABLE = 9999.0

MAP_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#ff6b6b", "#4ecdc4",
]

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

def plot_part1_detailed_base(
    ax: Any, G: nx.MultiDiGraph, title: str, show_title: bool = True
) -> None:
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    ax.set_facecolor("#f4efe7")
    bounds = nodes_gdf.total_bounds
    margin = 0.004
    ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
    ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

    for filename, color, edge, alpha, zorder in [
        ("dong_da_parks.geojson",     "#c8e6c9", "#a5d6a7", 0.85, 1),
        ("dong_da_water.geojson",     "#b3d9ff", "#90c4e8", 0.92, 2),
        ("dong_da_buildings.geojson", "#d6cfc8", "#bdb5ad", 0.72, 3),
    ]:
        layer = load_layer_geojson(filename)
        if layer is not None and len(layer) > 0:
            layer.plot(
                ax=ax, color=color, edgecolor=edge,
                linewidth=0.08 if "buildings" in filename else 0.25,
                alpha=alpha, zorder=zorder,
            )

    style = {
        "motorway":      ("#e67e22", "#f39c12", 3.2, 2.0, 9, 1.0),
        "trunk":         ("#e67e22", "#f39c12", 3.0, 1.9, 9, 1.0),
        "primary":       ("#d97706", "#fbbf24", 2.6, 1.6, 8, 1.0),
        "secondary":     ("#9ca3af", "#ffffff", 2.0, 1.2, 7, 1.0),
        "tertiary":      ("#9ca3af", "#ffffff", 1.5, 0.9, 6, 0.95),
        "residential":   ("#a3a3a3", "#ffffff", 1.1, 0.7, 5, 0.9),
        "service":       ("#bdbdbd", "#eeeeee", 0.8, 0.45, 4, 0.82),
        "living_street": ("#bdbdbd", "#f3f4f6", 0.8, 0.45, 4, 0.82),
        "unclassified":  ("#bdbdbd", "#eeeeee", 0.8, 0.45, 4, 0.82),
        "road":          ("#a3a3a3", "#ffffff", 1.0, 0.65, 5, 0.9),
    }
    highway_col = edges_gdf["highway"].map(lambda v: str(v).lower())
    render_order = [
        "service", "living_street", "unclassified", "road", "residential",
        "tertiary", "secondary", "primary", "trunk", "motorway",
    ]
    other_types = [t for t in highway_col.unique() if t not in render_order]
    for road_type in other_types + render_order:
        subset = edges_gdf[highway_col == road_type]
        if len(subset) == 0:
            continue
        outline, fill, outline_w, fill_w, zorder, alpha = style.get(
            road_type, ("#bdbdbd", "#dddddd", 0.7, 0.4, 3, 0.75),
        )
        subset.plot(ax=ax, color=outline, linewidth=outline_w, alpha=alpha * 0.6,
                    zorder=zorder, capstyle="round", joinstyle="round")
        subset.plot(ax=ax, color=fill, linewidth=fill_w, alpha=alpha,
                    zorder=zorder + 0.1, capstyle="round", joinstyle="round")

    if show_title:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10,
                     color="#111827")
    ax.set_axis_off()

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

def to_float(value: Any, default: float) -> float:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_graph_numeric_attrs(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    numeric_attrs = [
        "length", "speed_kph", "travel_time", "weight_distance",
        "weight_time", "weight_congestion", "weight_logistics",
        "congestion_factor", "capacity",
    ]
    for _, _, _, data in G.edges(keys=True, data=True):
        for attr in numeric_attrs:
            if attr in data:
                data[attr] = to_float(data[attr], 0.0)
    return G


def load_graph() -> nx.MultiDiGraph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Missing {GRAPH_PATH}. Run part1_optimized.py first.")
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
    
def order_idx_to_matrix_index(order_idx: int) -> int:
    return order_idx + N_DEPOTS

def save_folium_map(fmap: Any, filename: str) -> None:
    if fmap is None:
        print(f"  Skipped {filename}: folium not installed")
        return
    folium.LayerControl(collapsed=False).add_to(fmap)
    path = OUTPUT_DIR / filename
    fmap.save(path)
    print(f"  Interactive map: {path}")



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

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(h))

def initialize(X, k):
    np.random.seed(RANDOM_SEED)
    centroids = []
    centroids.append(X[np.random.randint(X.shape[0])])

    for _ in range(k - 1):
        distances = []
        for point in X:
            min_dist = min([haversine_m(point, c) for c in centroids])
            distances.append(min_dist)
        
        distances = np.array(distances)
        probabilities = distances**2 / np.sum(distances**2)
        
        next_centroid = next_centroid = X[np.random.choice(len(X), p=probabilities)]
        centroids.append(next_centroid)
    
    return np.array(centroids)

def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        
        curr_x = X[idx]
        
        for i in range(N_SHIPPERS):
            dis = haversine_m(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

def assign_zone(X, clusters):
    clust = {i: [] for i in range(N_SHIPPERS)}
    for idx in range(X.shape[0]):
        dist = []
        
        curr_x = X[idx]
        
        for i in range(N_SHIPPERS):
            dis = haversine_m(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clust[curr_cluster].append(idx)
    return clust


def update_clusters(X, clusters):
    for i in range(N_SHIPPERS):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center
    return clusters

def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(N_SHIPPERS):
            dist.append(haversine_m(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred   

def part3_zone_division(G: nx.MultiDiGraph | None = None) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("PART 3 — ZONE DIVISION")
    print("=" * 70)

    data = load_json(OUTPUT_DIR / "delivery_nodes.json")
    deliveries = data["deliveries"]
    time_matrix = np.load(OUTPUT_DIR / "time_matrix.npy")

    coords = [[d["lat"], d["lon"]] for d in deliveries]
    scaler = StandardScaler()
    scoords = scaler.fit_transform(coords)
    clusters = {}
    center = initialize(scoords, N_SHIPPERS)
    for idx in range(N_SHIPPERS):
        points = []
        cluster = {
            'center' : center[idx],
            'points' : []
        }
        
        clusters[idx] = cluster

    clusters = assign_clusters(scoords,clusters)
    clusters = update_clusters(scoords,clusters)
    pred = pred_cluster(scoords,clusters)
    clust = assign_zone(scoords,clusters)
    weights = [float(d["weight_kg"]) for d in deliveries]

    depots = data["depots"]
    zone_summary = {}
    for zone_id, members in clust.items():
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


if __name__ == "__main__":
    ensure_output_dir()
    G = load_graph()
    part3_zone_division(G)
