# bai-tap-dsa
bai tap nhom dsa city routing

"""
=============================================================================
CITY LOGISTICS & ROUTING SYSTEM — PART 1: BUILD REALISTIC ROAD GRAPH
=============================================================================
Địa bàn  : Quận Đống Đa, Hà Nội, Việt Nam
Mục tiêu : Xây dựng road graph nền tảng cho shortest path, VRP,
           shipper allocation, congestion simulation, logistics optimization

Libraries:
  • OSMnx     — tải road network + GIS features từ OpenStreetMap
  • NetworkX  — biểu diễn graph, tính toán đường đi, weight
  • GeoPandas — xử lý dữ liệu không gian (GeoDataFrame, geometry)
  • Matplotlib — render bản đồ GIS chất lượng cao
  • Shapely   — geometry operations (tự động qua GeoPandas)

Output:
  • dong_da_graph.graphml        — road graph cho NetworkX algorithms
  • dong_da_nodes.geojson        — nodes (intersections) với coordinates
  • dong_da_edges.geojson        — edges (road segments) với attributes
  • dong_da_gis_map.png          — bản đồ GIS high-resolution (300 DPI)
=============================================================================
"""

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================

import os
import time
import warnings
warnings.filterwarnings("ignore")          # suppress minor deprecation noise

import numpy as np
import osmnx as ox                          # OpenStreetMap network downloader
import networkx as nx                       # graph data structure & algorithms
import geopandas as gpd                     # GeoDataFrame = spatial DataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from shapely.geometry import box

print("=" * 70)
print("  CITY LOGISTICS & ROUTING — PART 1: REALISTIC ROAD GRAPH")
print("  Dong Da District, Hanoi, Vietnam")
print("=" * 70)


# =============================================================================
# 2. OSMNX CONFIGURATION
# =============================================================================
# OSMnx settings: tăng timeout vì Đống Đa là quận đô thị dày đặc,
# response từ Overpass API sẽ lớn (~5–15 MB raw XML)

ox.settings.log_console = True
ox.settings.use_cache = True               # cache queries → tái sử dụng nhanh
ox.settings.cache_folder = "./osm_cache"   # thư mục lưu cache
ox.settings.timeout = 300                  # 5 phút cho Overpass API
ox.settings.max_query_area_size = 50_000_000_000  # cho phép vùng lớn hơn

PLACE = "Đống Đa, Hà Nội, Việt Nam"

# Output paths
OUTPUT_DIR = "./dong_da_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./osm_cache", exist_ok=True)


# =============================================================================
# 3. DOWNLOAD ROAD NETWORK — TẠI SAO simplify=False?
# =============================================================================
"""
GIẢI THÍCH: simplify=True (default) gộp các node trung gian trên đường thẳng
thành 1 edge duy nhất → đơn giản hơn nhưng MẤT thông tin:
  • Mất các điểm uốn (curve points) của đường cong
  • Mất intermediate intersections nhỏ
  • Mất geometry chi tiết của ngõ hẻm, đường cong

simplify=False GIỮ NGUYÊN tất cả nodes và edges từ OSM:
  • Mỗi node là 1 điểm GPS thực tế trên bản đồ
  • Mỗi edge là 1 đoạn đường giữa 2 nodes liên tiếp
  • Full geometry: đường cong thật sự là đường cong trong graph
  → Quan trọng cho: GIS rendering, congestion modeling, distance accuracy

network_type="all" bao gồm:
  • "drive"       — đường ô tô/xe máy
  • "bike"        — đường xe đạp
  • "walk"        — đường đi bộ (ngõ hẻm, vỉa hè)
  → Cho logistics: shipper đi xe máy dùng TẤT CẢ loại đường
"""

print("\n[1/6] Downloading road network from OpenStreetMap...")
print(f"      Place: {PLACE}")
print("      network_type=all, simplify=False, retain_all=True")
print("      (This may take 30–90 seconds on first run...)\n")

start = time.time()

G = ox.graph_from_place(
    PLACE,
    network_type="all",          # tất cả loại đường (drive + bike + walk)
    simplify=False,              # giữ nguyên toàn bộ geometry chi tiết
    retain_all=True,             # giữ cả các components không kết nối
    truncate_by_edge=True,       # cắt theo boundary (không mất edge)
)

elapsed = time.time() - start
print(f"\n✓ Network downloaded in {elapsed:.1f}s")
print(f"  Nodes (intersections/points): {G.number_of_nodes():,}")
print(f"  Edges (road segments):        {G.number_of_edges():,}")
print(f"  Is directed graph:            {G.is_directed()}")


# =============================================================================
# 4. NODE & EDGE LÀ GÌ TRONG ROAD GRAPH?
# =============================================================================
"""
NODE (đỉnh):
  • Mỗi node = 1 điểm GPS (latitude, longitude) trên bản đồ
  • Có thể là: giao lộ, điểm uốn đường, đầu ngõ, điểm cuối
  • Attributes: osmid, y (lat), x (lon), street_count, ref, highway
  • Trong logistics: node = điểm giao hàng, kho, depot, intersection

EDGE (cạnh):
  • Mỗi edge = 1 đoạn đường nối 2 nodes
  • Trong directed graph: edge có HƯỚNG (u → v)
  • Attributes: length (m), speed (km/h), travel_time (s), highway,
               oneway, name, geometry, lanes, maxspeed, access
  • Trong logistics: edge = đoạn đường shipper phải đi qua

DIRECTED GRAPH — TẠI SAO QUAN TRỌNG CHO LOGISTICS?
  • Đường một chiều (oneway=True): chỉ đi được 1 hướng
  • Khác nhau về tắc đường theo giờ: edge u→v khác v→u
  • VRP cần biết hướng đi để tính route hợp lệ
  • Shipper không thể đi ngược chiều → vi phạm traffic law
  → MultiDiGraph: cho phép nhiều edges giữa cùng 2 nodes (song song)
    (vd: 2 làn đường cùng chiều nhưng khác tốc độ/loại)
"""

# Sample node info
sample_node = list(G.nodes(data=True))[0]
print(f"\n  Sample Node #{sample_node[0]}:")
for k, v in list(sample_node[1].items())[:6]:
    print(f"    {k}: {v}")

# Sample edge info
sample_edge = list(G.edges(data=True))[0]
print(f"\n  Sample Edge ({sample_edge[0]} → {sample_edge[1]}):")
for k, v in list(sample_edge[2].items())[:8]:
    print(f"    {k}: {v}")


# =============================================================================
# 5. ADD WEIGHTS — WEIGHTED DIRECTED GRAPH CHO LOGISTICS
# =============================================================================
"""
WEIGHTED DIRECTED GRAPH:
Mỗi edge cần được gán trọng số (weight) phản ánh "chi phí" di chuyển.
Trong logistics, có nhiều loại weight:

  1. length (m)         → minimize total distance (fuel cost)
  2. travel_time (s)    → minimize delivery time (SLA)
  3. congestion_weight  → penalize congested roads (realistic routing)
  4. speed_weight       → inverse of speed (slower road = higher weight)

OSMnx có built-in: add_edge_speeds() + add_edge_travel_times()
  • add_edge_speeds(): điền maxspeed dựa trên highway type nếu thiếu
  • add_edge_travel_times(): tính travel_time = length / speed
"""

print("\n[2/6] Adding edge weights for logistics routing...")

# Thêm tốc độ dựa trên loại đường (nếu maxspeed không có trong OSM)
# Hà Nội speed limits theo highway type:
heuristic_speeds = {
    "motorway": 80,
    "trunk": 60,
    "primary": 50,
    "secondary": 40,
    "tertiary": 30,
    "residential": 25,
    "service": 15,
    "unclassified": 20,
    "living_street": 10,
    "pedestrian": 5,
    "footway": 5,
    "path": 8,
    "cycleway": 12,
    "track": 15,
    "road": 25,
}

G = ox.add_edge_speeds(G,hwy_speeds=heuristic_speeds)
G = ox.add_edge_travel_times(G)

# Thêm custom weights cho từng use case
for u, v, k, data in G.edges(data=True, keys=True):

    length   = data.get("length", 50)            # meters
    speed    = data.get("speed_kph", 20)         # km/h
    travel_t = data.get("travel_time", 10)       # seconds
    highway  = data.get("highway", "unclassified")
    if isinstance(highway, list):
        highway = highway[0]
    oneway   = data.get("oneway", False)
    lanes    = data.get("lanes", 1)
    if isinstance(lanes, list):
        try:
            lanes = int(lanes[0])
        except (ValueError, TypeError):
            lanes = 1
    else:
        try:
            lanes = int(lanes)
        except (ValueError, TypeError):
            lanes = 1

    # --- Weight 1: distance weight (chuẩn hóa về km) ---
    G[u][v][k]["weight_distance"] = length / 1000.0

    # --- Weight 2: time weight (phút) ---
    G[u][v][k]["weight_time"] = travel_t / 60.0

    # --- Weight 3: congestion penalty ---
    # Mô phỏng tắc đường giờ cao điểm Hà Nội (07:00–09:00, 17:00–19:00)
    # Primary/Secondary roads hay tắc hơn residential/service
    congestion_factor = {
        "motorway": 1.2,
        "trunk": 1.4,
        "primary": 1.8,      # Nguyễn Trãi, Láng, Tây Sơn rất tắc
        "secondary": 1.6,
        "tertiary": 1.3,
        "residential": 1.1,
        "service": 1.0,
        "living_street": 1.0,
        "unclassified": 1.1,
    }.get(highway, 1.1)

    G[u][v][k]["congestion_factor"] = congestion_factor
    G[u][v][k]["weight_congestion"] = (travel_t / 60.0) * congestion_factor

    # --- Weight 4: logistics composite weight ---
    # Kết hợp: thời gian + phí xăng (distance) + penalty tắc
    fuel_cost_per_km = 2000  # VND/km (ước tính xe máy)
    time_value = 500         # VND/phút (opportunity cost)

    G[u][v][k]["weight_logistics"] = (
        (travel_t / 60.0) * time_value * congestion_factor
        + (length / 1000.0) * fuel_cost_per_km
    )

    # --- Road capacity (lanes × speed proxy) ---
    G[u][v][k]["capacity"] = lanes * speed

print(f"✓ Edge weights added")
print(f"  Weights: weight_distance, weight_time, weight_congestion, weight_logistics")


# =============================================================================
# 6. CONVERT TO GEODATAFRAMES — graph_to_gdfs()
# =============================================================================
"""
graph_to_gdfs(): chuyển OSMnx MultiDiGraph → 2 GeoDataFrames

  nodes_gdf: GeoDataFrame với geometry = Point(lon, lat)
    • Mỗi row = 1 intersection/point
    • CRS: EPSG:4326 (WGS84 — lat/lon toàn cầu)

  edges_gdf: GeoDataFrame với geometry = LineString([points...])
    • Mỗi row = 1 road segment
    • Có full geometry: đường cong = list các điểm GPS
    • Index = (u, v, key) — directed edge identifier

TẠI SAO CẦN GeoDataFrame?
  • Spatial operations: buffer, intersection, union, sjoin
  • Export GeoJSON: chuẩn cho web mapping (Leaflet, Mapbox)
  • Matplotlib plotting: GeoDataFrame.plot() tự xử lý CRS + geometry
  • Spatial queries: tìm shipper gần nhất, depot trong bán kính X km
"""

print("\n[3/6] Converting graph to GeoDataFrames...")

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

print(f"✓ GeoDataFrames created")
print(f"  nodes_gdf: {len(nodes_gdf):,} rows | CRS: {nodes_gdf.crs}")
print(f"  edges_gdf: {len(edges_gdf):,} rows | CRS: {edges_gdf.crs}")
print(f"  Columns in edges_gdf: {list(edges_gdf.columns[:10])}...")


# =============================================================================
# 7. DOWNLOAD GIS FEATURE LAYERS — features_from_place()
# =============================================================================
"""
features_from_place() tải các feature polygon/point từ OSM tags:
  • building    → building footprints (polygons)
  • landuse     → công viên, trường học, bệnh viện...
  • natural     → hồ nước, sông, cây xanh
  • leisure     → công viên, sân thể thao

TẠI SAO GIS RENDERING KHÁC GRAPH VISUALIZATION?

  Graph Visualization (NetworkX default):
    • Chỉ vẽ nodes + edges như đồ thị toán học
    • Không có không gian địa lý thực
    • Spring layout / circular layout → không phản ánh thực tế
    → CHỈ dùng để debug algorithm, visualize connectivity

  GIS Rendering (GeoPandas + Matplotlib / QGIS style):
    • Nodes/edges có CRS thực (EPSG:4326 hoặc projected)
    • Vẽ theo tọa độ GPS thực → đúng vị trí địa lý
    • Nhiều layers chồng lên nhau (buildings, parks, water, roads)
    • Road width = thực tế (primary rộng hơn residential)
    → DÙNG cho bản đồ hiển thị thực tế, logistics dashboard
"""

print("\n[4/6] Downloading GIS feature layers from OpenStreetMap...")

def safe_download_features(place, tags, layer_name):
    """Tải features với error handling — một số tags có thể không có data."""
    try:
        gdf = ox.features_from_place(place, tags=tags)
        # Chỉ giữ polygon features (bỏ points/lines cho layers này)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        print(f"  ✓ {layer_name}: {len(gdf):,} features")
        return gdf
    except Exception as e:
        print(f"  ⚠ {layer_name}: {e} — skipping")
        return None

# --- Buildings ---
buildings_gdf = safe_download_features(
    PLACE,
    tags={"building": True},
    layer_name="Buildings"
)

# --- Parks & Green spaces ---
parks_gdf = safe_download_features(
    PLACE,
    tags={"leisure": ["park", "garden", "recreation_ground"],
          "landuse": ["grass", "meadow", "recreation_ground", "village_green"]},
    layer_name="Parks/Green"
)

# --- Water bodies (hồ, sông) ---
water_gdf = safe_download_features(
    PLACE,
    tags={"natural": ["water", "wetland"],
          "waterway": ["riverbank"],
          "landuse": ["reservoir", "basin"]},
    layer_name="Water"
)

# --- Landuse (trường học, bệnh viện, commercial) ---
landuse_gdf = safe_download_features(
    PLACE,
    tags={"landuse": ["residential", "commercial", "industrial",
                      "retail", "education", "institutional"]},
    layer_name="Landuse"
)

# --- Amenity areas (hospitals, universities) ---
amenity_gdf = safe_download_features(
    PLACE,
    tags={"amenity": ["university", "hospital", "school", "college"]},
    layer_name="Amenity areas"
)

print("✓ GIS layers downloaded")


# =============================================================================
# 8. ROAD HIERARCHY — PHÂN CẤP ĐƯỜNG
# =============================================================================
"""
ROAD HIERARCHY TRONG ĐÔ THỊ:

Cấp 1 — Primary/Trunk (đường chính):
  • Rộng 4–8 làn, phân cách giữa
  • Ví dụ: Nguyễn Trãi, Láng, Tây Sơn, Hoàng Cầu
  • Line width: 3.5–5.0px | Color: #F5A623 (vàng cam)

Cấp 2 — Secondary (đường phụ chính):
  • Rộng 2–4 làn
  • Ví dụ: Đặng Tiến Đông, Thái Thịnh, Xã Đàn
  • Line width: 2.5–3.5px | Color: #FDD835 (vàng)

Cấp 3 — Tertiary (đường phụ):
  • Rộng 1–2 làn
  • Nhiều đường nội khu
  • Line width: 1.8–2.5px | Color: #FFFFFF (trắng)

Cấp 4 — Residential (đường dân sinh):
  • Đường phố nhỏ trong khu dân cư
  • Line width: 1.0–1.8px | Color: #FFFFFF

Cấp 5 — Service/Living street (ngõ nhỏ):
  • Ngõ hẻm, đường dịch vụ
  • Line width: 0.5–1.0px | Color: #E0E0E0

Cấp 6 — Footway/Path/Cycleway:
  • Đường đi bộ, xe đạp
  • Line width: 0.3–0.5px | Color: #AAAAAA (nét mờ)
"""

# Road styling config — giống Google Maps wireframe / QGIS style
ROAD_STYLE = {
    # (color_outline, color_fill, linewidth_outline, linewidth_fill, zorder, alpha)
    "motorway":      ("#E67E22", "#F39C12", 2.8, 2.0, 9, 1.0),
    "trunk":         ("#E67E22", "#F39C12", 2.5, 1.8, 9, 1.0),
    "primary":       ("#E67E22", "#FBBF24", 2.2, 1.5, 8, 1.0),
    "secondary":     ("#9CA3AF", "#FFFFFF", 1.8, 1.2, 7, 1.0),
    "tertiary":      ("#9CA3AF", "#FFFFFF", 1.4, 0.9, 6, 0.95),
    "residential":   ("#9CA3AF", "#FFFFFF", 1.0, 0.7, 5, 0.9),
    "living_street": ("#BDBDBD", "#F5F5F5", 0.8, 0.5, 4, 0.85),
    "service":       ("#BDBDBD", "#F0F0F0", 0.6, 0.4, 4, 0.8),
    "unclassified":  ("#BDBDBD", "#EEEEEE", 0.7, 0.5, 4, 0.8),
    "footway":       ("#C0C0C0", "#DDDDDD", 0.4, 0.3, 3, 0.6),
    "path":          ("#C0C0C0", "#DDDDDD", 0.4, 0.3, 3, 0.6),
    "cycleway":      ("#5BC0DE", "#87CEEB", 0.5, 0.3, 3, 0.7),
    "pedestrian":    ("#C0C0C0", "#EEEEEE", 0.5, 0.3, 3, 0.6),
    "steps":         ("#C0C0C0", "#DDDDDD", 0.3, 0.2, 3, 0.5),
    "track":         ("#B0B0B0", "#D0D0D0", 0.4, 0.3, 3, 0.6),
    "road":          ("#9CA3AF", "#FFFFFF", 1.0, 0.7, 5, 0.9),
    "_default":      ("#AAAAAA", "#DDDDDD", 0.5, 0.3, 3, 0.7),
}

def get_road_style(highway_value):
    """Trả về style cho road type, xử lý list values từ OSM."""
    if isinstance(highway_value, list):
        highway_value = highway_value[0]
    if isinstance(highway_value, str):
        # Xử lý "motorway_link" → "motorway"
        for key in ROAD_STYLE:
            if key != "_default" and highway_value.startswith(key):
                return ROAD_STYLE[key]
    return ROAD_STYLE["_default"]


# =============================================================================
# 9. GIS RENDERING — PRODUCTION QUALITY MAP
# =============================================================================
"""
Rendering strategy (từ dưới lên — back to front):
  Layer 0: Background (màu đất/nền thành phố)
  Layer 1: Landuse polygons (commercial, residential zones)
  Layer 2: Park/green polygons (màu xanh lá)
  Layer 3: Water polygons (màu xanh dương)
  Layer 4: Building footprints (màu xám nhạt)
  Layer 5: Road casings (viền ngoài — màu tối hơn, dày hơn)
  Layer 6: Road fills (màu đường thực — trắng/vàng theo hierarchy)
  Layer 7: Labels & Legend
"""

print("\n[5/6] Rendering GIS map (production quality)...")
print("      Layers: background → landuse → parks → water → buildings → roads")

# --- Figure setup ---
fig, ax = plt.subplots(1, 1, figsize=(20, 22), dpi=300,
                        facecolor="#F0EBE3")  # Warm paper background
ax.set_facecolor("#F0EBE3")

# Lấy bounding box của district
bbox = nodes_gdf.total_bounds   # [minx, miny, maxx, maxy]
margin = 0.005
ax.set_xlim(bbox[0] - margin, bbox[2] + margin)
ax.set_ylim(bbox[1] - margin, bbox[3] + margin)

# =============================================================================
# LAYER 1: LANDUSE
# =============================================================================
if landuse_gdf is not None and len(landuse_gdf) > 0:
    landuse_colors = {
        "residential": "#E8E0D8",
        "commercial":  "#F7E6C4",
        "retail":      "#FDEBD0",
        "industrial":  "#E0D8D0",
        "education":   "#DFE6E0",
        "institutional":"#DFE6E0",
    }
    for lu_type, color in landuse_colors.items():
        subset = landuse_gdf[
            landuse_gdf.get("landuse", gpd.pd.Series()).astype(str) == lu_type
        ] if "landuse" in landuse_gdf.columns else gpd.GeoDataFrame()
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor="none",
                       alpha=0.6, zorder=1)

# =============================================================================
# LAYER 2: PARKS & GREEN SPACES
# =============================================================================
if parks_gdf is not None and len(parks_gdf) > 0:
    parks_gdf.plot(ax=ax,
                   color="#C8E6C9",        # xanh lá nhạt
                   edgecolor="#A5D6A7",
                   linewidth=0.3,
                   alpha=0.85,
                   zorder=2)

# =============================================================================
# LAYER 3: WATER BODIES
# =============================================================================
if water_gdf is not None and len(water_gdf) > 0:
    water_gdf.plot(ax=ax,
                   color="#B3D9FF",        # xanh dương nhạt
                   edgecolor="#90C4E8",
                   linewidth=0.5,
                   alpha=0.9,
                   zorder=3)

# =============================================================================
# LAYER 4: BUILDINGS
# =============================================================================
if buildings_gdf is not None and len(buildings_gdf) > 0:
    buildings_gdf.plot(ax=ax,
                       color="#D6CFC8",    # xám ấm
                       edgecolor="#BDB5AD",
                       linewidth=0.15,
                       alpha=0.75,
                       zorder=4)
    print(f"  ✓ Buildings: {len(buildings_gdf):,} polygons rendered")

# =============================================================================
# LAYER 5 & 6: ROADS — Road Casing + Road Fill
# =============================================================================
# Group edges by highway type để render từng loại với style riêng
highway_col = edges_gdf["highway"].apply(
    lambda x: x[0] if isinstance(x, list) else (x if isinstance(x, str) else "unclassified")
)

# Render theo thứ tự cấp độ (thấp nhất trước, cao nhất sau)
RENDER_ORDER = [
    "steps", "footway", "path", "cycleway", "pedestrian",
    "track", "service", "living_street", "unclassified",
    "residential", "tertiary", "secondary", "primary",
    "trunk", "motorway"
]

# Thêm các loại chưa có trong list
other_types = [t for t in highway_col.unique() if t not in RENDER_ORDER]
render_sequence = other_types + RENDER_ORDER

road_counts = {}
for road_type in render_sequence:
    mask = highway_col == road_type
    subset = edges_gdf[mask]
    if len(subset) == 0:
        continue

    style = get_road_style(road_type)
    color_out, color_fill, lw_out, lw_fill, zorder, alpha = style
    road_counts[road_type] = len(subset)

    # Casing (viền ngoài — tạo hiệu ứng depth)
    subset.plot(ax=ax,
                color=color_out,
                linewidth=lw_out * 1.6,
                alpha=alpha * 0.7,
                zorder=zorder,
                capstyle="round",
                joinstyle="round")

    # Fill (màu đường thực)
    subset.plot(ax=ax,
                color=color_fill,
                linewidth=lw_fill,
                alpha=alpha,
                zorder=zorder + 0.5,
                capstyle="round",
                joinstyle="round")

# In thống kê road types
print("\n  Road type distribution:")
for rt in sorted(road_counts, key=road_counts.get, reverse=True)[:12]:
    bar = "█" * min(int(road_counts[rt] / 100), 40)
    print(f"    {rt:20s}: {road_counts[rt]:5,}  {bar}")

# =============================================================================
# LAYER 7: AMENITIES (trường đại học, bệnh viện)
# =============================================================================
if amenity_gdf is not None and len(amenity_gdf) > 0:
    amenity_gdf.plot(ax=ax,
                     color="#E8F5E9",
                     edgecolor="#66BB6A",
                     linewidth=0.5,
                     alpha=0.6,
                     zorder=5)

# =============================================================================
# MAP DECORATION — Title, Legend, North Arrow, Scale
# =============================================================================

# --- Title ---
ax.set_title(
    "Quận Đống Đa — Road Network & Urban Features\nHà Nội, Việt Nam",
    fontsize=16, fontweight="bold", color="#1A1A2E",
    pad=15, loc="left",
    fontfamily="DejaVu Sans"
)
ax.text(0.99, 1.01,
        "OpenStreetMap Contributors | OSMnx + GeoPandas",
        transform=ax.transAxes,
        fontsize=7, color="#888888", ha="right", va="bottom")

# --- Axes labels ---
ax.set_xlabel("Longitude (°E)", fontsize=9, color="#555555", labelpad=6)
ax.set_ylabel("Latitude (°N)", fontsize=9, color="#555555", labelpad=6)
ax.tick_params(axis="both", which="major", labelsize=7.5, colors="#666666")

# Grid
ax.grid(True, linestyle="--", linewidth=0.3, color="#BBBBBB", alpha=0.5,
        zorder=0)
ax.set_axisbelow(True)

# --- Legend ---
legend_elements = [
    mpatches.Patch(facecolor="#F39C12", edgecolor="#E67E22",
                   label="Primary/Trunk roads"),
    mpatches.Patch(facecolor="#FFFFFF", edgecolor="#9CA3AF",
                   label="Secondary/Tertiary roads"),
    mpatches.Patch(facecolor="#FFFFFF", edgecolor="#BDBDBD",
                   label="Residential/Service/Alleys"),
    Line2D([0], [0], color="#87CEEB", linewidth=1.5,
           label="Cycleways"),
    mpatches.Patch(facecolor="#C8E6C9", edgecolor="#A5D6A7",
                   label="Parks & Green spaces"),
    mpatches.Patch(facecolor="#B3D9FF", edgecolor="#90C4E8",
                   label="Water bodies (hồ/sông)"),
    mpatches.Patch(facecolor="#D6CFC8", edgecolor="#BDB5AD",
                   label="Buildings"),
    mpatches.Patch(facecolor="#F7E6C4", edgecolor="none",
                   label="Commercial areas"),
]

legend = ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=8,
    title="Map Legend",
    title_fontsize=9,
    framealpha=0.92,
    facecolor="#FAFAFA",
    edgecolor="#CCCCCC",
    borderpad=0.8,
    handlelength=1.4,
)
legend.get_title().set_fontweight("bold")
legend.get_title().set_color("#1A1A2E")

# --- Stats box ---
stats_text = (
    f"Graph Statistics\n"
    f"─────────────────\n"
    f"Nodes: {G.number_of_nodes():,}\n"
    f"Edges: {G.number_of_edges():,}\n"
    f"Buildings: {len(buildings_gdf):,}\n" if buildings_gdf is not None else ""
    f"Road types: {len(road_counts)}"
)
ax.text(0.995, 0.995, stats_text,
        transform=ax.transAxes, va="top", ha="right",
        fontsize=7.5, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA",
                  edgecolor="#CCCCCC", alpha=0.92),
        color="#1A1A2E")

# --- Frame ---
for spine in ax.spines.values():
    spine.set_edgecolor("#888888")
    spine.set_linewidth(0.7)

plt.tight_layout(pad=1.0)


# =============================================================================
# 10. SAVE OUTPUTS
# =============================================================================

print("\n[6/6] Saving outputs...")

# --- Save PNG map ---
map_path = os.path.join(OUTPUT_DIR, "dong_da_gis_map.png")
fig.savefig(map_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            format="png")
print(f"  ✓ GIS Map PNG saved: {map_path}")

plt.close(fig)

# --- Save GraphML ---
graphml_path = os.path.join(OUTPUT_DIR, "dong_da_graph.graphml")
ox.save_graphml(G, filepath=graphml_path)
print(f"  ✓ GraphML saved:     {graphml_path}")

# --- Save GeoJSON: Nodes ---
nodes_export = nodes_gdf.copy()
# GeoJSON không hỗ trợ list values — convert to string
for col in nodes_export.columns:
    if nodes_export[col].dtype == object:
        nodes_export[col] = nodes_export[col].astype(str)
nodes_path = os.path.join(OUTPUT_DIR, "dong_da_nodes.geojson")
nodes_export.to_file(nodes_path, driver="GeoJSON")
print(f"  ✓ Nodes GeoJSON:     {nodes_path}")

# --- Save GeoJSON: Edges ---
edges_export = edges_gdf.copy().reset_index()
for col in edges_export.columns:
    if edges_export[col].dtype == object:
        edges_export[col] = edges_export[col].astype(str)
# Giữ các columns quan trọng
keep_cols = [
    "u", "v", "key", "geometry", "highway", "name", "length",
    "oneway", "speed_kph", "travel_time", "lanes", "maxspeed",
    "weight_distance", "weight_time", "weight_congestion",
    "weight_logistics", "congestion_factor", "capacity"
]
keep_cols = [c for c in keep_cols if c in edges_export.columns]
edges_export = edges_export[keep_cols]
edges_path = os.path.join(OUTPUT_DIR, "dong_da_edges.geojson")
edges_export.to_file(edges_path, driver="GeoJSON")
print(f"  ✓ Edges GeoJSON:     {edges_path}")

# --- Save feature layers ---
if buildings_gdf is not None and len(buildings_gdf) > 0:
    bld = buildings_gdf[["geometry"]].copy()
    bld.to_file(os.path.join(OUTPUT_DIR, "dong_da_buildings.geojson"), driver="GeoJSON")
    print(f"  ✓ Buildings GeoJSON: dong_da_buildings.geojson")

if water_gdf is not None and len(water_gdf) > 0:
    w = water_gdf[["geometry"]].copy()
    w.to_file(os.path.join(OUTPUT_DIR, "dong_da_water.geojson"), driver="GeoJSON")
    print(f"  ✓ Water GeoJSON:     dong_da_water.geojson")

if parks_gdf is not None and len(parks_gdf) > 0:
    p = parks_gdf[["geometry"]].copy()
    p.to_file(os.path.join(OUTPUT_DIR, "dong_da_parks.geojson"), driver="GeoJSON")
    print(f"  ✓ Parks GeoJSON:     dong_da_parks.geojson")


# =============================================================================
# 11. SUMMARY & NEXT STEPS
# =============================================================================

print("\n" + "=" * 70)
print("  PART 1 COMPLETE — Road Graph Foundation Built")
print("=" * 70)
print(f"""
GRAPH SUMMARY:
  • Type        : {type(G).__name__} (Directed + Multi-edge)
  • Nodes       : {G.number_of_nodes():,}  (intersections + geometry points)
  • Edges       : {G.number_of_edges():,}  (road segments, directed)
  • simplify    : False  (full curved road geometry retained)
  • CRS         : EPSG:4326 (WGS84)

EDGE WEIGHTS AVAILABLE:
  • weight_distance  — minimize fuel/distance (km)
  • weight_time      — minimize delivery time (minutes)
  • weight_congestion— time × congestion factor (peak hour)
  • weight_logistics — composite: time + fuel cost (VND)

OUTPUT FILES (./dong_da_output/):
  • dong_da_gis_map.png       — High-res GIS map (300 DPI)
  • dong_da_graph.graphml     — Load with: ox.load_graphml(path)
  • dong_da_nodes.geojson     — Intersections with lat/lon
  • dong_da_edges.geojson     — Roads with all weights
  • dong_da_buildings.geojson — Building footprints
  • dong_da_water.geojson     — Water bodies
  • dong_da_parks.geojson     — Parks & green spaces

READY FOR PART 2 — LOGISTICS ALGORITHMS:
  # Load graph
  G = ox.load_graphml("./dong_da_output/dong_da_graph.graphml")

  # Shortest path (by time)
  path = nx.shortest_path(G, source=u, target=v,
                           weight="weight_time")

  # Shortest path (by logistics cost)
  path = nx.shortest_path(G, source=u, target=v,
                           weight="weight_logistics")

  # Nearest node to a GPS coordinate
  node = ox.nearest_nodes(G, X=105.8412, Y=21.0285)

  # Load edges as GeoDataFrame
  _, edges = ox.graph_to_gdfs(G)
""")


# =============================================================================
# 12. BONUS: QUICK CONGESTION ANALYSIS DEMO
# =============================================================================

print("BONUS: Congestion analysis demo...")

# Đường nào có congestion factor cao nhất?
if "congestion_factor" in edges_gdf.columns:
    top_congested = (
        edges_gdf
        .assign(highway_str=highway_col)
        .groupby("highway_str")["congestion_factor"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
    )
    print("\n  Average congestion factor by road type:")
    for rt, cf in top_congested.items():
        bar = "█" * int(cf * 5)
        print(f"    {rt:20s}: {cf:.2f}  {bar}")

# Tổng chiều dài network
if "length" in edges_gdf.columns:
    total_km = edges_gdf["length"].sum() / 1000
    print(f"\n  Total road network length: {total_km:.1f} km")
    print(f"  Average edge length:       {edges_gdf['length'].mean():.1f} m")
    print(f"  Road density:              {total_km:.0f} km in Dong Da District")

print("\n✓ All done! Check ./dong_da_output/ for all files.")
print("=" * 70)
