
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
