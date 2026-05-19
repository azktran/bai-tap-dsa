# Part 2 — Optimized Shortest Route Calculation

## What Changed

### 1. Parallel matrix build (`_matrix_worker` + `build_matrix_parallel`)

**Original** — serial loop over all n source nodes, one Dijkstra per iteration on the main process.

**Fixed** — `multiprocessing.Pool` distributes source nodes across `min(cpu_count, 8)` workers via `imap_unordered`. Each worker loads the graph independently from disk, runs `single_source_dijkstra`, and returns both the distance row and the full path list. Results are merged into the shared matrix and path cache as they arrive, so wall-clock time scales with the slowest worker batch, not the total work.

---

### 2. Full path cache (`time_path_cache.pkl`, `dist_path_cache.pkl`)

**Original** — matrix stores only scalar distances. Every visualization call re-runs `shortest_path` from scratch.

**Fixed** — each worker also returns the actual node sequences from `single_source_dijkstra`. These are stored in a `dict[(src_idx, dst_idx) → list[node]]` and serialized with `pickle.HIGHEST_PROTOCOL`. Visualization and Part 3 VRP can look up any route in O(1) without touching the graph.

---

### 3. Bidirectional Dijkstra for on-demand single-pair queries (`shortest_node_route`)

**Original** — `nx.shortest_path` = forward Dijkstra from source, expands full reachable subgraph.

**Fixed** — `nx.bidirectional_dijkstra` expands two frontiers simultaneously and halts when they meet. Practical speedup ~2× on road graphs because the search radius is roughly halved.

---

### 4. Load-balanced depot assignment with time-window scoring (`assign_depots_balanced`)

**Original** — each delivery is independently assigned to whichever depot has the minimum travel time. No capacity limit, no awareness of time windows.

**Fixed** — deliveries are sorted descending by `weight_kg` (heaviest-first bin-packing heuristic). For each delivery, the scoring function combines:
- A hard soft-cap: depots that have already reached `DEPOT_CAPACITY = ceil(N_DELIVERIES / N_DEPOTS)` orders sort last.
- Travel time from depot to delivery node.
- A time-window penalty: `max(0, open − arrival)` for early arrivals, `max(0, arrival − close) × 3` for late arrivals (configurable via `TW_LATE_PENALTY_FACTOR`).

This produces balanced loads and minimises TW violations as a secondary objective, without requiring a full ILP solver.

---

### 5. Time-window penalty scoring (`tw_penalty_score`)

**Original** — time windows are stored in metadata but never evaluated.

**Fixed** — `tw_penalty_score(arrival_min, tw_open_str, tw_close_str)` returns 0 when the arrival is feasible, a wait penalty (minutes early) when the delivery point is not yet open, and a weighted late penalty (minutes late × `TW_LATE_PENALTY_FACTOR`) when the window has closed. Used in both depot assignment and in the folium tooltip so each route is labelled `on time` or `penalty=Xm`.

---

### 6. TW compliance reporting

**Original** — no compliance metrics.

**Fixed** — `part2_summary.json` now includes `tw_violations` and `tw_compliance_pct`. Printed to stdout after matrix build completes.

---

## Time Complexity

| Operation | Original | Fixed |
|---|---|---|
| Matrix build (wall clock) | O(n · (E log V)) serial | O((n/w) · (E log V)) parallel, w = workers |
| Matrix build (total work) | O(n · (E log V)) | O(n · (E log V)) — same work, distributed |
| Single-pair shortest path | O(E log V) forward Dijkstra | O(b^(d/2)) bidirectional ≈ 2× faster in practice |
| Depot assignment | O(n · d) | O(n log n + n · d) — sort + assign with TW scoring |
| TW penalty evaluation | — | O(1) per delivery |
| Path cache lookup | O(E log V) re-query | O(1) dict lookup |

n = total matrix points (103), E = graph edges, V = graph nodes, d = number of depots (3), w = parallel workers (≤ 8), b = graph branching factor, d/2 = half the path depth.

---

## Space Complexity

| Structure | Original | Fixed |
|---|---|---|
| Distance matrix | O(n²) float32 = ~43 KB for n=103 | O(n²) float32 — unchanged |
| Path storage | None | O(n² · L̄) where L̄ = mean path length in nodes |
| Worker memory | 1 graph copy | w graph copies in parallel (each ~same RAM as original) |
| Pickle files | None | O(n² · L̄) on disk |

For Dong Da (n = 103, L̄ ≈ 20–40 nodes per path): path cache ≈ 200K–400K node references. Pickled size typically 5–20 MB, negligible for the gains in Part 3 VRP.

---

## New Output Files

| File | Description |
|---|---|
| `time_matrix.npy` | n × n float32 travel-time matrix (minutes) — unchanged format |
| `dist_matrix.npy` | n × n float32 distance matrix (km) — unchanged format |
| `time_path_cache.pkl` | dict[(i,j) → list[node]] for weight_time paths |
| `dist_path_cache.pkl` | dict[(i,j) → list[node]] for weight_distance paths |
| `depot_assignment.json` | balanced depot → order_id mapping |
| `part2_summary.json` | adds `n_workers`, `path_cache_entries`, `tw_violations`, `tw_compliance_pct` |

---

## Part 3 Integration

Load the path cache directly — no Dijkstra needed in VRP:

```python
import pickle
import numpy as np

time_matrix = np.load("dong_da_output/time_matrix.npy")
with open("dong_da_output/time_path_cache.pkl", "rb") as f:
    path_cache = pickle.load(f)

route_nodes = path_cache.get((depot_matrix_idx, delivery_matrix_idx), [])
```
