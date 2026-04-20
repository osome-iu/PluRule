from typing import List, Tuple, Set, Dict
import sys
from pathlib import Path
import math

# Import Paul Tol's color schemes
sys.path.append(str(Path(__file__).resolve().parent))
from paul_tol_schemes import tol_cmap


# ========================================
# GEOMETRIC UTILITIES
# ========================================

def get_geometric_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate geometric centroid using shoelace formula."""
    n = len(polygon)
    if n < 3:
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)

    area = 0.0
    cx = 0.0
    cy = 0.0

    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        area += cross
        cx += (polygon[i][0] + polygon[j][0]) * cross
        cy += (polygon[i][1] + polygon[j][1]) * cross

    area *= 0.5
    if abs(area) < 1e-10:
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)

    return (cx / (6.0 * area), cy / (6.0 * area))


def expand_polygon(poly: List[Tuple[float, float]], buffer: float) -> List[Tuple[float, float]]:
    """Expand polygon outward by buffer distance."""
    if len(poly) < 3:
        return poly

    # Calculate centroid
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)

    # Expand each vertex away from centroid
    expanded = []
    for x, y in poly:
        dx = x - cx
        dy = y - cy
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 1e-10:
            # Move point outward by buffer
            expanded.append((x + buffer * dx / dist, y + buffer * dy / dist))
        else:
            expanded.append((x, y))

    return expanded


def polygons_overlap_sat(poly1: List[Tuple[float, float]],
                         poly2: List[Tuple[float, float]],
                         buffer: float = 0.0) -> bool:
    """Check if two convex polygons overlap using Separating Axis Theorem.

    Args:
        poly1: First polygon
        poly2: Second polygon
        buffer: Distance to expand polygons before checking (default: 0.5)
    """
    # Expand polygons by buffer
    poly1 = expand_polygon(poly1, buffer)
    poly2 = expand_polygon(poly2, buffer)

    # Quick bounding box check
    def get_bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))

    bbox1 = get_bbox(poly1)
    bbox2 = get_bbox(poly2)

    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False

    # SAT: Check all edge normals
    def get_edges(poly):
        edges = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length > 1e-10:
                edges.append((normal[0] / length, normal[1] / length))
        return edges

    def project_polygon(poly, axis):
        projections = [p[0] * axis[0] + p[1] * axis[1] for p in poly]
        return min(projections), max(projections)

    for axis in get_edges(poly1) + get_edges(poly2):
        min1, max1 = project_polygon(poly1, axis)
        min2, max2 = project_polygon(poly2, axis)
        if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
            return False

    return True


# ========================================
# COLOR ASSIGNMENT
# ========================================

def assign_colors_with_conflicts(centroids: List[Tuple[float, float]],
                                 polygons: List[List[Tuple[float, float]]],
                                 logger=None) -> List[str]:
    """
    Assign colors with conflict resolution using graph coloring.

    Strategy:
    - Build conflict graph from overlapping convex hulls
    - Greedy coloring: assign colors left-to-right
    - Overlapping clusters: distance ≥ 2 in palette
    - All clusters: unique colors when possible (allows reuse if >26 clusters)

    Args:
        centroids: List of (x, y) median centroid tuples
        polygons: List of convex hull polygons for overlap detection
        logger: Optional logger for debug output

    Returns:
        List of hex color strings, one per cluster
    """
    n = len(centroids)
    if n == 0:
        return []

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Rainbow_PuBr discrete colors (26 colors)
    rainbow_pubr_colors = [
        '#6F4C9B', '#6059A9', '#5568B8', '#4E79C5', '#4D8AC6',
        '#4E96BC', '#549EB3', '#59A5A9', '#60AB9E', '#69B190',
        '#77B77D', '#8CBC68', '#A6BE54', '#BEBC48', '#D1B541',
        '#DDAA3C', '#E49C39', '#E78C35', '#E67932', '#E4632D',
        '#DF4828', '#DA2222', '#B8221E', '#95211B', '#721E17',
        '#521A13'
    ]

    # Step 1: Build overlap graph
    overlaps: Dict[int, Set[int]] = {i: set() for i in range(n)}
    conflict_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_sat(polygons[i], polygons[j]):
                overlaps[i].add(j)
                overlaps[j].add(i)
                conflict_count += 1

    log(f"  Found {conflict_count} overlapping cluster pairs")
    if conflict_count > 0:
        for i, neighbors in overlaps.items():
            if neighbors:
                log(f"    Cluster {i} overlaps with: {sorted(neighbors)}")

    # Step 2: Sort clusters by X position (left to right)
    x_sorted = sorted(range(n), key=lambda i: centroids[i][0])
    x_rank_map = {idx: rank for rank, idx in enumerate(x_sorted)}

    # Step 3: Greedy graph coloring with spatial ordering
    color_assignments: Dict[int, int] = {}  # cluster_idx -> color_index
    used_colors: Set[int] = set()

    for cluster_idx in x_sorted:
        # Get colors used by overlapping neighbors
        neighbor_colors = {color_assignments[n] for n in overlaps[cluster_idx] if n in color_assignments}

        # Find valid colors: not used AND distance ≥ 2 from overlapping neighbors
        num_colors = len(rainbow_pubr_colors)  # 26 colors
        valid_colors = []
        for c in range(num_colors):
            if c in used_colors:
                continue  # Color already used by another cluster
            if all(abs(c - nc) >= 2 for nc in neighbor_colors):
                valid_colors.append(c)

        if not valid_colors:
            # Fallback: relax distance constraint to 1
            valid_colors = [c for c in range(num_colors)
                           if c not in used_colors and c not in neighbor_colors]

        if not valid_colors:
            # Emergency fallback: just avoid used colors
            valid_colors = [c for c in range(num_colors) if c not in used_colors]

        if not valid_colors:
            # Final fallback: allow color reuse, just avoid neighbors
            valid_colors = [c for c in range(num_colors) if c not in neighbor_colors]

        if not valid_colors:
            # Ultimate fallback: use any color (shouldn't happen, but prevents crash)
            valid_colors = list(range(num_colors))

        # Prefer color closest to ideal X-position
        ideal_color = int((x_rank_map[cluster_idx] / max(n - 1, 1)) * (num_colors - 1))
        best_color = min(valid_colors, key=lambda c: abs(c - ideal_color))

        # Log color assignment
        is_reused = best_color in used_colors
        if neighbor_colors:
            if best_color != ideal_color:
                reuse_note = " [REUSED]" if is_reused else ""
                log(f"  Cluster {cluster_idx}: SHIFTED from ideal {ideal_color} to {best_color} (neighbors using {sorted(neighbor_colors)}){reuse_note}")
            else:
                log(f"  Cluster {cluster_idx}: kept ideal color {ideal_color} (neighbors using {sorted(neighbor_colors)}, no conflict)")
        elif is_reused:
            log(f"  Cluster {cluster_idx}: REUSED color {best_color} (no overlapping neighbors)")

        color_assignments[cluster_idx] = best_color
        used_colors.add(best_color)

    # Step 4: Map to hex colors
    result = [rainbow_pubr_colors[color_assignments[i]] for i in range(n)]
    return result


