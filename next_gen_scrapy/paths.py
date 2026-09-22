"""
Turn a binary mask of drawn lines (routes / carries) into ordered polylines.

The mask is thinned to a 1-px skeleton, the skeleton is split into chains at junction pixels, and
chains are re-joined *through* junctions by pairing the pieces whose directions continue
most smoothly. That lets two lines that cross (X) come out as two lines rather than four stubs,
and lets a line that merely touches another (T) stay one line.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]   # circular order


def _crossing_number(sk):
    """For every skeleton pixel: number of separate neighbour runs (1 = end, 2 = line, >=3 = junction)."""
    p = np.pad(sk, 1).astype(np.uint8)
    h, w = sk.shape
    ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _OFFS]
    cn = np.zeros(sk.shape, np.int32)
    for i in range(8):
        cn += ((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.int32)
    n = sum(r.astype(np.int32) for r in ring)
    cn[(n == 8)] = 0                      # interior of a blob
    return np.where(sk, cn, 0), np.where(sk, n, 0)


def _neighbors(y, x, sk):
    h, w = sk.shape
    for dy, dx in _OFFS:
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
            yield yy, xx


def _trace_chain(pixels_set, start):
    """Order the pixels of a simple chain beginning at `start`."""
    order = [start]
    seen = {start}
    cur = start
    while True:
        nxt = None
        # prefer 4-neighbours first to avoid cutting corners
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]:
            q = (cur[0] + dy, cur[1] + dx)
            if q in pixels_set and q not in seen:
                nxt = q
                break
        if nxt is None:
            return order
        order.append(nxt)
        seen.add(nxt)
        cur = nxt


def _prune_spurs(sk, spur_len):
    """Remove short side branches (skeleton hairs) that end at a junction."""
    for _ in range(3):
        cn, _n = _crossing_number(sk)
        ends = np.argwhere((cn == 1) & sk)
        removed = False
        for y, x in ends:
            path = [(y, x)]
            cur = (y, x)
            prev = None
            hit_junction = False
            while len(path) <= spur_len:
                nb = [q for q in _neighbors(cur[0], cur[1], sk) if q != prev and q not in path]
                if not nb:
                    break
                if cn[cur] >= 3 or len(nb) > 1 and cn[nb[0]] >= 3:
                    hit_junction = True
                    break
                prev, cur = cur, nb[0]
                path.append(cur)
                if cn[cur] >= 3:
                    hit_junction = True
                    path.pop()
                    break
            if hit_junction and len(path) < spur_len:
                for q in path:
                    sk[q] = False
                removed = True
        if not removed:
            break
    return sk


def _tangent(pts, k=12):
    """Unit vector pointing from pts[0] into the chain, using up to k pixels."""
    j = min(len(pts) - 1, k)
    v = np.array([pts[j][1] - pts[0][1], pts[j][0] - pts[0][0]], float)      # (dx, dy)
    n = np.linalg.norm(v)
    return v / n if n else v


def extract_paths(mask, min_len=25, spur_len=10, max_turn_deg=70):
    """
    mask: boolean image of the lines. Returns a list of Nx2 float arrays of (col, row) pixel
    positions along each traced line (direction arbitrary).
    """
    sk = skeletonize(mask > 0)
    sk = _prune_spurs(sk, spur_len)
    sk = skeletonize(sk)
    cn, _ = _crossing_number(sk)
    junc = (cn >= 3) & sk
    # merge neighbouring junction pixels into clusters
    jl, nj = ndi.label(ndi.binary_dilation(junc, np.ones((3, 3))) & sk, structure=np.ones((3, 3)))
    chain_mask = sk & (jl == 0)
    cl, nc = ndi.label(chain_mask, structure=np.ones((3, 3)))

    centroids = {j: np.array(ndi.center_of_mass(jl == j))[::-1] for j in range(1, nj + 1)}

    edges = []      # dict(pts=[(y,x)...], ends=[node_a, node_b])
    cnc, _ = _crossing_number(chain_mask)
    objs = ndi.find_objects(cl)
    for c in range(1, nc + 1):
        sl = objs[c - 1]
        ys, xs = np.nonzero(cl[sl] == c)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        pix = set(zip(ys.tolist(), xs.tolist()))
        ends = [p for p in pix if cnc[p] == 1] or [next(iter(pix))]
        order = _trace_chain(pix, ends[0])
        if len(order) < 2:
            continue
        edges.append(dict(pts=order))
    # attach chain ends to junction clusters
    for e in edges:
        ends = []
        for pt in (e["pts"][0], e["pts"][-1]):
            node = 0
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    yy, xx = pt[0] + dy, pt[1] + dx
                    if 0 <= yy < jl.shape[0] and 0 <= xx < jl.shape[1] and jl[yy, xx] > 0:
                        node = jl[yy, xx]
            ends.append(node)
        e["nodes"] = ends

    # pair chain ends at each junction by straightest continuation
    partner = {}
    by_node = {}
    for i, e in enumerate(edges):
        for side in (0, 1):
            if e["nodes"][side]:
                by_node.setdefault(e["nodes"][side], []).append((i, side))
    cos_min = -np.cos(np.radians(max_turn_deg))          # tangents point away from node; straight = -1
    for node, items in by_node.items():
        tans = {}
        for (i, side) in items:
            pts = edges[i]["pts"] if side == 0 else edges[i]["pts"][::-1]
            tans[(i, side)] = _tangent(pts, 15)
        cands = []
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                if items[a][0] == items[b][0]:
                    continue
                c = float(np.dot(tans[items[a]], tans[items[b]]))
                if c < cos_min:
                    cands.append((c, items[a], items[b]))
        used = set()
        for c, ia, ib in sorted(cands, key=lambda t: t[0]):
            if ia in used or ib in used:
                continue
            partner[ia] = ib
            partner[ib] = ia
            used.update([ia, ib])

    # walk
    visited = set()
    paths = []

    def walk(i, side):
        pts_out = []
        cur_i, cur_side = i, side
        while True:
            visited.add(cur_i)
            e = edges[cur_i]
            seq = e["pts"] if cur_side == 0 else e["pts"][::-1]
            pts_out.extend((x, y) for y, x in seq)
            far = (cur_i, 1 - cur_side)
            nxt = partner.get(far)
            if nxt is None or nxt[0] in visited:
                return pts_out
            node = edges[far[0]]["nodes"][far[1]]
            pts_out.append(tuple(centroids[node]))
            cur_i, cur_side = nxt

    for i, e in enumerate(edges):
        for side in (0, 1):
            if i not in visited and (i, side) not in partner:
                paths.append(np.array(walk(i, side), float))
                break
    for i in range(len(edges)):
        if i not in visited:
            paths.append(np.array(walk(i, 0), float))
    return [p for p in paths if _length(p) >= min_len]


def _length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def _end_tangent(p, k=15):
    """Unit vector at the last point of p pointing outward (away from the path body)."""
    q = p[::-1]
    j = min(len(q) - 1, k)
    v = q[0] - q[j]
    n = np.linalg.norm(v)
    return v / n if n else v


def _occluded_fraction(a, b, occluder):
    """Fraction of the straight segment a->b that lies on `occluder` pixels."""
    n = max(2, int(np.linalg.norm(b - a)))
    t = np.linspace(0, 1, n)
    pts = np.round(a[None, :] + t[:, None] * (b - a)[None, :]).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, occluder.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, occluder.shape[0] - 1)
    return float(occluder[pts[:, 1], pts[:, 0]].mean())


def join_collinear(paths, max_gap=30.0, max_turn_deg=35.0, occluder=None, free_gap=8.0, occl_frac=0.6,
                  max_merges=None, reject=None):
    """
    Join path ends across a gap (a line hidden under another line) when they line up: the two outward
    directions must be opposed and point at each other. Gaps longer than `free_gap` pixels are only
    bridged if `occluder` (a mask of other drawn lines) covers at least `occl_frac` of the gap.
    `reject(merged_path)` may veto a candidate merge.
    """
    paths = [p.copy() for p in paths]
    cos_t = np.cos(np.radians(max_turn_deg))
    merges = 0
    while max_merges is None or merges < max_merges:
        best = None
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                for ei in (0, 1):                       # 0 = start of path, 1 = end
                    for ej in (0, 1):
                        a = paths[i] if ei else paths[i][::-1]
                        b = paths[j] if ej else paths[j][::-1]
                        gap_v = b[-1] - a[-1]
                        gap = np.linalg.norm(gap_v)
                        if gap > max_gap:
                            continue
                        if occluder is not None and gap > free_gap and \
                                _occluded_fraction(a[-1], b[-1], occluder) < occl_frac:
                            continue
                        ta, tb = _end_tangent(a), _end_tangent(b)
                        if np.dot(ta, tb) > -cos_t:
                            continue
                        if gap > 1e-6:
                            u = gap_v / gap
                            if np.dot(ta, u) < cos_t or np.dot(tb, -u) < cos_t:
                                continue
                        if reject is not None and reject(np.vstack([a, b[::-1]])):
                            continue
                        # score: shorter gap and straighter is better
                        score = gap + 20 * (1 + np.dot(ta, tb))
                        if best is None or score < best[0]:
                            best = (score, i, j, ei, ej)
        if best is None:
            return paths
        _, i, j, ei, ej = best
        a = paths[i] if ei else paths[i][::-1]
        b = paths[j] if ej else paths[j][::-1]
        merged = np.vstack([a, b[::-1]])
        paths = [p for k, p in enumerate(paths) if k not in (i, j)] + [merged]
        merges += 1
    return paths


# ---------------------------------------------------------------------------------------------
# Lines that all leave the same start point (carries from one handoff, routes from one alignment)
# share their first few yards. `lines` below are dicts with a pixel path "px" (start -> end), an
# optional per-point label array "seg", and a "kind" used to prefer stems of the same type.
# ---------------------------------------------------------------------------------------------
def v_index(px, img_to_field, dip=1.5, deep=-1.5):
    """
    Index where a line dips deeper than both of its ends (a "V"), else None. A single carry / route never
    goes deeper than its own start, so such a line is two lines that were traced fused at their common
    start point.
    """
    y = img_to_field(px)[:, 1]
    i = int(np.argmin(y))
    if 5 <= i <= len(px) - 6 and y[i] <= deep and y[0] - y[i] > dip and y[-1] - y[i] > dip:
        return i
    return None


def split_v(lines, img_to_field, dip=1.5, deep=-1.5):
    out = []
    for l in lines:
        i = v_index(l["px"], img_to_field, dip, deep)
        if i is None:
            out.append(l)
            continue
        a, b = dict(l), dict(l)
        a["px"], b["px"] = l["px"][:i + 1][::-1], l["px"][i:]
        if "seg" in l:
            a["seg"], b["seg"] = l["seg"][:i + 1][::-1], l["seg"][i:]
        out += [a, b]
    return out


def attach_trunks(lines, img_to_field, max_start_y, max_dist=32.0, min_align=0.55, passes=3):
    """
    Give every line that does not start where lines should start (y <= max_start_y) the stretch it is
    missing: find the line it branches from (a nearby point where that line, followed backwards, continues
    our direction) and prepend that line's path from its own start up to the branch point.
    """
    def start_y(l):
        return img_to_field(l["px"][:1])[0, 1]

    for _ in range(passes):
        changed = False
        for l in sorted((l for l in lines if start_y(l) > max_start_y), key=start_y):
            s = l["px"][0]
            t_out = _end_tangent(l["px"][::-1])                     # direction leaving our start, backwards
            best = None
            for o in lines:
                if o is l or start_y(o) > max_start_y:
                    continue
                P = o["px"]
                d = np.linalg.norm(P - s, axis=1)
                for k in np.nonzero(d < max_dist)[0]:
                    if k < 3:
                        continue
                    b = P[max(k - 10, 0)] - P[k]
                    nb = np.linalg.norm(b)
                    if nb < 1e-6:
                        continue
                    align = float(np.dot(t_out, b / nb))
                    if align < min_align:
                        continue
                    cost = d[k] + 40 * (1 - align) + (0 if o.get("kind") == l.get("kind") else 12)
                    if best is None or cost < best[0]:
                        best = (cost, o, int(k))
            if best is not None:
                _, o, k = best
                l["px"] = np.vstack([o["px"][:k + 1], l["px"]])
                if "seg" in l and "seg" in o:
                    l["seg"] = np.concatenate([o["seg"][:k + 1], l["seg"]])
                changed = True
        if not changed:
            break
