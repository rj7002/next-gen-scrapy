// What a heatmap shows: how many plays end where (volume), how well they went there (a value map: EPA, CPOE, catch
// or success rate, as a kernel-weighted average that fades out where there are few plays), or where one half of a
// split goes more than the other (compare: each side's plays as a share of that side, so it's tendency, not volume).
import type { Kind, Play } from "./api"
import { heat } from "./colors"
import { isCatch } from "./stats"

export type HeatMetric = "volume" | "epa" | "cpoe" | "catch" | "success" | "compare"

export const METRICS: Record<Kind, [HeatMetric, string][]> = {
  pass: [["volume", "Volume"], ["epa", "EPA"], ["cpoe", "CPOE"], ["compare", "Compare"]],
  route: [["volume", "Volume"], ["epa", "EPA"], ["catch", "Catch rate"], ["compare", "Compare"]],
  carry: [["volume", "Volume"], ["epa", "EPA"], ["success", "Success rate"], ["compare", "Compare"]],
}

const truthy = (v: unknown) => v === true || v === 1 || v === "1" || v === "true"
const falsy = (v: unknown) => v === false || v === 0 || v === "0" || v === "false"
const num = (v: unknown): number | null => (v == null || v === "" || Number.isNaN(Number(v)) ? null : Number(v))

export interface Split { key: string; label: string; a: string; b: string; kinds: Kind[]; side: (p: Play) => "a" | "b" | null }

const ALL: Kind[] = ["pass", "route", "carry"]
export const SPLITS: Split[] = [
  { key: "coverage", label: "Man vs zone", a: "vs man", b: "vs zone", kinds: ["pass", "route"],
    side: (p) => (p.defense_man_zone_type === "MAN_COVERAGE" ? "a" : p.defense_man_zone_type === "ZONE_COVERAGE" ? "b" : null) },
  { key: "pressure", label: "Pressured vs clean", a: "pressured", b: "clean pocket", kinds: ["pass", "route"],
    side: (p) => (truthy(p.was_pressure) ? "a" : falsy(p.was_pressure) ? "b" : null) },
  { key: "blitz", label: "Blitz vs no blitz", a: "blitzed", b: "no blitz", kinds: ALL,
    side: (p) => { const n = num(p.n_blitzers); return n == null ? null : n > 0 ? "a" : "b" } },
  { key: "pa", label: "Play action vs not", a: "play action", b: "no play action", kinds: ALL,
    side: (p) => (truthy(p.is_play_action) ? "a" : falsy(p.is_play_action) ? "b" : null) },
  { key: "shotgun", label: "Shotgun vs under center", a: "shotgun", b: "under center", kinds: ALL,
    side: (p) => (truthy(p.shotgun) ? "a" : falsy(p.shotgun) ? "b" : null) },
  { key: "box", label: "Stacked vs light box", a: "8+ in the box", b: "7 or fewer", kinds: ["carry"],
    side: (p) => { const n = num(p.n_defense_box); return n == null ? null : n >= 8 ? "a" : "b" } },
  { key: "downs", label: "Early vs late downs", a: "1st & 2nd down", b: "3rd & 4th down", kinds: ALL,
    side: (p) => { const d = num(p.down); return d == null ? null : d <= 2 ? "a" : "b" } },
  { key: "half", label: "1st vs 2nd half", a: "1st half", b: "2nd half", kinds: ALL,
    side: (p) => { const q = num(p.qtr); return q == null ? null : q <= 2 ? "a" : q <= 4 ? "b" : null } },
  { key: "score", label: "Leading vs trailing", a: "leading", b: "trailing", kinds: ALL,
    side: (p) => { const s = num(p.score_differential); return s == null || s === 0 ? null : s > 0 ? "a" : "b" } },
  { key: "home", label: "Home vs away", a: "at home", b: "on the road", kinds: ALL,
    side: (p) => (p.posteam_type === "home" ? "a" : p.posteam_type === "away" ? "b" : null) },
]
export const splitsFor = (kind: Kind) => SPLITS.filter((s) => s.kinds.includes(kind))
export const splitOf = (kind: Kind, key: string) => splitsFor(kind).find((s) => s.key === key) ?? splitsFor(kind)[0]

/** A value map's per-play value, the value its colour scale centres on, and how far either way is full colour. */
export function valueScale(metric: HeatMetric, kind: Kind, plays: Play[]) {
  const rate = (f: (p: Play) => number | null) => {
    const v = plays.map(f).filter((x): x is number => x != null)
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0.5
  }
  switch (metric) {
    case "epa": return { value: (p: Play) => (p.matched ? num(p.epa) : null), mid: 0, span: 0.6, fmt: (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)} EPA` }
    case "cpoe": return { value: (p: Play) => num(p.cpoe), mid: 0, span: 25, fmt: (v: number) => `${v > 0 ? "+" : ""}${Math.round(v)}% CPOE` }
    case "catch": {
      const f = (p: Play) => (p.matched ? (isCatch(p, kind) ? 1 : 0) : null)
      return { value: f, mid: rate(f), span: 0.35, fmt: (v: number) => `${Math.round(100 * v)}% caught` }
    }
    case "success": {
      const f = (p: Play) => (p.success == null ? null : truthy(p.success) ? 1 : 0)
      return { value: f, mid: rate(f), span: 0.35, fmt: (v: number) => `${Math.round(100 * v)}% success` }
    }
    default: return null
  }
}

const PRIOR = 3

export const VALUE_RAMP: [string, string, string] = ["#ef4444", "#a1a1aa", "#10b981"]
export const COMPARE_A = "#38bdf8"
export const COMPARE_B = "#f97316"

const hex = (h: string): [number, number, number] => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number]
const mix = (a: number[], b: number[], t: number) => a.map((x, i) => x + (b[i] - x) * t)

/** One heatmap on a half-yard grid: per cell its height / strength h (0-1: the 3D slices rise to it), how opaque the
 * flat map is there, and its colour - or rgb null for the volume map, which is coloured by height. */
export interface HeatField { nx: number; ny: number; h: Float32Array; alpha: Float32Array; rgb: Uint8ClampedArray | null; floor: number }

export function heatField(plays: Play[], kind: Kind, heatOf: "end" | "path", metric: HeatMetric, splitKey: string,
                          x0: number, width: number, ymin: number, ymax: number,
                          spot: (p: Play, kind: Kind) => [number, number] | null): HeatField | null {
  const res = 0.5
  const nx = Math.round(width / res)
  const ny = Math.round((ymax - ymin) / res)
  // averages and comparisons need more plays under the kernel than a count does, or a few plays swing them
  const wide = metric !== "volume"
  const sigma = heatOf === "path" && kind !== "pass" ? (wide ? 2.0 : 1.2) : wide ? 2.6 : 1.7
  const rad = Math.ceil((3 * sigma) / res)
  const ker: number[] = []
  for (let dy = -rad; dy <= rad; dy++)
    for (let dx = -rad; dx <= rad; dx++) ker.push(Math.exp(-((dx * res) ** 2 + (dy * res) ** 2) / (2 * sigma * sigma)))
  // every play is points with a weight (an end point, or its path's points sharing one play's weight)
  const each = (p: Play, f: (x: number, y: number, w: number) => void) => {
    if (heatOf === "path" && kind !== "pass") {
      const xs: number[] = p.path_x ?? []
      const yv: number[] = p.path_y ?? []
      const w = 1 / Math.max(1, xs.length / 2)
      for (let i = 0; i < xs.length; i += 2) f(xs[i], yv[i], w)
    } else {
      const l = spot(p, kind)
      if (l) f(l[0], l[1], 1)
    }
  }
  const splat = (grid: Float32Array, x: number, y: number, w: number) => {
    const cx = Math.round((x - x0) / res)
    const cy = Math.round((ymax - y) / res)
    let k = 0
    for (let dy = -rad; dy <= rad; dy++)
      for (let dx = -rad; dx <= rad; dx++, k++) {
        const gx = cx + dx
        const gy = cy + dy
        if (gx >= 0 && gx < nx && gy >= 0 && gy < ny) grid[gy * nx + gx] += ker[k] * w
      }
  }
  const N = nx * ny
  const h = new Float32Array(N)
  const alpha = new Float32Array(N)

  if (metric === "compare") {
    const split = splitOf(kind, splitKey)
    const A = new Float32Array(N)
    const B = new Float32Array(N)
    let na = 0
    let nb = 0
    for (const p of plays) {
      const s = split?.side(p)
      if (s === "a") (na++, each(p, (x, y, w) => splat(A, x, y, w)))
      else if (s === "b") (nb++, each(p, (x, y, w) => splat(B, x, y, w)))
    }
    if (!na || !nb) return null
    const d = new Float32Array(N)
    let m = 0
    for (let i = 0; i < N; i++) {
      d[i] = A[i] / na - B[i] / nb
      m = Math.max(m, Math.abs(d[i]))
    }
    if (m <= 0) return null
    const rgb = new Uint8ClampedArray(3 * N)
    const ca = hex(COMPARE_A)
    const cb = hex(COMPARE_B)
    for (let i = 0; i < N; i++) {
      const s = d[i] / m
      h[i] = Math.pow(Math.abs(s), 0.8)
      alpha[i] = Math.min(1, h[i] * 1.6) * 0.92
      rgb.set(s >= 0 ? ca : cb, 3 * i)
    }
    return { nx, ny, h, alpha, rgb, floor: 0.08 }
  }

  const den = new Float32Array(N)
  const scale = valueScale(metric, kind, plays)
  const tot = scale ? new Float32Array(N) : null
  const wsum = scale ? new Float32Array(N) : null
  for (const p of plays) {
    const v = scale?.value(p)
    each(p, (x, y, w) => {
      splat(den, x, y, w)
      if (scale && v != null) (splat(wsum!, x, y, w), splat(tot!, x, y, w * v))
    })
  }
  let max = 0
  for (const v of den) max = Math.max(max, v)
  if (max <= 0) return null
  for (let i = 0; i < N; i++) {
    h[i] = Math.pow(den[i] / max, 0.65)
    alpha[i] = Math.min(1, h[i] * 1.6) * 0.92
  }
  if (!scale) return { nx, ny, h, alpha, rgb: null, floor: 0.14 }
  // a value map: the colour is the local average, pulled toward his overall number where it rests on few plays
  // (as if PRIOR more plays at his average were there too), and fading out where there are hardly any
  const rgb = new Uint8ClampedArray(3 * N)
  const [lo, mid, hi] = VALUE_RAMP.map(hex)
  const vals = plays.map(scale.value).filter((v): v is number => v != null)
  const overall = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : scale.mid
  for (let i = 0; i < N; i++) {
    const w = wsum![i]
    if (w <= 1e-6) {
      alpha[i] = 0
      continue
    }
    const avg = (tot![i] + PRIOR * overall) / (w + PRIOR)
    const t = Math.max(-1, Math.min(1, (avg - scale.mid) / scale.span))
    rgb.set(t >= 0 ? mix(mid, hi, t) : mix(mid, lo, -t), 3 * i)
    alpha[i] = Math.min(1, w / 4) * 0.88
  }
  return { nx, ny, h, alpha, rgb, floor: 0.1 }
}

/** The flat heatmap image (a data URL) of a heat field. */
export function heatImage(f: HeatField): string {
  const canvas = document.createElement("canvas")
  canvas.width = f.nx
  canvas.height = f.ny
  const ctx = canvas.getContext("2d")!
  const img = ctx.createImageData(f.nx, f.ny)
  for (let i = 0; i < f.h.length; i++) {
    const c = f.rgb ? [f.rgb[3 * i], f.rgb[3 * i + 1], f.rgb[3 * i + 2]] : heat(f.h[i])
    img.data.set([c[0], c[1], c[2], Math.round(255 * f.alpha[i])], 4 * i)
  }
  ctx.putImageData(img, 0, 0)
  return canvas.toDataURL()
}

/** The 3D heatmap: slices, slice k showing where the field's height reaches its level - stacked at heights in
 * proportion they make a contoured mound. The volume map colours each slice for its level; the others keep each
 * spot's own colour all the way up. Below the floor the flat map underneath does the showing. */
export function heatSlices(f: HeatField, slices = 36): string[] {
  const canvas = document.createElement("canvas")
  canvas.width = f.nx
  canvas.height = f.ny
  const ctx = canvas.getContext("2d")!
  const out: string[] = []
  for (let k = 1; k <= slices; k++) {
    const level = f.floor + ((1 - f.floor) * k) / slices
    const lc = f.rgb ? null : heat(level)
    const shade = 0.72 + 0.28 * level              // lower slices a touch darker: reads as depth from the side
    const opacity = 0.5 + 0.45 * (k / slices)      // and the lower ones partly see-through
    const img = ctx.createImageData(f.nx, f.ny)
    let any = false
    for (let i = 0; i < f.h.length; i++) {
      const a = Math.min(1, Math.max(0, ((f.h[i] - level) * slices) / (1 - f.floor) + 0.5))
      if (a <= 0 || (f.rgb && f.alpha[i] <= 0.02)) continue
      any = true
      const c = lc ?? [f.rgb![3 * i], f.rgb![3 * i + 1], f.rgb![3 * i + 2]]
      img.data.set([c[0] * shade, c[1] * shade, c[2] * shade, Math.round(255 * a * opacity * (f.rgb ? Math.min(1, f.alpha[i] * 1.2) : 1))], 4 * i)
    }
    if (!any) break
    ctx.putImageData(img, 0, 0)
    out.push(canvas.toDataURL())
  }
  return out
}
