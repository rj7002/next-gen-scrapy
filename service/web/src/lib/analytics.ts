// The Analytics tab's numbers: form over the season, the spread of outcomes, down & distance, field vs boundary,
// box counts, and the per-play pairs its scatter plots draw. All from the plays on screen (filters applied).
import { type Kind, type Play, weekLabel } from "./api"
import { isCatch } from "./stats"

const num = (v: unknown): number | null => (v == null || v === "" || Number.isNaN(Number(v)) ? null : Number(v))
const truthy = (v: unknown) => v === true || v === 1 || v === "1" || v === "true"
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)
const weekOrder = (w: unknown) => {
  const s = String(w ?? "")
  return s.startsWith("post") ? 100 + Number(s.split("-")[1] ?? 0) : Number(s) || 0
}

/** plays in the order they happened */
export function chronological(plays: Play[]): Play[] {
  return [...plays].sort((a, b) =>
    weekOrder(a.week) - weekOrder(b.week) || String(a.game_id).localeCompare(String(b.game_id)) || (num(a.play_id) ?? 0) - (num(b.play_id) ?? 0))
}

/** rolling EPA / success over the last `n` plays, one point per play, with the week each play was in */
export function rollingForm(plays: Play[], n = 40) {
  const ps = chronological(plays.filter((p) => p.matched && num(p.epa) != null))
  const out: { i: number; epa: number; success: number; week: string; label: string }[] = []
  const epa: number[] = []
  const suc: number[] = []
  ps.forEach((p, i) => {
    epa.push(num(p.epa)!)
    suc.push(truthy(p.success) ? 1 : 0)
    if (epa.length > n) (epa.shift(), suc.shift())
    if (epa.length >= Math.min(n, Math.max(8, Math.floor(ps.length / 4))))
      out.push({ i: i + 1, epa: +mean(epa)!.toFixed(3), success: +(100 * mean(suc)!).toFixed(1), week: String(p.week ?? ""),
                 label: weekLabel(p.week, true) })
  })
  return { points: out, window: n, total: ps.length }
}

/** EPA per play in half-point bins, and the share of booms (+1 or better) and busts (-1 or worse) */
export function epaSpread(plays: Play[]) {
  const vals = plays.filter((p) => p.matched).map((p) => num(p.epa)).filter((v): v is number => v != null)
  const lo = -4, hi = 6, w = 0.5
  const bins = Array.from({ length: (hi - lo) / w }, (_, i) => ({ x: lo + i * w, label: (lo + i * w).toFixed(1), n: 0 }))
  for (const v of vals) bins[Math.min(bins.length - 1, Math.max(0, Math.floor((v - lo) / w)))].n++
  return { bins, total: vals.length, boom: vals.filter((v) => v >= 1).length / (vals.length || 1),
           bust: vals.filter((v) => v <= -1).length / (vals.length || 1), median: [...vals].sort((a, b) => a - b)[Math.floor(vals.length / 2)] ?? null }
}

export const DIST_BUCKETS: [string, number, number][] = [["1-2", 1, 2], ["3-6", 3, 6], ["7-10", 7, 10], ["11+", 11, 99]]

/** EPA / play, success and count for each down x distance bucket */
export function downDistance(plays: Play[]) {
  return [1, 2, 3, 4].map((down) => ({
    down,
    cells: DIST_BUCKETS.map(([label, a, b]) => {
      const ps = plays.filter((p) => p.matched && num(p.down) === down && (num(p.ydstogo) ?? -1) >= a && (num(p.ydstogo) ?? -1) <= b)
      const e = ps.map((p) => num(p.epa)).filter((v): v is number => v != null)
      return { label, n: ps.length, epa: mean(e), success: mean(ps.map((p) => (truthy(p.success) ? 1 : 0))) }
    }),
  }))
}

/** which way a play went, left / middle / right, from where it ended up (or pbp's direction) */
function direction(p: Play, kind: Kind, landing: (p: Play, kind: Kind) => [number, number] | null): "left" | "middle" | "right" | null {
  if (kind === "carry") return (p.run_location as "left" | "middle" | "right") ?? null
  const at = landing(p, kind)
  if (at) return at[0] < -5 ? "left" : at[0] > 5 ? "right" : "middle"
  return (p.pass_location as "left" | "middle" | "right") ?? null
}

/** toward the wide side of the field (away from the hash the ball was on), the middle, or the short boundary side */
export function fieldBoundary(plays: Play[], kind: Kind, landing: (p: Play, kind: Kind) => [number, number] | null) {
  const groups: Record<string, Play[]> = { Field: [], Middle: [], Boundary: [] }
  let unknown = 0
  for (const p of plays) {
    const hash = p.starting_hash
    const dir = direction(p, kind, landing)
    if ((hash !== "L" && hash !== "R") || !dir) {
      unknown++
      continue
    }
    const side = dir === "middle" ? "Middle" : (hash === "L") === (dir === "right") ? "Field" : "Boundary"
    groups[side].push(p)
  }
  const total = groups.Field.length + groups.Middle.length + groups.Boundary.length
  return {
    rows: Object.entries(groups).map(([side, ps]) => {
      const m = ps.filter((p) => p.matched)
      return {
        side, n: ps.length, share: total ? +(100 * ps.length / total).toFixed(1) : 0,
        epa: +(mean(m.map((p) => num(p.epa)).filter((v): v is number => v != null)) ?? 0).toFixed(3),
        rate: kind === "carry"
          ? +(mean(m.map((p) => num(p.yards_gained)).filter((v): v is number => v != null)) ?? 0).toFixed(1)
          : +(100 * (ps.filter((p) => isCatch(p, kind)).length / (ps.length || 1))).toFixed(1),
      }
    }),
    total, unknown,
  }
}

/** yards per carry and success by defenders in the box */
export function boxCounts(plays: Play[]) {
  const groups: [string, (n: number) => boolean][] = [["≤5", (n) => n <= 5], ["6", (n) => n === 6], ["7", (n) => n === 7],
    ["8", (n) => n === 8], ["9+", (n) => n >= 9]]
  return groups.map(([label, test]) => {
    const ps = plays.filter((p) => p.matched && num(p.n_defense_box) != null && test(num(p.n_defense_box)!))
    return { label, n: ps.length,
             ypc: +(mean(ps.map((p) => num(p.yards_gained)).filter((v): v is number => v != null)) ?? 0).toFixed(1),
             success: +(100 * (mean(ps.map((p) => (truthy(p.success) ? 1 : 0))) ?? 0)).toFixed(1) }
  }).filter((g) => g.n > 0)
}

/** (x, y) pairs for a scatter plot, with the play's outcome for its colour */
export function pairs(plays: Play[], x: string, y: string, keep: (p: Play) => boolean = () => true) {
  return plays.filter((p) => p.matched && keep(p) && num(p[x]) != null && num(p[y]) != null)
    .map((p) => ({ x: num(p[x])!, y: num(p[y])!, id: p.id, outcome: String(p.pass_type ?? p.route_type ?? p.gain_class ?? "") }))
}

/** air vs YAC EPA on completions, summed per group (receiver, route...) - the biggest groups first. (On an
 * incompletion nflverse's YAC EPA is just minus the air EPA, so those would only cancel out.) */
export function airYac(plays: Play[], kind: Kind, groupOf: (p: Play) => string | null, top = 8) {
  const g = new Map<string, { air: number; yac: number; n: number }>()
  for (const p of plays) {
    if (!isCatch(p, kind)) continue
    const k = groupOf(p)
    const a = num(p.air_epa), y = num(p.yac_epa)
    if (!k || a == null) continue
    const cur = g.get(k) ?? { air: 0, yac: 0, n: 0 }
    cur.air += a
    cur.yac += y ?? 0
    cur.n++
    g.set(k, cur)
  }
  return [...g.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, top)
    .map(([name, v]) => ({ name, air: +v.air.toFixed(2), yac: +v.yac.toFixed(2), n: v.n }))
}

export const READS: [string, string][] = [["1", "1st read"], ["2", "2nd read"], ["CHK", "Checkdown"], ["SD", "Scramble drill"], ["DES", "Designed"]]

/** throws by which read they went to (FTN charting): share, completion %, EPA / play */
export function reads(plays: Play[]) {
  const all = plays.filter((p) => READS.some(([k]) => String(p.read_thrown) === k))
  return READS.map(([k, label]) => {
    const ps = all.filter((p) => String(p.read_thrown) === k)
    const m = ps.filter((p) => p.matched)
    return { label, n: ps.length, share: all.length ? +(100 * ps.length / all.length).toFixed(1) : 0,
             comp: +(100 * (ps.filter((p) => isCatch(p, "pass")).length / (ps.length || 1))).toFixed(1),
             epa: +(mean(m.map((p) => num(p.epa)).filter((v): v is number => v != null)) ?? 0).toFixed(3) }
  }).filter((r) => r.n > 0)
}

// ---------------------------------------------------------------- alignment (route charts) and the hole map (carries)
// The ball's spot across the field: the FTN hash it was snapped from (2022+), or - for a carry from under center - the
// back's own start, right behind the ball. Chart x is yards from the middle of the field, offense's left negative.
const HASH_X: Record<string, number> = { L: -3.08, M: 0, R: 3.08 }
const arr = (v: unknown): number[] => (Array.isArray(v) ? (v as number[]) : [])

function ballX(p: Play, kind: Kind): number | null {
  const h = HASH_X[String(p.starting_hash)]
  if (h != null) return h
  if (kind === "carry" && (p.shotgun === false || p.shotgun === 0 || p.shotgun === "0")) return arr(p.path_x)[0] ?? null
  return null
}

export type Align = "slot" | "wide" | "inline" | "backfield"
export const ALIGN_LABEL: Record<Align, string> = { slot: "Slot / tight split", wide: "Wide", inline: "Inline / tight", backfield: "Backfield" }
export const ALIGNS: Align[] = ["wide", "slot", "inline", "backfield"]

/** where a targeted receiver lined up: his route's start, against the ball (needs a clean start and the hash) */
export function alignmentOf(p: Play): Align | null {
  if (!truthy(p.start_ok)) return null
  const xs = arr(p.path_x), ys = arr(p.path_y)
  const b = HASH_X[String(p.starting_hash)]
  if (!xs.length || b == null) return null
  if (ys[0] < -2.5) return "backfield"
  const dx = Math.abs(xs[0] - b)
  return dx < 4.5 ? "inline" : dx < 10 ? "slot" : "wide"
}

export function alignment(plays: Play[]) {
  const known = plays.filter((p) => p.matched && alignmentOf(p))
  const rows = ALIGNS.map((a) => {
    const ps = known.filter((p) => alignmentOf(p) === a)
    return { key: a, label: ALIGN_LABEL[a], n: ps.length, share: known.length ? ps.length / known.length : 0,
             epa: mean(ps.map((p) => num(p.epa)).filter((v): v is number => v != null)),
             catch: ps.length ? ps.filter((p) => p.route_type === "COMPLETE").length / ps.length : null,
             adot: mean(ps.map((p) => num(p.air_yards)).filter((v): v is number => v != null)) }
  })
  const wk = new Map<string, Play[]>()
  for (const p of chronological(known)) wk.set(String(p.week), [...(wk.get(String(p.week)) ?? []), p])
  const weekly = [...wk.entries()].map(([w, ps]) => ({
    week: w.startsWith("post") ? weekLabel(w) : `W${w}`, n: ps.length,
    ...Object.fromEntries(ALIGNS.map((a) => [a, Math.round((100 * ps.filter((p) => alignmentOf(p) === a).length) / ps.length)])),
  }))
  return { rows, weekly, known: known.length, unknown: plays.filter((p) => p.matched).length - known.length }
}

/** where a run crossed the line of scrimmage, yards from the ball (negative = offense's left) */
export function holeOf(p: Play): number | null {
  const xs = arr(p.path_x), ys = arr(p.path_y)
  const b = ballX(p, "carry")
  if (xs.length < 2 || b == null) return null
  const i = ys.findIndex((y) => y >= 0)
  if (i <= 0) return null
  const t = ys[i] !== ys[i - 1] ? (0 - ys[i - 1]) / (ys[i] - ys[i - 1]) : 0
  return xs[i - 1] + t * (xs[i] - xs[i - 1]) - b
}

export const OUTSIDE = 5      // yards from the ball: outside the tackles

export function holes(plays: Play[]) {
  const runs = plays.filter((p) => p.matched && !truthy(p.implausible)).map((p) => ({ p, h: holeOf(p) })).filter((r) => r.h != null) as
    { p: Play; h: number }[]
  const bins = Array.from({ length: 40 }, (_, i) => ({ x: i - 20, n: 0, epa: [] as number[] }))
  for (const r of runs) {
    const b = bins[Math.min(39, Math.max(0, Math.floor(r.h) + 20))]
    b.n++
    const e = num(r.p.epa)
    if (e != null) b.epa.push(e)
  }
  const side = (out: boolean) => runs.filter((r) => (Math.abs(r.h) >= OUTSIDE) === out)
  const stat = (rs: { p: Play }[]) => ({
    n: rs.length, epa: mean(rs.map((r) => num(r.p.epa)).filter((v): v is number => v != null)),
    ypc: mean(rs.map((r) => num(r.p.yards_gained)).filter((v): v is number => v != null)),
  })
  const label = (p: Play) => (p.run_location === "middle" ? "middle" : (p.run_gap as string) || null)
  const byGap = ["middle", "guard", "tackle", "end"].map((g) => {
    const hs = runs.filter((r) => label(r.p) === g).map((r) => Math.abs(r.h)).sort((a, b) => a - b)
    return { gap: g, n: hs.length, median: hs.length ? hs[Math.floor(hs.length / 2)] : null }
  })
  return { runs: runs.length, missing: plays.filter((p) => p.matched).length - runs.length,
           bins: bins.map((b) => ({ x: b.x, n: b.n, share: runs.length ? b.n / runs.length : 0, epa: mean(b.epa) })),
           outside: runs.length ? side(true).length / runs.length : null, inside: stat(side(false)), out: stat(side(true)), byGap,
           left: runs.length ? runs.filter((r) => r.h < -1).length / runs.length : null }
}
