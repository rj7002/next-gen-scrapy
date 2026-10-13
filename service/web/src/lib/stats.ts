import { type Kind, type Play, weekLabel } from "./api"

const num = (v: unknown): number | null => (v == null || v === "" || Number.isNaN(Number(v)) ? null : Number(v))
const truthy = (v: unknown) => v === true || v === 1 || v === "1" || v === "true"

export function mean(xs: (number | null)[]): number | null {
  const v = xs.filter((x): x is number => x != null)
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}
const sum = (xs: (number | null)[]) => xs.reduce<number>((a, b) => a + (b ?? 0), 0)

export function isCatch(p: Play, kind: Kind): boolean {
  if (kind === "pass") return p.pass_type === "COMPLETE" || p.pass_type === "TOUCHDOWN"
  if (kind === "route") return p.route_type === "COMPLETE"
  return false
}

export function isTd(p: Play, kind: Kind): boolean {
  if (kind === "pass") return p.pass_type === "TOUCHDOWN"
  return truthy(p.chart_touchdown)
}

export interface Kpi {
  label: string
  value: string
  sub?: string
  hint?: string
  tone?: "good" | "bad" | "neutral"
}

const pct = (x: number | null) => (x == null ? "—" : `${(100 * x).toFixed(1)}%`)
const f1 = (x: number | null, d = 1) => (x == null ? "—" : x.toFixed(d))
const signed = (x: number | null, d = 2) => (x == null ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(d)}`)
const tone = (x: number | null): Kpi["tone"] => (x == null ? "neutral" : x > 0.02 ? "good" : x < -0.02 ? "bad" : "neutral")

export function kpis(kind: Kind, plays: Play[]): Kpi[] {
  const m = plays.filter((p) => p.matched)
  const epa = mean(m.map((p) => num(p.epa)))
  const succ = mean(m.map((p) => (p.success == null ? null : truthy(p.success) ? 1 : 0)))
  if (kind === "pass") {
    const comp = plays.filter((p) => isCatch(p, kind))
    const yds = sum(m.filter((p) => truthy(p.complete_pass)).map((p) => num(p.yards_gained)))
    return [
      { label: "Attempts", value: String(plays.length), sub: `${comp.length} complete` },
      { label: "Completion", value: pct(plays.length ? comp.length / plays.length : null), sub: "of all attempts" },
      { label: "Yards", value: yds.toLocaleString(), sub: `${f1(m.length ? yds / m.length : null)} / att` },
      {
        label: "TD : INT",
        value: `${plays.filter((p) => p.pass_type === "TOUCHDOWN").length} : ${plays.filter((p) => p.pass_type === "INTERCEPTION").length}`,
      },
      { label: "aDOT", value: f1(mean(m.map((p) => num(p.air_yards)))), sub: "avg depth of target", hint: "yds" },
      { label: "EPA / play", value: signed(epa), tone: tone(epa) },
      { label: "Success", value: pct(succ), sub: "positive-EPA plays" },
    ]
  }
  if (kind === "route") {
    const rec = plays.filter((p) => isCatch(p, kind))
    const yds = sum(m.filter((p) => truthy(p.complete_pass)).map((p) => num(p.yards_gained)))
    const yac = mean(m.filter((p) => truthy(p.complete_pass)).map((p) => num(p.yards_after_catch)))
    return [
      { label: "Charted targets", value: String(plays.length), sub: `${rec.length} caught` },
      { label: "Catch rate", value: pct(plays.length ? rec.length / plays.length : null) },
      { label: "Yards", value: yds.toLocaleString(), sub: `${f1(rec.length ? yds / rec.length : null)} / rec` },
      { label: "TDs", value: String(plays.filter((p) => isTd(p, kind)).length) },
      { label: "aDOT", value: f1(mean(m.map((p) => num(p.air_yards)))), sub: "avg depth of target" },
      { label: "YAC / rec", value: f1(yac) },
      { label: "EPA / target", value: signed(epa), tone: tone(epa) },
    ]
  }
  const yds = sum(m.map((p) => num(p.yards_gained)))
  const gains = m.map((p) => num(p.yards_gained)).filter((x): x is number => x != null)
  return [
    { label: "Carries", value: String(plays.length) },
    { label: "Yards", value: yds.toLocaleString(), sub: `${f1(gains.length ? yds / gains.length : null)} / carry` },
    { label: "TDs", value: String(plays.filter((p) => isTd(p, kind)).length) },
    { label: "Explosive", value: pct(gains.length ? gains.filter((g) => g >= 10).length / gains.length : null), sub: "10+ yd runs" },
    { label: "Stuffed", value: pct(gains.length ? gains.filter((g) => g <= 0).length / gains.length : null), sub: "no gain or loss" },
    { label: "EPA / carry", value: signed(epa), tone: tone(epa) },
    { label: "Success", value: pct(succ) },
  ]
}

export function byWeek(kind: Kind, plays: Play[]) {
  const g = new Map<string, Play[]>()
  for (const p of plays) {
    const k = String(p.week ?? "?")
    g.set(k, [...(g.get(k) ?? []), p])
  }
  const order = (w: string) => (w.startsWith("post") ? 100 + Number(w.split("-")[1] ?? 0) : Number(w))
  return [...g.entries()]
    .sort((a, b) => order(a[0]) - order(b[0]))
    .map(([week, ps]) => {
      const m = ps.filter((p) => p.matched)
      const yards = sum(
        (kind === "carry" ? m : m.filter((p) => truthy(p.complete_pass))).map((p) => num(p.yards_gained))
      )
      return {
        week: week.startsWith("post") ? weekLabel(week) : `W${week}`,
        plays: ps.length,
        yards,
        epa: Number((mean(m.map((p) => num(p.epa))) ?? 0).toFixed(3)),
        opponent: m.find((p) => p.defteam)?.defteam ?? "",
      }
    })
}

export const DEPTH_BANDS: [string, number, number][] = [
  ["Behind LOS", -99, 0],
  ["0–9", 0, 10],
  ["10–19", 10, 20],
  ["20+", 20, 99],
]

export function depth(kind: Kind, plays: Play[]) {
  return DEPTH_BANDS.map(([band, lo, hi]) => {
    const ps = plays.filter((p) => {
      const a = num(p.air_yards) ?? (kind === "pass" ? num(p.y_coord) : null)
      return a != null && a >= lo && a < hi
    })
    const c = ps.filter((p) => isCatch(p, kind)).length
    return { band, attempts: ps.length, caught: c, rate: ps.length ? Math.round((100 * c) / ps.length) : 0 }
  })
}

export function direction(kind: Kind, plays: Play[]) {
  const col = kind === "carry" ? "run_location" : "pass_location"
  return ["left", "middle", "right"].map((d) => {
    const ps = plays.filter((p) => p[col] === d && p.matched)
    return {
      side: d[0].toUpperCase() + d.slice(1),
      plays: ps.length,
      epa: Number((mean(ps.map((p) => num(p.epa))) ?? 0).toFixed(3)),
      yards: Number((mean(ps.map((p) => num(p.yards_gained))) ?? 0).toFixed(2)),
    }
  })
}

export const GAIN_BINS: [string, number, number][] = [
  ["Loss", -99, 0],
  ["0", 0, 1],
  ["1–2", 1, 3],
  ["3–4", 3, 5],
  ["5–9", 5, 10],
  ["10–19", 10, 20],
  ["20+", 20, 999],
]

export function gains(plays: Play[]) {
  return GAIN_BINS.map(([bin, lo, hi]) => ({
    bin,
    carries: plays.filter((p) => {
      const g = num(p.yards_gained)
      return g != null && g >= lo && g < hi
    }).length,
  }))
}

export function coverage(plays: Play[]) {
  const g = new Map<string, Play[]>()
  for (const p of plays) {
    const c = p.defense_coverage_type
    if (c) g.set(c, [...(g.get(c) ?? []), p])
  }
  return [...g.entries()]
    .map(([cov, ps]) => ({
      coverage: cov.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (s: string) => s.toUpperCase()),
      plays: ps.length,
      epa: Number((mean(ps.map((p) => num(p.epa))) ?? 0).toFixed(3)),
    }))
    .sort((a, b) => b.plays - a.plays)
    .slice(0, 8)
}

export function situation(p: Play): string {
  if (p.down == null) return ""
  const d = ["", "1st", "2nd", "3rd", "4th"][Number(p.down)] ?? ""
  const togo = p.yardline_100 != null && Number(p.ydstogo) >= Number(p.yardline_100) ? "Goal" : p.ydstogo
  return `${d} & ${togo}`
}

export interface TreeRoute { route: string; targets: number; catches: number; yards: number; ypt: number; epa: number; td: number }

/** route-tree numbers per route type, over the plays on the field: tree_route is NGS's label for the target's route
 * where play-by-play has it, else the route read from the chart line's shape (engine._route_tree) - an unmatched
 * route line still has one; an unmatched pass dot doesn't */
export function routeTree(kind: Kind, plays: Play[]) {
  const g = new Map<string, Play[]>()
  let ngs = 0, shape = 0, unmatched = 0, unlabeled = 0
  for (const p of plays) {
    if (!p.tree_route) {
      if (p.matched) unlabeled++
      else unmatched++
      continue
    }
    if (p.tree_route_src === "ngs") ngs++
    else shape++
    g.set(p.tree_route, [...(g.get(p.tree_route) ?? []), p])
  }
  const routes: TreeRoute[] = [...g.entries()].map(([route, ps]) => ({
    route,
    targets: ps.length,
    catches: ps.filter((p) => isCatch(p, kind)).length,
    yards: sum(ps.map((p) => num(p.yards_gained))),
    ypt: mean(ps.filter((p) => p.matched).map((p) => num(p.yards_gained))) ?? 0,   // play-by-play's, so matched only
    epa: mean(ps.map((p) => num(p.epa))) ?? 0,
    td: ps.filter((p) => isTd(p, kind)).length,
  }))
  return { routes, total: ngs + shape, ngs, shape, unmatched, unlabeled }
}

// play-by-play's run_location + run_gap, outside-left to outside-right; guard = A gap, tackle = B, end = C
export const GAPS: { key: string; label: string; letter: string | null; loc: string; gap: string | null }[] = [
  { key: "le", label: "L Edge", letter: "C", loc: "left", gap: "end" },
  { key: "lt", label: "L Tackle", letter: "B", loc: "left", gap: "tackle" },
  { key: "lg", label: "L Guard", letter: "A", loc: "left", gap: "guard" },
  { key: "mid", label: "Middle", letter: null, loc: "middle", gap: null },
  { key: "rg", label: "R Guard", letter: "A", loc: "right", gap: "guard" },
  { key: "rt", label: "R Tackle", letter: "B", loc: "right", gap: "tackle" },
  { key: "re", label: "R Edge", letter: "C", loc: "right", gap: "end" },
]

export function runGaps(plays: Play[]) {
  const runs = plays.filter((p) => p.matched && p.run_location)
  const inGap = (p: Play, g: (typeof GAPS)[number]) => p.run_location === g.loc && (g.gap == null || p.run_gap === g.gap)
  const gaps = GAPS.map((g) => {
    const ps = runs.filter((p) => inGap(p, g))
    return {
      ...g,
      carries: ps.length,
      epa: mean(ps.map((p) => num(p.epa))) ?? 0,
      ypc: mean(ps.map((p) => num(p.yards_gained))) ?? 0,
      success: mean(ps.map((p) => (truthy(p.success) ? 1 : 0))) ?? 0,
      td: ps.filter((p) => truthy(p.rush_touchdown)).length,
    }
  })
  const total = gaps.reduce((n, g) => n + g.carries, 0)
  return { gaps, total, unknown: plays.filter((p) => p.matched).length - total, unmatched: plays.filter((p) => !p.matched).length }
}
