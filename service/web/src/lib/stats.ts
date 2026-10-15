import { type Kind, type Play, weekLabel } from "./api"

const num = (v: unknown): number | null => (v == null || v === "" || Number.isNaN(Number(v)) ? null : Number(v))
const truthy = (v: unknown) => v === true || v === 1 || v === "1" || v === "true"

export function mean(xs: (number | null)[]): number | null {
  const v = xs.filter((x): x is number => x != null)
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}
const sum = (xs: (number | null)[]) => xs.reduce<number>((a, b) => a + (b ?? 0), 0)

// a play with play-by-play says what happened on it; a chart line / dot with no play has only the chart's own colour
export function isCatch(p: Play, kind: Kind): boolean {
  if (kind === "carry") return false
  if (p.complete_pass != null && p.complete_pass !== "") return truthy(p.complete_pass)
  if (kind === "pass") return p.pass_type === "COMPLETE" || p.pass_type === "TOUCHDOWN"
  return p.route_type === "COMPLETE"
}

export function isTd(p: Play, kind: Kind): boolean {
  if (p.player_td != null) return truthy(p.player_td)
  if (kind === "pass") return p.pass_type === "TOUCHDOWN"
  return truthy(p.chart_touchdown)
}

/** a play that's one of his targets / carries / attempts in the game log (engine._official): every play-by-play
 * play of his, drawn or not - not a chart line no play fits, a two-point try, or a play he only got on a lateral */
export const inLog = (p: Play) => truthy(p.in_log)
const PASS_EST = "Not drawn by Next Gen Stats. Its spot is estimated from play-by-play air yards and direction: depth is usually within a yard, but across the field it's only known to the side."

/** why a play has no dot / line of its own (or a line drawn back to the line of scrimmage), or null */
export function lineNote(kind: Kind, p: Play): string | null {
  const noun = kind === "route" ? "target" : "carry"
  if (kind === "pass") return p.estimated ? PASS_EST : null
  if (p.lateral) return `He got the ball on a lateral: his yards on it count in his totals, but it isn't one of his ${noun === "carry" ? "carries" : "targets"}.`
  if (p.estimated) return `No chart line for this ${noun}: Next Gen Stats didn't draw it, or its line couldn't be told apart from the others. It still counts in his totals, from play-by-play.`
  if (p.line_hidden) return "Its chart line couldn't be traced back to the line of scrimmage (only a piece of it was readable), so it isn't drawn."
  if (p.start_extended) return "The first yard or two of its chart line is hidden under the line of scrimmage or other lines; it's drawn straight back to the line."
  return null
}

/** his yards in the game log on these plays (passing / receiving / rushing, laterals to him included) */
export const logYards = (plays: Play[]) => sum(plays.filter((p) => p.matched).map((p) => num(p.player_yards)))

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
  const log = plays.filter(inLog)
  const m = log
  const epa = mean(m.map((p) => num(p.epa)))
  const succ = mean(m.map((p) => (p.success == null ? null : truthy(p.success) ? 1 : 0)))
  const yds = logYards(plays)
  const tds = plays.filter((p) => p.matched && isTd(p, kind)).length
  if (kind === "pass") {
    const comp = log.filter((p) => isCatch(p, kind))
    return [
      { label: "Attempts", value: String(log.length), sub: `${comp.length} complete` },
      { label: "Completion", value: pct(log.length ? comp.length / log.length : null), sub: "of all attempts" },
      { label: "Yards", value: yds.toLocaleString(), sub: `${f1(log.length ? yds / log.length : null)} / att` },
      { label: "TD : INT", value: `${tds} : ${log.filter((p) => truthy(p.interception)).length}` },
      { label: "aDOT", value: f1(mean(m.map((p) => num(p.air_yards)))), sub: "avg depth of target", hint: "yds" },
      { label: "EPA / play", value: signed(epa), tone: tone(epa) },
      { label: "Success", value: pct(succ), sub: "positive-EPA plays" },
    ]
  }
  if (kind === "route") {
    const rec = log.filter((p) => isCatch(p, kind))
    const yac = mean(rec.map((p) => num(p.yards_after_catch)))
    return [
      { label: "Targets", value: String(log.length), sub: `${rec.length} caught` },
      { label: "Catch rate", value: pct(log.length ? rec.length / log.length : null) },
      { label: "Yards", value: yds.toLocaleString(), sub: `${f1(rec.length ? yds / rec.length : null)} / rec` },
      { label: "TDs", value: String(tds) },
      { label: "aDOT", value: f1(mean(m.map((p) => num(p.air_yards)))), sub: "avg depth of target" },
      { label: "YAC / rec", value: f1(yac) },
      { label: "EPA / target", value: signed(epa), tone: tone(epa) },
    ]
  }
  const gains = m.map((p) => num(p.yards_gained)).filter((x): x is number => x != null)
  return [
    { label: "Carries", value: String(log.length) },
    { label: "Yards", value: yds.toLocaleString(), sub: `${f1(log.length ? yds / log.length : null)} / carry` },
    { label: "TDs", value: String(tds) },
    { label: "Explosive", value: pct(gains.length ? gains.filter((g) => g >= 10).length / gains.length : null), sub: "10+ yd runs" },
    { label: "Stuffed", value: pct(gains.length ? gains.filter((g) => g <= 0).length / gains.length : null), sub: "no gain or loss" },
    { label: "EPA / carry", value: signed(epa), tone: tone(epa) },
    { label: "Success", value: pct(succ) },
  ]
}

export function byWeek(_kind: Kind, plays: Play[]) {
  const g = new Map<string, Play[]>()
  for (const p of plays) {
    const k = String(p.week ?? "?")
    g.set(k, [...(g.get(k) ?? []), p])
  }
  const order = (w: string) => (w.startsWith("post") ? 100 + Number(w.split("-")[1] ?? 0) : Number(w))
  return [...g.entries()]
    .sort((a, b) => order(a[0]) - order(b[0]))
    .map(([week, ps]) => {
      const m = ps.filter(inLog)
      return {
        week: week.startsWith("post") ? weekLabel(week) : `W${week}`,
        plays: m.length,
        yards: logYards(ps),
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
    yards: logYards(ps),
    ypt: mean(ps.filter((p) => p.matched).map((p) => num(p.player_yards))) ?? 0,   // play-by-play's, so matched only
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

export interface ReceiverRow {
  name: string; position: string | null; targets: number; share: number; catches: number; yards: number
  adot: number | null; epa: number | null; td: number
}

/** a quarterback's throws by target: volume, share, and what came of them; plus each week's target share for
 * the top `top` receivers (the rest as "Others") */
export function receivers(plays: Play[], top = 5) {
  const g = new Map<string, Play[]>()
  for (const p of plays) {
    const n = p.receiver_player_name
    if (!n || !inLog(p)) continue
    g.set(n, [...(g.get(n) ?? []), p])
  }
  const total = [...g.values()].reduce((a, ps) => a + ps.length, 0)
  const rows: ReceiverRow[] = [...g.entries()].map(([name, ps]) => {
    const m = ps.filter((p) => p.matched)
    const comp = ps.filter((p) => isCatch(p, "pass"))
    return {
      name, position: ps.find((p) => p.receiver_position)?.receiver_position ?? null,
      targets: ps.length, share: total ? ps.length / total : 0, catches: comp.length,
      yards: logYards(ps),
      adot: mean(m.map((p) => num(p.air_yards))), epa: mean(m.map((p) => num(p.epa))),
      td: ps.filter((p) => isTd(p, "pass")).length,
    }
  }).sort((a, b) => b.targets - a.targets)
  const lead = rows.slice(0, top).map((r) => r.name)
  const wk = new Map<string, Play[]>()
  for (const p of plays) if (p.receiver_player_name && inLog(p)) {
    const k = String(p.week ?? "?")
    wk.set(k, [...(wk.get(k) ?? []), p])
  }
  const order = (w: string) => (w.startsWith("post") ? 100 + Number(w.split("-")[1] ?? 0) : Number(w))
  const weekly = [...wk.entries()].sort((a, b) => order(a[0]) - order(b[0])).map(([week, ps]) => {
    const row: Record<string, number | string> = { week: week.startsWith("post") ? weekLabel(week) : `W${week}`, n: ps.length }
    for (const name of lead) row[name] = Math.round((1000 * ps.filter((p) => p.receiver_player_name === name).length) / ps.length) / 10
    row.Others = Math.round((1000 * ps.filter((p) => !lead.includes(p.receiver_player_name)).length) / ps.length) / 10
    return row
  })
  return { rows, lead, weekly, total }
}
