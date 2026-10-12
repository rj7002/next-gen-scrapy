import type { FilterSpec, Play } from "./api"

export type FilterValue =
  | { op: "in"; values: string[] }
  | { op: "bool"; value: boolean }
  | { op: "range"; min?: number; max?: number }

export type FilterState = Record<string, FilterValue>

export interface ViewOptions {
  weeks: Set<string> | null // null = every week
}

const truthy = (v: unknown) => v === true || v === 1 || v === "1" || v === "true"
const falsy = (v: unknown) => v === false || v === 0 || v === "0" || v === "false"

export function matches(play: Play, col: string, f: FilterValue): boolean {
  const v = play[col]
  if (f.op === "in") return v != null && f.values.includes(String(v))
  if (f.op === "bool") return f.value ? truthy(v) : falsy(v)
  if (v == null || v === "") return false
  const n = Number(v)
  if (Number.isNaN(n)) return false
  return (f.min == null || n >= f.min) && (f.max == null || n <= f.max)
}

export function applyFilters(plays: Play[], state: FilterState, opts: ViewOptions): Play[] {
  const active = Object.entries(state)
  return plays.filter((p) => {
    if (opts.weeks && !opts.weeks.has(String(p.week))) return false
    for (const [col, f] of active) if (!matches(p, col, f)) return false
    return true
  })
}

/** The NL parser's query params (col, col__in, col__gte, ...) -> filter state. */
export function fromParams(params: Record<string, string>, specs: FilterSpec[]): FilterState {
  const byCol = new Map(specs.map((s) => [s.col, s]))
  const out: FilterState = {}
  for (const [key, raw] of Object.entries(params)) {
    const [col, op = "eq"] = key.split("__")
    const spec = byCol.get(col)
    if (!spec) continue
    if (spec.kind === "bool") {
      out[col] = { op: "bool", value: truthy(raw) }
    } else if (op === "in" || op === "eq") {
      out[col] = { op: "in", values: raw.split(",") }
    } else if (["gte", "gt", "lte", "lt"].includes(op)) {
      const cur = (out[col] as Extract<FilterValue, { op: "range" }>) ?? { op: "range" }
      const n = Number(raw)
      if (op === "gte" || op === "gt") cur.min = n
      else cur.max = n
      out[col] = { ...cur, op: "range" }
    }
  }
  return out
}

export function describe(spec: FilterSpec | undefined, col: string, f: FilterValue): string {
  const label = spec?.label ?? col
  if (f.op === "bool") return `${label}: ${f.value ? "yes" : "no"}`
  if (f.op === "in") {
    const names = new Map((spec?.choices ?? []).map(([v, l]) => [String(v), l]))
    return `${label}: ${f.values.map((v) => names.get(v) ?? pretty(v)).join(", ")}`
  }
  if (f.min != null && f.max != null) return `${label} ${fmt(f.min)}–${fmt(f.max)}`
  if (f.min != null) return `${label} ≥ ${fmt(f.min)}`
  return `${label} ≤ ${fmt(f.max!)}`
}

const fmt = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(2))

export function pretty(v: unknown): string {
  if (v == null) return "—"
  const s = String(v)
  if (/^[A-Z0-9_]+$/.test(s) && s.length > 2)
    return s
      .toLowerCase()
      .split("_")
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ")
  return s
}

/** Distinct non-empty values of a column, most common first. */
export function distinct(plays: Play[], col: string): { value: string; count: number }[] {
  const m = new Map<string, number>()
  for (const p of plays) {
    const v = p[col]
    if (v == null || v === "") continue
    const k = String(v)
    m.set(k, (m.get(k) ?? 0) + 1)
  }
  return [...m.entries()].map(([value, count]) => ({ value, count })).sort((a, b) => b.count - a.count)
}

export function numericRange(plays: Play[], col: string): [number, number] | null {
  let lo = Infinity
  let hi = -Infinity
  for (const p of plays) {
    const v = p[col]
    if (v == null || v === "") continue
    const n = Number(v)
    if (Number.isNaN(n)) continue
    lo = Math.min(lo, n)
    hi = Math.max(hi, n)
  }
  return Number.isFinite(lo) ? [lo, hi] : null
}

export function hasData(plays: Play[], col: string): boolean {
  return plays.some((p) => p[col] != null && p[col] !== "")
}
