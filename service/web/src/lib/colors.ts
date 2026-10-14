import type { Kind, Play } from "./api"

export type ColorBy = "outcome" | "epa" | "route" | "confidence" | "down"

// Next Gen Stats' own chart colours
export const OUTCOME: Record<string, string> = {
  COMPLETE: "#34d399",
  INCOMPLETE: "#d4d4d8",
  TOUCHDOWN: "#60a5fa",
  INTERCEPTION: "#f87171",
}
export const GAIN: Record<string, string> = { LOSS: "#f87171", SHORT: "#facc15", LONG: "#34d399" }
export const ROUTE_LINE = { complete: "#fafafa", incomplete: "#8b8b94", after: "#34d399", td: "#60a5fa" }

export const ROUTE_COLORS: Record<string, string> = {
  GO: "#f43f5e",
  POST: "#f97316",
  CORNER: "#f59e0b",
  OUT: "#3b82f6",
  IN: "#14b8a6",
  SLANT: "#ec4899",
  HITCH: "#a78bfa",
  CROSS: "#22c55e",
  FLAT: "#06b6d4",
  SCREEN: "#94a3b8",
  WHEEL: "#eab308",
  ANGLE: "#84cc16",
}

export const DOWN_COLORS: Record<string, string> = { "1": "#38bdf8", "2": "#a78bfa", "3": "#f59e0b", "4": "#f43f5e" }

function lerp(a: number[], b: number[], t: number) {
  return a.map((x, i) => Math.round(x + (b[i] - x) * t))
}
const hex = (c: number[]) => `#${c.map((x) => x.toString(16).padStart(2, "0")).join("")}`

/** red (bad) -> zinc -> green (good), clamped at +/-3 EPA */
export function epaColor(epa: number | null | undefined): string {
  if (epa == null) return "#71717a"
  const t = Math.max(-1, Math.min(1, epa / 3))
  const mid = [161, 161, 170]
  return t < 0 ? hex(lerp(mid, [239, 68, 68], -t)) : hex(lerp(mid, [16, 185, 129], t))
}

/** amber (unsure) -> cyan (certain) */
export function confColor(p: number | null | undefined): string {
  if (p == null) return "#71717a"
  const t = Math.max(0, Math.min(1, (p - 0.4) / 0.6))
  return hex(lerp([245, 158, 11], [34, 211, 238], t))
}

export function routeGuess(play: Play): string | null {
  return play.route_guess?.[0]?.[0] ?? null
}

/** the play's route class: NGS's own label for the target's route where there is one, else what the line's shape
 * says (a route chart's own line, or for a pass the target's route chart) - the same label the route tree uses */
export function playRoute(play: Play): string | null {
  return play.tree_route ?? routeGuess(play)
}

/** main colour of a play under a colour mode (routes: the pre-catch line) */
export function playColor(play: Play, kind: Kind, by: ColorBy): string {
  switch (by) {
    case "epa":
      return epaColor(play.epa)
    case "confidence":
      return confColor(play.location_prob)
    case "down":
      return DOWN_COLORS[String(play.down ?? "")] ?? "#71717a"
    case "route":
      return ROUTE_COLORS[playRoute(play) ?? ""] ?? "#a1a1aa"
    default:
      if (kind === "pass") return OUTCOME[play.pass_type] ?? "#a1a1aa"
      if (kind === "carry") return GAIN[play.gain_class] ?? "#a1a1aa"
      return play.route_type === "COMPLETE" ? ROUTE_LINE.complete : ROUTE_LINE.incomplete
  }
}

// heatmap ramp: transparent -> violet -> magenta -> orange -> yellow
const RAMP: [number, number[]][] = [
  [0, [17, 12, 48]],
  [0.25, [91, 33, 182]],
  [0.5, [219, 39, 119]],
  [0.75, [249, 115, 22]],
  [1, [253, 224, 71]],
]
export function heat(t: number): [number, number, number] {
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1]
      const [t1, c1] = RAMP[i]
      const c = lerp(c0, c1, (t - t0) / (t1 - t0))
      return [c[0], c[1], c[2]]
    }
  }
  return [253, 224, 71]
}
