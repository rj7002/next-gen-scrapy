import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { Kind, Play } from "@/lib/api"
import { epaColor } from "@/lib/colors"
import { routeTree, type TreeRoute } from "@/lib/stats"
import { cn } from "@/lib/utils"

// Branches drawn from the receiver's spot (bottom centre) as a Hudl-style tree: outside (toward the receiver's own
// sideline) on the left, inside on the right. Route types are route_type.py's classes.
const ROOT: [number, number] = [210, 372]
const LOS = 350
type Branch = { d: string; fruit: [number, number]; label: [number, number, "start" | "middle" | "end"]; dashed?: boolean }
const BRANCHES: Record<string, Branch> = {
  GO: { d: "M210,372 L210,58", fruit: [210, 50], label: [210, 30, "middle"] },
  CORNER: { d: "M210,372 L210,150 L130,84", fruit: [124, 79], label: [110, 76, "end"] },
  POST: { d: "M210,372 L210,150 L290,84", fruit: [296, 79], label: [310, 76, "start"] },
  WHEEL: { d: "M210,372 C166,370 70,362 46,316 L40,150", fruit: [40, 140], label: [40, 120, "middle"] },
  OUT: { d: "M210,372 L210,218 L124,218", fruit: [116, 218], label: [102, 222, "end"] },
  IN: { d: "M210,372 L210,218 L296,218", fruit: [304, 218], label: [318, 222, "start"] },
  SLANT: { d: "M210,372 L210,322 L270,264", fruit: [276, 258], label: [290, 256, "start"] },
  HITCH: { d: "M210,372 L210,270 Q212,292 192,298", fruit: [185, 300], label: [176, 286, "end"] },
  CROSS: { d: "M210,372 L210,334 C214,312 300,312 362,290", fruit: [369, 288], label: [369, 312, "middle"] },
  ANGLE: { d: "M210,372 C202,362 186,358 190,348 L252,336", fruit: [259, 335], label: [271, 339, "start"] },
  FLAT: { d: "M210,372 C196,346 150,334 112,332", fruit: [104, 332], label: [104, 316, "middle"] },
  SCREEN: { d: "M210,372 Q270,396 342,382", fruit: [350, 381], label: [350, 405, "middle"], dashed: true },
}
const NAME = (r: string) => r[0] + r.slice(1).toLowerCase()
const sign = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`
// average EPA per target, scaled so +/-0.5 reaches the ends of the field's per-play EPA scale
const avgColor = (v: number) => epaColor(Math.max(-3, Math.min(3, v * 6)))

/** selected / onPick: the route the chart is filtered to - clicking (or tapping) a branch filters to it, clicking it
 * again clears it; hovering only previews its numbers */
export function RouteTreePanel({ kind, plays, selected = null, onPick }: {
  kind: Kind; plays: Play[]; selected?: string | null; onPick?: (route: string | null) => void
}) {
  const t = routeTree(kind, plays)
  const [hover, setFocus] = useState<string | null>(null)
  const focus = hover ?? selected
  if (!t.total) return null
  const byRoute = new Map(t.routes.map((r) => [r.route, r]))
  const order = Object.keys(BRANCHES).sort((a, b) => (byRoute.get(b)?.targets ?? 0) - (byRoute.get(a)?.targets ?? 0))
  const f: TreeRoute | undefined = focus ? byRoute.get(focus) : undefined
  const noun = kind === "pass" ? "throws" : "targets"
  return (
    <Card className="gap-3 border-border/60 bg-card/50 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-sm font-medium">{kind === "pass" ? "Routes thrown to" : "Route tree"}</CardTitle>
        <CardDescription className="text-xs">
          {t.total} {noun}: branch thickness is the share of {noun}, fruit colour is EPA per target
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 px-4 md:grid-cols-[minmax(0,1fr)_240px] md:items-center">
        <svg viewBox="0 0 420 414" className="mx-auto w-full max-w-[560px] select-none" role="img"
          aria-label={`Route tree: ${t.routes.map((r) => `${NAME(r.route)} ${r.targets}`).join(", ")}`}
          onMouseLeave={() => setFocus(null)}>
          <line x1={14} x2={406} y1={LOS} y2={LOS} className="stroke-muted-foreground/50" strokeDasharray="5 4" strokeWidth={1} />
          {order.map((r) => {
            const b = BRANCHES[r]
            const n = byRoute.get(r)?.targets ?? 0
            const w = n ? 1.75 + 16 * (n / t.total) : 1
            return (
              <path key={r} d={b.d} fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth={w}
                strokeDasharray={!n ? "1 4" : b.dashed ? "6 5" : undefined}
                className={cn("stroke-foreground transition-opacity", !n && "stroke-muted-foreground/60",
                  focus && focus !== r && "opacity-25")} />
            )
          })}
          <circle cx={ROOT[0]} cy={ROOT[1]} r={4} className="fill-foreground" />
          {Object.entries(BRANCHES).map(([r, b]) => {
            const s = byRoute.get(r)
            const [lx, ly, anchor] = b.label
            return (
              <g key={r} className={cn("cursor-pointer transition-opacity", focus && focus !== r && "opacity-30")}
                onMouseEnter={() => s && setFocus(r)} onClick={() => s && onPick?.(selected === r ? null : r)}>
                <circle cx={b.fruit[0]} cy={b.fruit[1]} r={18} fill="transparent" />
                <circle cx={b.fruit[0]} cy={b.fruit[1]} r={s ? 10 : 6} strokeWidth={2}
                  className={s ? "stroke-card" : "fill-transparent stroke-muted-foreground/60"}
                  style={s ? { fill: avgColor(s.epa) } : undefined} />
                <text x={lx} y={ly} textAnchor={anchor} className={cn("text-[12px]", s ? "fill-foreground" : "fill-muted-foreground/70")}>
                  {NAME(r)}
                  {s && <tspan className="fill-muted-foreground font-mono"> {s.targets}</tspan>}
                </text>
              </g>
            )
          })}
        </svg>
        <div className="flex flex-col gap-3 text-xs">
          {f ? (
            <div className="flex flex-col gap-1.5 rounded-lg border border-border/60 p-3">
              <div className="flex items-center gap-2 text-sm font-medium">
                <span className="size-2.5 rounded-full" style={{ background: avgColor(f.epa) }} /> {NAME(f.route)}
              </div>
              <Row k={noun[0].toUpperCase() + noun.slice(1)} v={`${f.targets} (${Math.round((100 * f.targets) / t.total)}%)`} />
              <Row k="Caught" v={`${f.catches} (${Math.round((100 * f.catches) / f.targets)}%)`} />
              <Row k="Yards" v={`${f.yards} (${f.ypt.toFixed(1)} / target)`} />
              <Row k="EPA / target" v={sign(f.epa)} />
              {f.td > 0 && <Row k="Touchdowns" v={String(f.td)} />}
            </div>
          ) : (
            <p className="text-muted-foreground">Hover a route for its numbers; click or tap it to show only those {noun} on the field.</p>
          )}
          <div className="flex flex-col gap-1.5">
            <div className="h-2 w-full rounded-full" style={{ background: `linear-gradient(90deg, ${avgColor(-0.5)}, ${avgColor(0)}, ${avgColor(0.5)})` }} />
            <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
              <span>−0.5</span><span>EPA / target</span><span>+0.5</span>
            </div>
          </div>
          <p className="leading-relaxed text-muted-foreground">
            {t.ngs > 0 && `NGS's route label for ${t.ngs}`}
            {t.ngs > 0 && t.shape > 0 && "; "}
            {t.shape > 0 && `read from the ${kind === "pass" ? "receiver's route chart" : "line"} for ${t.shape}`}.
            {kind === "pass" && t.unmatched > 0 && ` Not in the tree: ${t.unmatched} ${t.unmatched === 1 ? "dot" : "dots"} not matched to a play (no play, so no route).`}
            {kind === "pass" && t.unlabeled > 0 && ` ${t.unlabeled} ${t.unlabeled === 1 ? "throw has" : "throws have"} no route label and no route chart for the target.`}
            {kind === "route" && t.unmatched + t.unlabeled > 0 && ` ${t.unmatched + t.unlabeled} ${t.unmatched + t.unlabeled === 1 ? "line is" : "lines are"} too short to read a route from.`}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-muted-foreground">{k}</span>
      <span className="font-mono tabular-nums">{v}</span>
    </div>
  )
}
