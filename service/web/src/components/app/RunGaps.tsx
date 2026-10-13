import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { Play } from "@/lib/api"
import { epaColor } from "@/lib/colors"
import { runGaps } from "@/lib/stats"
import { cn } from "@/lib/utils"

// x of each gap's column; linemen sit between them, the back behind the center
const COL = [44, 122, 200, 280, 360, 438, 516]
const OL: [string, number][] = [["LT", 161], ["LG", 240], ["C", 280], ["RG", 320], ["RT", 399]]
const OL_Y = 236
const TIP_Y = 116
const RB: [number, number] = [280, 292]
// where each gap's arrow passes the line (between linemen) and its gap letter
const THROUGH: [number, string | null][] = [[96, "C"], [200, "B"], [260, "A"], [280, null], [300, "A"], [360, "B"], [464, "C"]]
const sign = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`
// rushing EPA per carry spans less than passing: +/-0.3 reaches the ends of the scale
const avgColor = (v: number) => epaColor(Math.max(-3, Math.min(3, v * 10)))

export function RunGapsPanel({ plays }: { plays: Play[] }) {
  const g = runGaps(plays)
  const [focus, setFocus] = useState<number | null>(null)
  if (!g.total) return null
  const f = focus != null ? g.gaps[focus] : null
  return (
    <Card className="gap-3 border-border/60 bg-card/50 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-sm font-medium">Run gaps</CardTitle>
        <CardDescription className="text-xs">
          {g.total} carries, by the gap play-by-play records: arrow thickness is the share of carries, colour is EPA per rush
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 px-2 sm:px-4">
        <svg viewBox="0 0 560 320" className="mx-auto w-full max-w-[720px] select-none" role="img"
          aria-label={`Run gaps: ${g.gaps.map((x) => `${x.label} ${x.carries} carries ${sign(x.epa)} EPA`).join(", ")}`}
          onMouseLeave={() => setFocus(null)}>
          {g.gaps.map((x, i) => {
            const [gx] = THROUGH[i]
            const share = x.carries / g.total
            const w = x.carries ? 2.5 + 30 * share : 1.25
            const color = x.carries ? avgColor(x.epa) : undefined
            // arrowhead: a triangle along the last segment, sized with the line
            const [ax, ay] = [COL[i], TIP_Y]
            const ang = Math.atan2(ay - OL_Y, ax - gx)
            const L = 8 + w * 0.9
            const hw = 5 + w * 0.6
            const bx = ax - L * Math.cos(ang), by = ay - L * Math.sin(ang)
            const head = `${ax},${ay} ${bx + hw * Math.sin(ang)},${by - hw * Math.cos(ang)} ${bx - hw * Math.sin(ang)},${by + hw * Math.cos(ang)}`
            return (
              <g key={x.key} className={cn("transition-opacity", focus != null && focus !== i && "opacity-25")}>
                <polyline points={`${RB[0]},${RB[1]} ${gx},${OL_Y} ${bx},${by}`} fill="none" strokeWidth={w}
                  strokeLinejoin="round" strokeDasharray={x.carries ? undefined : "1 4"}
                  className={x.carries ? undefined : "stroke-muted-foreground/60"} style={color ? { stroke: color } : undefined} />
                {x.carries > 0 && <polygon points={head} style={{ fill: color }} />}
              </g>
            )
          })}
          {OL.map(([n, x]) => (
            <g key={n}>
              <circle cx={x} cy={OL_Y} r={15} className="fill-slate-700 stroke-card dark:fill-slate-600" strokeWidth={2} />
              <text x={x} y={OL_Y + 4} textAnchor="middle" className="fill-white text-[11px] font-semibold">{n}</text>
            </g>
          ))}
          {THROUGH.map(([gx, letter], i) => letter && (
            <g key={i} className={cn(focus != null && focus !== i && "opacity-25")}>
              <circle cx={gx} cy={OL_Y - 42} r={9} className="fill-card stroke-muted-foreground/70" strokeWidth={1} />
              <text x={gx} y={OL_Y - 38.5} textAnchor="middle" className="fill-muted-foreground text-[10px] font-semibold">{letter}</text>
            </g>
          ))}
          <circle cx={RB[0]} cy={RB[1]} r={16} className="fill-rose-600 stroke-card" strokeWidth={2} />
          <text x={RB[0]} y={RB[1] + 4} textAnchor="middle" className="fill-white text-[11px] font-semibold">RB</text>
          {g.gaps.map((x, i) => (
            <g key={x.key} className={cn("cursor-pointer transition-opacity", focus != null && focus !== i && "opacity-35")}
              onMouseEnter={() => setFocus(i)} onClick={() => setFocus(focus === i ? null : i)}>
              <rect x={COL[i] - 38} y={0} width={76} height={TIP_Y + 4} fill="transparent" />
              <text x={COL[i]} y={20} textAnchor="middle" className="fill-muted-foreground text-[13px]">{x.label}</text>
              <text x={COL[i]} y={56} textAnchor="middle"
                className={cn("font-mono text-[21px] font-semibold", x.carries ? "fill-foreground" : "fill-muted-foreground/60")}>
                {x.carries ? sign(x.epa) : "—"}
              </text>
              <text x={COL[i]} y={80} textAnchor="middle" className="fill-muted-foreground text-[12px]">
                {x.carries} car
              </text>
              {x.carries > 0 && <rect x={COL[i] - 16} y={90} width={32} height={4} rx={2} style={{ fill: avgColor(x.epa) }} />}
            </g>
          ))}
        </svg>
        <div className="flex min-h-[2.5rem] flex-wrap items-center justify-between gap-x-6 gap-y-2 px-2 text-xs">
          {f && f.carries ? (
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <span className="font-medium">{f.label}{f.letter && <span className="text-muted-foreground"> ({f.letter} gap)</span>}</span>
              <span><span className="font-mono">{f.carries}</span> <span className="text-muted-foreground">carries ({Math.round((100 * f.carries) / g.total)}%)</span></span>
              <span><span className="font-mono">{f.ypc.toFixed(1)}</span> <span className="text-muted-foreground">yds / carry</span></span>
              <span><span className="font-mono">{sign(f.epa)}</span> <span className="text-muted-foreground">EPA / rush</span></span>
              <span><span className="font-mono">{Math.round(100 * f.success)}%</span> <span className="text-muted-foreground">success</span></span>
              {f.td > 0 && <span><span className="font-mono">{f.td}</span> <span className="text-muted-foreground">TD</span></span>}
            </div>
          ) : (
            <span className="text-muted-foreground">
              Hover or tap a gap for its numbers.
              {g.unmatched > 0 && ` Not shown: ${g.unmatched} ${g.unmatched === 1 ? "line" : "lines"} on the chart not matched to a play (the gap comes from play-by-play).`}
              {g.unknown > 0 && ` ${g.unknown} matched ${g.unknown === 1 ? "carry has" : "carries have"} no gap in play-by-play.`}
            </span>
          )}
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-muted-foreground">−0.3</span>
            <div className="h-2 w-28 rounded-full" style={{ background: `linear-gradient(90deg, ${avgColor(-0.3)}, ${avgColor(0)}, ${avgColor(0.3)})` }} />
            <span className="font-mono text-[10px] text-muted-foreground">+0.3 EPA / rush</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
