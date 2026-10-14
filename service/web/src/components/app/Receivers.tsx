import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { type ChartConfig, ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import type { Play } from "@/lib/api"
import { useReducedMotion } from "@/lib/motion"
import { receivers } from "@/lib/stats"
import { cn } from "@/lib/utils"

const COLORS = ["#38bdf8", "#a78bfa", "#34d399", "#f59e0b", "#f472b6", "#52525b"]
const signed = (x: number | null) => (x == null ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(2)}`)

/** A quarterback's receivers: who gets the ball and what comes of it, and how the targets moved week to week.
 * Clicking a receiver filters the chart to the throws to him (clicking him again clears it). */
export function ReceiversPanel({ plays, selected, onPick }: { plays: Play[]; selected: string | null; onPick: (name: string | null) => void }) {
  const animate = !useReducedMotion()
  const { rows, lead, weekly, total } = receivers(plays)
  if (!rows.length) return null
  const shown = rows.slice(0, 10)
  const maxShare = Math.max(...shown.map((r) => r.share))
  const config = Object.fromEntries([...lead, "Others"].map((n, i) => [n, { label: n, color: COLORS[Math.min(i, COLORS.length - 1)] }])) satisfies ChartConfig
  return (
    <Card className="gap-3 border-border/60 bg-card/50 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-sm font-medium">Receivers</CardTitle>
        <CardDescription className="text-xs">
          {total} throws with a target · click a receiver to see only the throws to him
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-5 px-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-xs">
            <thead>
              <tr className="text-left text-[11px] text-muted-foreground">
                <th className="pb-2 font-medium">Receiver</th>
                <th className="pb-2 font-medium">Target share</th>
                <th className="pb-2 text-right font-medium">Tgt</th>
                <th className="pb-2 text-right font-medium">Catch %</th>
                <th className="pb-2 text-right font-medium">Yds</th>
                <th className="pb-2 text-right font-medium">aDOT</th>
                <th className="pb-2 text-right font-medium">TD</th>
                <th className="pb-2 text-right font-medium">EPA/tgt</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => {
                const on = selected === r.name
                return (
                  <tr key={r.name} onClick={() => onPick(on ? null : r.name)}
                    className={cn("cursor-pointer border-t border-border/40 transition-colors hover:bg-muted/40", on && "bg-muted/60",
                      selected && !on && "opacity-50")}>
                    <td className="py-1.5 pr-2 whitespace-nowrap">
                      <span className="mr-1.5 inline-block size-2 rounded-full" style={{ background: COLORS[lead.includes(r.name) ? lead.indexOf(r.name) : 5] }} />
                      {r.name}
                      {r.position && <span className="ml-1.5 text-[10px] text-muted-foreground">{r.position}</span>}
                    </td>
                    <td className="py-1.5 pr-3">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-24 rounded-full bg-muted">
                          <div className="h-full rounded-full bg-sky-400" style={{ width: `${(100 * r.share) / maxShare}%` }} />
                        </div>
                        <span className="font-mono text-[11px] text-muted-foreground">{Math.round(100 * r.share)}%</span>
                      </div>
                    </td>
                    <td className="py-1.5 text-right font-mono">{r.targets}</td>
                    <td className="py-1.5 text-right font-mono">{Math.round((100 * r.catches) / r.targets)}%</td>
                    <td className="py-1.5 text-right font-mono">{r.yards}</td>
                    <td className="py-1.5 text-right font-mono">{r.adot == null ? "—" : r.adot.toFixed(1)}</td>
                    <td className="py-1.5 text-right font-mono">{r.td}</td>
                    <td className={cn("py-1.5 text-right font-mono", (r.epa ?? 0) > 0.02 ? "text-emerald-400" : (r.epa ?? 0) < -0.02 ? "text-rose-400" : "")}>
                      {signed(r.epa)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <div>
          <div className="mb-1 text-xs text-muted-foreground">Target share by week</div>
          <ChartContainer config={config} className="aspect-auto h-[230px] w-full">
            <BarChart data={weekly} margin={{ left: 0, right: 8 }}>
              <CartesianGrid vertical={false} strokeOpacity={0.15} />
              <XAxis dataKey="week" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} interval={weekly.length > 12 ? "preserveStartEnd" : 0} />
              <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11 }} width={38} domain={[0, 100]} ticks={[0, 25, 50, 75, 100]}
                allowDataOverflow tickFormatter={(v: number) => `${v}%`} />
              <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent formatter={(v, n) => `${n}: ${v}%`} />} />
              <ChartLegend content={<ChartLegendContent className="flex-wrap text-[11px]" />} />
              {[...lead, "Others"].map((n) => (
                <Bar key={n} isAnimationActive={animate} dataKey={n} stackId="s" fill={config[n].color} />
              ))}
            </BarChart>
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  )
}
