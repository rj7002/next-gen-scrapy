import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { type ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { useReducedMotion } from "@/lib/motion"
import type { Kind, Play } from "@/lib/api"
import { byWeek, coverage, depth, direction, gains } from "@/lib/stats"
import { cn } from "@/lib/utils"
import { ReceiversPanel } from "./Receivers"
import { RouteTreePanel } from "./RouteTree"
import { RunGapsPanel } from "./RunGaps"

const POS = "#34d399"
const NEG = "#fb7185"

function Panel({ title, desc, children }: { title: string; desc?: string; children: React.ReactNode }) {
  return (
    <Card className="gap-3 border-border/60 bg-card/50 py-4">
      <CardHeader className="px-4">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {desc && <CardDescription className="text-xs">{desc}</CardDescription>}
      </CardHeader>
      <CardContent className="px-2">{children}</CardContent>
    </Card>
  )
}

const axis = { tickLine: false, axisLine: false, tick: { fontSize: 11 } }

function EpaByWeek({ kind, plays }: { kind: Kind; plays: Play[] }) {
  const animate = !useReducedMotion()
  const data = byWeek(kind, plays)
  const config = { epa: { label: "EPA / play" } } satisfies ChartConfig
  return (
    <Panel title="Week by week" desc="EPA per play">
      <ChartContainer config={config} className="aspect-auto h-[190px] w-full">
        <BarChart data={data} margin={{ left: 0, right: 8 }}>
          <CartesianGrid vertical={false} strokeOpacity={0.15} />
          <XAxis dataKey="week" {...axis} interval={data.length > 12 ? "preserveStartEnd" : 0} minTickGap={2} tickFormatter={(w: string) => w.replace(/^W/, "")} />
          <YAxis {...axis} width={34} />
          <ReferenceLine y={0} strokeOpacity={0.3} />
          <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent labelFormatter={(l, p) => `${l} ${p?.[0]?.payload?.opponent ? "vs " + p[0].payload.opponent : ""}`} />} />
          <Bar isAnimationActive={animate} dataKey="epa" radius={4}>
            {data.map((d) => (
              <Cell key={d.week} fill={d.epa >= 0 ? POS : NEG} />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
    </Panel>
  )
}

function Depth({ kind, plays }: { kind: Kind; plays: Play[] }) {
  const animate = !useReducedMotion()
  const data = depth(kind, plays)
  const config = {
    attempts: { label: kind === "route" ? "Targets" : "Attempts", color: "#52525b" },
    caught: { label: kind === "route" ? "Catches" : "Completions", color: POS },
  } satisfies ChartConfig
  return (
    <Panel title="By depth" desc="Air yards at the catch / target">
      <ChartContainer config={config} className="aspect-auto h-[190px] w-full">
        <BarChart data={data} margin={{ left: 0, right: 8 }}>
          <CartesianGrid vertical={false} strokeOpacity={0.15} />
          <XAxis dataKey="band" {...axis} />
          <YAxis {...axis} width={30} />
          <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent />} />
          <Bar isAnimationActive={animate} dataKey="attempts" fill="var(--color-attempts)" radius={4} />
          <Bar isAnimationActive={animate} dataKey="caught" fill="var(--color-caught)" radius={4} />
        </BarChart>
      </ChartContainer>
    </Panel>
  )
}

function Direction({ kind, plays }: { kind: Kind; plays: Play[] }) {
  const animate = !useReducedMotion()
  const data = direction(kind, plays)
  const key = kind === "carry" ? "yards" : "epa"
  const config = { [key]: { label: kind === "carry" ? "Yards / carry" : "EPA / play" } } satisfies ChartConfig
  return (
    <Panel title="Direction" desc={kind === "carry" ? "Yards per carry by run direction" : "EPA by pass direction"}>
      <ChartContainer config={config} className="aspect-auto h-[190px] w-full">
        <BarChart data={data} margin={{ left: 0, right: 8 }}>
          <CartesianGrid vertical={false} strokeOpacity={0.15} />
          <XAxis dataKey="side" {...axis} />
          <YAxis {...axis} width={34} />
          <ReferenceLine y={0} strokeOpacity={0.3} />
          <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent />} />
          <Bar isAnimationActive={animate} dataKey={key} radius={4}>
            {data.map((d) => (
              <Cell key={d.side} fill={(d as any)[key] >= 0 ? (kind === "carry" ? "#38bdf8" : POS) : NEG} />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
    </Panel>
  )
}

function Gains({ plays }: { plays: Play[] }) {
  const animate = !useReducedMotion()
  const data = gains(plays)
  const config = { carries: { label: "Carries" } } satisfies ChartConfig
  const tone = (b: string) => (b === "Loss" ? "#f87171" : b === "0" || b === "1–2" || b === "3–4" ? "#facc15" : "#34d399")
  return (
    <Panel title="Gain distribution" desc="Yards gained per carry">
      <ChartContainer config={config} className="aspect-auto h-[190px] w-full">
        <BarChart data={data} margin={{ left: 0, right: 8 }}>
          <CartesianGrid vertical={false} strokeOpacity={0.15} />
          <XAxis dataKey="bin" {...axis} />
          <YAxis {...axis} width={30} />
          <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent />} />
          <Bar isAnimationActive={animate} dataKey="carries" radius={4}>
            {data.map((d) => (
              <Cell key={d.bin} fill={tone(d.bin)} />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
    </Panel>
  )
}

function Coverage({ plays }: { plays: Play[] }) {
  const animate = !useReducedMotion()
  const data = coverage(plays)
  if (!data.length) return null
  const config = { epa: { label: "EPA / play" } } satisfies ChartConfig
  return (
    <Panel title="Vs coverage" desc="EPA per play by the defense's coverage">
      <ChartContainer config={config} className="aspect-auto h-[190px] w-full">
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 12 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="coverage" {...axis} width={86} interval={0} />
          <ReferenceLine x={0} strokeOpacity={0.3} />
          <ChartTooltip cursor={{ fillOpacity: 0.06 }}
            content={<ChartTooltipContent formatter={(v, _n, item) => `${Number(v) >= 0 ? "+" : ""}${v} EPA · ${item.payload.plays} plays`} />} />
          <Bar isAnimationActive={animate} dataKey="epa" radius={4}>
            {data.map((d) => (
              <Cell key={d.coverage} fill={d.epa >= 0 ? POS : NEG} />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
    </Panel>
  )
}

/** fieldPlays: what's drawn on the field (no undrawn throws unless they're shown) - the route tree and run gaps
 * count exactly those */
export function Insights({ kind, plays, fieldPlays, treePlays, route, onRoute, gap, onGap, receiverPlays, receiver, onReceiver }: {
  kind: Kind; plays: Play[]; fieldPlays: Play[]
  treePlays?: Play[]; route?: string | null; onRoute?: (r: string | null) => void; gap?: string | null; onGap?: (g: string | null) => void
  receiverPlays?: Play[]; receiver?: string | null; onReceiver?: (name: string | null) => void
}) {
  const panels = [
    kind === "carry" && <Gains key="gains" plays={plays} />,
    kind !== "carry" && <Depth key="depth" kind={kind} plays={plays} />,
    <EpaByWeek key="week" kind={kind} plays={plays} />,
    <Direction key="dir" kind={kind} plays={plays} />,
    kind !== "carry" && coverage(plays).length > 0 && <Coverage key="cov" plays={plays} />,
  ].filter(Boolean)
  return (
    <div className="flex flex-col gap-3">
    {kind === "carry" ? <RunGapsPanel plays={treePlays ?? fieldPlays} selected={gap} onPick={onGap} />
      : <RouteTreePanel kind={kind} plays={treePlays ?? fieldPlays} selected={route} onPick={onRoute} />}
    {kind === "pass" && onReceiver && <ReceiversPanel plays={receiverPlays ?? plays} selected={receiver ?? null} onPick={onReceiver} />}
    <div className={cn("grid gap-3 md:grid-cols-2", panels.length % 3 === 0 && "xl:grid-cols-3", panels.length === 5 && "xl:grid-cols-6 xl:[&>*:nth-child(-n+3)]:col-span-2 xl:[&>*:nth-child(n+4)]:col-span-3")}>
      {panels}
    </div>
    </div>
  )
}
