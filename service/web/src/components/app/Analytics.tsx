import { useEffect, useMemo, useState } from "react"
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ReferenceLine, Scatter, ScatterChart, XAxis, YAxis, ZAxis } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { type ChartConfig, ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { airYac, ALIGNS, type Align, alignment, boxCounts, downDistance, epaSpread, fieldBoundary, holes, OUTSIDE, pairs, reads,
  rollingForm } from "@/lib/analytics"
import { api, type Kind, type Play } from "@/lib/api"
import { epaColor, GAIN, OUTCOME } from "@/lib/colors"
import { useReducedMotion } from "@/lib/motion"
import { cn } from "@/lib/utils"
import { landing } from "./Field"

const POS = "#34d399"
const NEG = "#fb7185"
const axis = { tickLine: false, axisLine: false, tick: { fontSize: 11 } }
const signed = (v: number | null, d = 2) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(d)}`)

function Panel({ title, desc, children, className }: { title: string; desc?: string; children: React.ReactNode; className?: string }) {
  return (
    <Card className={cn("gap-3 border-border/60 bg-card/50 py-4", className)}>
      <CardHeader className="px-4">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {desc && <CardDescription className="text-xs">{desc}</CardDescription>}
      </CardHeader>
      <CardContent className="px-2">{children}</CardContent>
    </Card>
  )
}

/** The Analytics tab: plots beyond the field - form, the spread of outcomes, down & distance, field vs boundary, and
 * per-chart-kind ones (time to throw, YAC, reads, box counts, distance run). They follow the filters. A few use
 * columns the play rows leave out; those are fetched for this job when the tab opens. */
export function Analytics({ kind, plays, jobId }: { kind: Kind; plays: Play[]; jobId?: string }) {
  const animate = !useReducedMotion()
  const want = kind === "pass" ? ["air_epa", "yac_epa", "read_thrown"] : kind === "route" ? ["air_epa", "yac_epa"] : ["ngs_carrier_dist"]
  const [extra, setExtra] = useState<{ job?: string; cols: Record<string, Record<string, unknown>> }>({ cols: {} })
  useEffect(() => {
    if (!jobId || extra.job === jobId) return
    let cancelled = false
    api.columns(jobId, want).then((cols) => !cancelled && setExtra({ job: jobId, cols })).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [jobId, kind])
  const ps = useMemo(() => {
    const cols = extra.job === jobId ? Object.entries(extra.cols) : []
    if (!cols.length) return plays
    return plays.map((p) => ({ ...p, ...Object.fromEntries(cols.map(([c, v]) => [c, v[p.id] ?? null])) }))
  }, [plays, extra, jobId])

  const form = rollingForm(ps)
  // one tick per week, at that week's first play
  const weekTicks = form.points.filter((p, i, all) => i === 0 || all[i - 1].week !== p.week).map((p) => p.i)
  const win = Math.min(form.window, form.total)
  const spread = epaSpread(ps)
  const dd = downDistance(ps)
  const fb = fieldBoundary(ps, kind, landing)
  const noun = kind === "pass" ? "throws" : kind === "route" ? "targets" : "carries"
  const scale = kind === "carry" ? 10 : 6                       // EPA per play: rushing spans less than passing
  const cellColor = (v: number | null) => epaColor(v == null ? null : Math.max(-3, Math.min(3, v * scale)))

  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Panel title="Form" desc={`EPA per play and success rate over a rolling ${win}-play window, through the season`}>
        <ChartContainer config={{ epa: { label: "EPA / play", color: POS }, success: { label: "Success %", color: "#a78bfa" } }}
          className="aspect-auto h-[210px] w-full">
          <ComposedChart data={form.points} margin={{ left: 0, right: 8 }}>
            <CartesianGrid vertical={false} strokeOpacity={0.15} />
            <XAxis dataKey="i" type="number" domain={["dataMin", "dataMax"]} ticks={weekTicks} {...axis} minTickGap={14}
              tickFormatter={(i: number) => { const w = form.points.find((p) => p.i === i)?.week ?? ""; return w.startsWith("post") ? "P" : w }} />
            <YAxis yAxisId="epa" {...axis} width={38} />
            <YAxis yAxisId="suc" orientation="right" {...axis} width={34} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
            <ReferenceLine yAxisId="epa" y={0} strokeOpacity={0.3} />
            <ChartTooltip content={<ChartTooltipContent labelFormatter={(_, p) => `Play ${p?.[0]?.payload?.i} · ${p?.[0]?.payload?.label}`} />} />
            <Line yAxisId="epa" dataKey="epa" stroke="var(--color-epa)" dot={false} strokeWidth={2} isAnimationActive={animate} />
            <Line yAxisId="suc" dataKey="success" stroke="var(--color-success)" dot={false} strokeWidth={1.5} strokeDasharray="4 3" isAnimationActive={animate} />
            <ChartLegend content={<ChartLegendContent className="text-[11px]" />} />
          </ComposedChart>
        </ChartContainer>
      </Panel>

      <Panel title="Boom or bust" desc={`EPA per play: ${Math.round(100 * spread.boom)}% of plays +1 or better, ${Math.round(100 * spread.bust)}% −1 or worse (median ${signed(spread.median)})`}>
        <ChartContainer config={{ n: { label: "Plays" } }} className="aspect-auto h-[210px] w-full">
          <BarChart data={spread.bins} margin={{ left: 0, right: 8 }} barCategoryGap={1}>
            <CartesianGrid vertical={false} strokeOpacity={0.15} />
            <XAxis dataKey="label" {...axis} interval={1} />
            <YAxis {...axis} width={30} />
            <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent labelFormatter={(l) => `EPA ${l} to ${(Number(l) + 0.5).toFixed(1)}`} />} />
            <Bar dataKey="n" radius={2} isAnimationActive={animate}>
              {spread.bins.map((b) => <Cell key={b.label} fill={b.x >= 1 ? POS : b.x < -1 ? NEG : "#71717a"} fillOpacity={b.x >= 0 ? 0.95 : 0.8} />)}
            </Bar>
          </BarChart>
        </ChartContainer>
      </Panel>

      <Panel title="Down & distance" desc="EPA per play (colour) and how many plays, by down and yards to go">
        <div className="overflow-x-auto px-2">
          <table className="w-full min-w-[320px] border-separate border-spacing-1 text-xs">
            <thead>
              <tr className="text-muted-foreground">
                <th />
                {dd[0].cells.map((c) => <th key={c.label} className="font-medium">{c.label} to go</th>)}
              </tr>
            </thead>
            <tbody>
              {dd.map((r) => (
                <tr key={r.down}>
                  <td className="pr-2 text-right text-muted-foreground">{["", "1st", "2nd", "3rd", "4th"][r.down]}</td>
                  {r.cells.map((c) => (
                    <td key={c.label} className="rounded-md px-2 py-2 text-center"
                      style={{ background: c.n ? `${cellColor(c.epa)}33` : undefined, boxShadow: c.n ? `inset 0 0 0 1px ${cellColor(c.epa)}55` : undefined }}>
                      {c.n ? (
                        <>
                          <div className="font-mono text-sm font-medium" style={{ color: cellColor(c.epa) }}>{signed(c.epa)}</div>
                          <div className="text-[10px] text-muted-foreground">{c.n} · {Math.round(100 * (c.success ?? 0))}% succ</div>
                        </>
                      ) : <span className="text-muted-foreground/50">—</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Field vs boundary" desc={`${noun} to the wide side of the field, the middle, or the short side (from the hash the ball was on)`}>
        <ChartContainer config={{ share: { label: `Share of ${noun}`, color: "#38bdf8" } }} className="aspect-auto h-[170px] w-full">
          <BarChart data={fb.rows} layout="vertical" margin={{ left: 4, right: 12 }}>
            <XAxis type="number" hide domain={[0, "dataMax"]} />
            <YAxis type="category" dataKey="side" {...axis} width={70} />
            <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent formatter={(v, _n, item) =>
              `${v}% · ${item.payload.n} ${noun} · EPA ${signed(item.payload.epa)} · ${kind === "carry" ? `${item.payload.rate} yds/carry` : `${item.payload.rate}% caught`}`} />} />
            <Bar dataKey="share" radius={4} isAnimationActive={animate}>
              {fb.rows.map((r) => <Cell key={r.side} fill={epaColor(Math.max(-3, Math.min(3, r.epa * scale)))} />)}
            </Bar>
          </BarChart>
        </ChartContainer>
        <div className="flex flex-wrap gap-x-5 gap-y-1 px-2 text-xs text-muted-foreground">
          {fb.rows.map((r) => (
            <span key={r.side}>{r.side}: <span className="font-mono text-foreground">{r.share}%</span> · EPA <span className="font-mono"
              style={{ color: epaColor(Math.max(-3, Math.min(3, r.epa * scale))) }}>{signed(r.epa)}</span></span>
          ))}
          {fb.unknown > 0 && <span>{fb.unknown} without a hash or direction left out</span>}
        </div>
      </Panel>

      {kind === "pass" && <TimeToThrow plays={ps} animate={animate} />}
      {kind === "route" && <Alignment plays={ps} animate={animate} />}
      {kind === "carry" && <HoleMap plays={ps} />}
      {kind === "route" && <DepthYac plays={ps} animate={animate} />}
      {kind !== "carry" && <AirYac plays={ps} kind={kind} animate={animate} />}
      {kind === "pass" && <Reads plays={ps} animate={animate} />}
      {kind === "carry" && <Boxes plays={ps} animate={animate} />}
      {kind === "carry" && <DistanceRun plays={ps} animate={animate} />}
    </div>
  )
}

function TimeToThrow({ plays, animate }: { plays: Play[]; animate: boolean }) {
  const pts = pairs(plays, "time_to_throw", "air_yards")
  if (pts.length < 5) return null
  const by = Object.keys(OUTCOME).map((o) => [o, pts.filter((p) => p.outcome === o)] as const).filter(([, v]) => v.length)
  return (
    <Panel title="Time to throw vs depth" desc="How long he held it, how far it went - colour is the result">
      <ChartContainer config={Object.fromEntries(by.map(([o]) => [o, { label: o[0] + o.slice(1).toLowerCase(), color: OUTCOME[o] }])) satisfies ChartConfig}
        className="aspect-auto h-[240px] w-full">
        <ScatterChart margin={{ left: 0, right: 12 }}>
          <CartesianGrid strokeOpacity={0.12} />
          <XAxis type="number" dataKey="x" name="Time to throw" unit="s" {...axis} domain={["dataMin - 0.2", "dataMax + 0.2"]} tickFormatter={(v: number) => v.toFixed(1)} />
          <YAxis type="number" dataKey="y" name="Air yards" {...axis} width={32} />
          <ZAxis range={[22, 22]} />
          <ReferenceLine y={0} strokeOpacity={0.3} />
          <ReferenceLine x={2.5} strokeOpacity={0.25} strokeDasharray="3 3" label={{ value: "2.5s", fontSize: 10, fill: "#a1a1aa", position: "insideTopRight" }} />
          <ChartTooltip content={<ChartTooltipContent formatter={(v, n) => `${n}: ${Number(v).toFixed(1)}`} />} />
          {by.map(([o, v]) => <Scatter key={o} name={o} data={v} fill={OUTCOME[o]} fillOpacity={0.75} isAnimationActive={animate} />)}
          <ChartLegend content={<ChartLegendContent className="text-[11px]" />} />
        </ScatterChart>
      </ChartContainer>
    </Panel>
  )
}

function DepthYac({ plays, animate }: { plays: Play[]; animate: boolean }) {
  const pts = pairs(plays, "air_yards", "yards_after_catch", (p) => p.route_type === "COMPLETE")
  if (pts.length < 3) return null
  return (
    <Panel title="Depth vs yards after catch" desc="Where he caught it and how much he added after - each dot a catch">
      <ChartContainer config={{ y: { label: "YAC", color: POS } }} className="aspect-auto h-[240px] w-full">
        <ScatterChart margin={{ left: 0, right: 12 }}>
          <CartesianGrid strokeOpacity={0.12} />
          <XAxis type="number" dataKey="x" name="Air yards" {...axis} />
          <YAxis type="number" dataKey="y" name="YAC" {...axis} width={32} />
          <ZAxis range={[30, 30]} />
          <ChartTooltip content={<ChartTooltipContent formatter={(v, n) => `${n}: ${v}`} />} />
          <Scatter data={pts} fill={POS} fillOpacity={0.8} isAnimationActive={animate} />
        </ScatterChart>
      </ChartContainer>
    </Panel>
  )
}

function AirYac({ plays, kind, animate }: { plays: Play[]; kind: Kind; animate: boolean }) {
  const groups = airYac(plays, kind, (p) => (kind === "pass" ? (p.receiver_player_name as string) : (p.tree_route as string)) ?? null)
  if (!groups.length) return null
  return (
    <Panel title="Air vs YAC value" desc={kind === "pass"
      ? "On completions: EPA from the throw itself vs from the run after the catch, by receiver (totals)"
      : "On his catches: EPA from the throw itself vs from the run after the catch, by route (totals)"}>
      <ChartContainer config={{ air: { label: "Air EPA", color: "#38bdf8" }, yac: { label: "YAC EPA", color: "#f59e0b" } }}
        className="aspect-auto h-[240px] w-full">
        <BarChart data={groups} margin={{ left: 0, right: 8 }} stackOffset="sign">
          <CartesianGrid vertical={false} strokeOpacity={0.15} />
          <XAxis dataKey="name" {...axis} interval={0} tickFormatter={(s: string) => (s.length > 11 ? s.slice(0, 10) + "…" : /^[A-Z]+$/.test(s) ? s[0] + s.slice(1).toLowerCase() : s)} />
          <YAxis {...axis} width={34} />
          <ReferenceLine y={0} strokeOpacity={0.3} />
          <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent labelFormatter={(l, p) => `${l} · ${p?.[0]?.payload?.n} catches`} />} />
          <Bar dataKey="air" stackId="v" fill="var(--color-air)" isAnimationActive={animate} />
          <Bar dataKey="yac" stackId="v" fill="var(--color-yac)" radius={[3, 3, 0, 0]} isAnimationActive={animate} />
          <ChartLegend content={<ChartLegendContent className="text-[11px]" />} />
        </BarChart>
      </ChartContainer>
    </Panel>
  )
}

function Reads({ plays, animate }: { plays: Play[]; animate: boolean }) {
  const rows = reads(plays)
  if (!rows.length) return null
  return (
    <Panel title="Reads" desc="Which read the ball went to (FTN charting): share of throws, colour is EPA per play">
      <ChartContainer config={{ share: { label: "Share of throws" } }} className="aspect-auto h-[190px] w-full">
        <BarChart data={rows} layout="vertical" margin={{ left: 4, right: 12 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="label" {...axis} width={94} />
          <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent formatter={(v, _n, item) =>
            `${v}% · ${item.payload.n} throws · ${item.payload.comp}% complete · EPA ${signed(item.payload.epa)}`} />} />
          <Bar dataKey="share" radius={4} isAnimationActive={animate}>
            {rows.map((r) => <Cell key={r.label} fill={epaColor(Math.max(-3, Math.min(3, r.epa * 6)))} />)}
          </Bar>
        </BarChart>
      </ChartContainer>
    </Panel>
  )
}

function Boxes({ plays, animate }: { plays: Play[]; animate: boolean }) {
  const rows = boxCounts(plays)
  if (!rows.length) return null
  return (
    <Panel title="Box count" desc="Yards per carry (bars) and success rate (line) by defenders in the box">
      <ChartContainer config={{ ypc: { label: "Yds / carry", color: POS }, success: { label: "Success %", color: "#a78bfa" } }}
        className="aspect-auto h-[220px] w-full">
        <ComposedChart data={rows} margin={{ left: 0, right: 8 }}>
          <CartesianGrid vertical={false} strokeOpacity={0.15} />
          <XAxis dataKey="label" {...axis} tickFormatter={(l: string) => `${l} in box`} />
          <YAxis yAxisId="y" {...axis} width={30} />
          <YAxis yAxisId="s" orientation="right" {...axis} width={34} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
          <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent labelFormatter={(l, p) => `${l} in the box · ${p?.[0]?.payload?.n} carries`} />} />
          <Bar yAxisId="y" dataKey="ypc" fill="var(--color-ypc)" radius={4} isAnimationActive={animate} />
          <Line yAxisId="s" dataKey="success" stroke="var(--color-success)" strokeWidth={2} isAnimationActive={animate} />
          <ChartLegend content={<ChartLegendContent className="text-[11px]" />} />
        </ComposedChart>
      </ChartContainer>
    </Panel>
  )
}

function DistanceRun({ plays, animate }: { plays: Play[]; animate: boolean }) {
  const pts = pairs(plays, "ngs_carrier_dist", "yards_gained")
  if (pts.length < 5) return null
  const ratio = pts.reduce((a, p) => a + p.x, 0) / Math.max(1, pts.reduce((a, p) => a + Math.max(0, p.y), 0))
  const by = Object.keys(GAIN).map((g) => [g, pts.filter((p) => p.outcome === g)] as const).filter(([, v]) => v.length)
  return (
    <Panel title="Distance run vs yards gained" desc={`NGS distance the ball carrier ran for each yard gained: ${ratio.toFixed(1)} - lower is more north-south`}>
      <ChartContainer config={Object.fromEntries(by.map(([g]) => [g, { label: { LOSS: "Loss", SHORT: "0-5 yds", LONG: "5+ yds" }[g] ?? g, color: GAIN[g] }])) satisfies ChartConfig}
        className="aspect-auto h-[240px] w-full">
        <ScatterChart margin={{ left: 0, right: 12 }}>
          <CartesianGrid strokeOpacity={0.12} />
          <XAxis type="number" dataKey="x" name="Distance run" unit=" yds" {...axis} />
          <YAxis type="number" dataKey="y" name="Yards gained" {...axis} width={32} />
          <ZAxis range={[22, 22]} />
          <ReferenceLine y={0} strokeOpacity={0.3} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 60, y: 60 }]} strokeOpacity={0.25} strokeDasharray="4 3" />
          <ChartTooltip content={<ChartTooltipContent formatter={(v, n) => `${n}: ${Number(v).toFixed(1)}`} />} />
          {by.map(([g, v]) => <Scatter key={g} name={g} data={v} fill={GAIN[g]} fillOpacity={0.75} isAnimationActive={animate} />)}
          <ChartLegend content={<ChartLegendContent className="text-[11px]" />} />
        </ScatterChart>
      </ChartContainer>
    </Panel>
  )
}


const ALIGN_COLOR: Record<Align, string> = { wide: "#38bdf8", slot: "#a78bfa", inline: "#f59e0b", backfield: "#34d399" }
const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(100 * v)}%`)

function Alignment({ plays, animate }: { plays: Play[]; animate: boolean }) {
  const a = alignment(plays)
  if (a.known < 5) {
    return (
      <Panel title="Alignment" desc="Where he lined up on his targets">
        <p className="px-2 text-xs text-muted-foreground">
          {a.known ? "Too few targets with a clean route start." : "Needs the ball's hash (FTN charting, 2022 on) and a cleanly traced route start."}
        </p>
      </Panel>
    )
  }
  return (
    <Panel title="Alignment" className="md:col-span-2"
      desc={`Where he lined up on his ${a.known} targets, from where each route starts against the ball (targeted routes only, not every snap)`}>
      <div className="grid gap-4 px-2 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-[11px] text-muted-foreground">
              <th className="pb-2 font-medium">Lined up</th>
              <th className="pb-2 font-medium">Share of targets</th>
              <th className="pb-2 text-right font-medium">Tgt</th>
              <th className="pb-2 text-right font-medium">EPA / tgt</th>
              <th className="pb-2 text-right font-medium">Catch %</th>
              <th className="pb-2 text-right font-medium">aDOT</th>
            </tr>
          </thead>
          <tbody>
            {a.rows.map((r) => {
              return (
                <tr key={r.key} className="border-t border-border/40">
                  <td className="py-2 pr-2 whitespace-nowrap">
                    <span className="mr-1.5 inline-block size-2 rounded-full" style={{ background: ALIGN_COLOR[r.key] }} />{r.label}
                  </td>
                  <td className="py-2 pr-3">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-28 rounded-full bg-muted">
                        <div className="h-full rounded-full" style={{ width: `${100 * r.share}%`, background: ALIGN_COLOR[r.key] }} />
                      </div>
                      <span className="font-mono text-[11px] text-muted-foreground">{pct(r.share)}</span>
                    </div>
                  </td>
                  <td className="py-2 text-right font-mono">{r.n}</td>
                  <td className="py-2 text-right font-mono">
                    <span style={{ color: r.epa == null ? undefined : epaColor(Math.max(-3, Math.min(3, r.epa * 6))) }}>{r.n ? signed(r.epa) : "—"}</span>
                  </td>
                  <td className="py-2 text-right font-mono">{r.n ? pct(r.catch) : "—"}</td>
                  <td className="py-2 text-right font-mono">{r.adot == null ? "—" : r.adot.toFixed(1)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div>
          <div className="mb-1 text-xs text-muted-foreground">By week (share of his targets)</div>
          <ChartContainer config={Object.fromEntries(ALIGNS.map((k) => [k, { label: k === "slot" ? "Slot" : k[0].toUpperCase() + k.slice(1), color: ALIGN_COLOR[k] }])) satisfies ChartConfig}
            className="aspect-auto h-[200px] w-full">
            <BarChart data={a.weekly} margin={{ left: 0, right: 8 }}>
              <CartesianGrid vertical={false} strokeOpacity={0.15} />
              <XAxis dataKey="week" {...axis} interval={a.weekly.length > 12 ? "preserveStartEnd" : 0} />
              <YAxis {...axis} width={42} domain={[0, 100]} ticks={[0, 50, 100]} tickFormatter={(v: number) => `${v}%`} />
              <ChartTooltip cursor={{ fillOpacity: 0.06 }} content={<ChartTooltipContent labelFormatter={(l, p) => `${l} · ${p?.[0]?.payload?.n} targets`} />} />
              {ALIGNS.map((k) => <Bar key={k} dataKey={k} stackId="a" fill={ALIGN_COLOR[k]} isAnimationActive={animate} />)}
              <ChartLegend content={<ChartLegendContent className="text-[11px]" />} />
            </BarChart>
          </ChartContainer>
        </div>
      </div>
      <p className="mt-2 px-2 text-[11px] text-muted-foreground">
        Slot / tight split = 4.5-10 yds from the ball; wide = farther; inline = closer; backfield = starting 2.5+ yds behind the line.
        An outside receiver in a tight split counts as slot here - the charts don't show who lined up outside him.
        {a.unknown > 0 && ` ${a.unknown} targets without a hash or a clean start are left out.`}
      </p>
    </Panel>
  )
}

/** the line of scrimmage, seen from behind the offense: where his runs crossed it */
function HoleMap({ plays }: { plays: Play[] }) {
  const h = holes(plays)
  if (h.runs < 5) {
    return (
      <Panel title="Hole map" desc="Where his runs crossed the line of scrimmage">
        <p className="px-2 text-xs text-muted-foreground">Too few runs with a traced line and the ball's spot.</p>
      </Panel>
    )
  }
  const W = 640, H = 210, X0 = 30, X1 = W - 30, base = 150, top = 22
  const lo = -15, hi = 15
  const sx = (x: number) => X0 + ((x - lo) / (hi - lo)) * (X1 - X0)
  const view = h.bins.filter((b) => b.x >= lo && b.x < hi)
  const edgeL = h.bins.filter((b) => b.x < lo), edgeR = h.bins.filter((b) => b.x >= hi)
  const max = Math.max(...view.map((b) => b.share), 0.01)
  const sy = (v: number) => base - (v / max) * (base - top)
  const colorOf = (e: number | null) => epaColor(e == null ? null : Math.max(-3, Math.min(3, e * 10)))
  const OL: [number, string][] = [[-2.9, "LT"], [-1.45, "LG"], [0, "C"], [1.45, "RG"], [2.9, "RT"]]
  const side = (n: number) => edgeL.concat(edgeR).some((b) => b.n > 0) && n
  return (
    <Panel title="Hole map" className="md:col-span-2"
      desc={`Where his ${h.runs} runs crossed the line of scrimmage, yards from the ball - bar height is the share of his runs, colour is EPA per run`}>
      <div className="px-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto w-full max-w-[820px]" role="img" aria-label="Hole map">
          <rect x={sx(-OUTSIDE)} y={top - 8} width={sx(OUTSIDE) - sx(-OUTSIDE)} height={base - top + 8} className="fill-muted/40" rx={4} />
          <text x={sx(0)} y={top - 11} textAnchor="middle" className="fill-muted-foreground text-[10px]">inside the tackles</text>
          <text x={sx(-10)} y={top - 11} textAnchor="middle" className="fill-muted-foreground text-[10px]">outside left</text>
          <text x={sx(10)} y={top - 11} textAnchor="middle" className="fill-muted-foreground text-[10px]">outside right</text>
          {view.map((b) => {
            const x = sx(b.x), w = sx(b.x + 1) - x - 1.5
            return b.n > 0 && (
              <rect key={b.x} x={x + 0.75} y={sy(b.share)} width={w} height={base - sy(b.share)} rx={2} fill={colorOf(b.epa)}
                fillOpacity={Math.min(1, 0.45 + b.n / 12)}>
                <title>{`${b.x} to ${b.x + 1} yds: ${b.n} runs (${pct(b.share)}), EPA ${signed(b.epa)}`}</title>
              </rect>
            )
          })}
          <line x1={X0} x2={X1} y1={base} y2={base} className="stroke-sky-400" strokeWidth={2} />
          {OL.map(([x, n]) => (
            <g key={n}>
              <circle cx={sx(x)} cy={base + 16} r={9} className="fill-slate-600 stroke-card" strokeWidth={1.5} />
              <text x={sx(x)} y={base + 19.5} textAnchor="middle" className="fill-white text-[8px] font-semibold">{n}</text>
            </g>
          ))}
          {[-15, -10, -5, 0, 5, 10, 15].map((t) => (
            <text key={t} x={sx(t)} y={H - 6} textAnchor="middle" className="fill-muted-foreground font-mono text-[10px]">{t > 0 ? `+${t}` : t}</text>
          ))}
          {side(1) ? <text x={X1} y={base + 34} textAnchor="end" className="fill-muted-foreground text-[9px]">
            {edgeL.reduce((a, b) => a + b.n, 0) + edgeR.reduce((a, b) => a + b.n, 0)} beyond ±15</text> : null}
        </svg>
        <div className="mt-1 grid gap-x-6 gap-y-2 text-xs sm:grid-cols-3">
          <div><div className="text-muted-foreground">Outside the tackles</div>
            <div className="font-mono text-sm">{pct(h.outside)}</div></div>
          <div><div className="text-muted-foreground">Inside: EPA / yds per run</div>
            <div className="font-mono text-sm">{signed(h.inside.epa)} / {h.inside.ypc?.toFixed(1) ?? "—"} <span className="text-[11px] text-muted-foreground">· {h.inside.n} runs</span></div></div>
          <div><div className="text-muted-foreground">Outside: EPA / yds per run</div>
            <div className="font-mono text-sm">{signed(h.out.epa)} / {h.out.ypc?.toFixed(1) ?? "—"} <span className="text-[11px] text-muted-foreground">· {h.out.n} runs</span></div></div>
        </div>
        <div className="mt-3 text-xs">
          <div className="mb-1 text-muted-foreground">Called vs where he hit - median yards from the ball, by the gap play-by-play recorded</div>
          <div className="flex flex-wrap gap-2">
            {h.byGap.filter((g) => g.n > 0).map((g) => (
              <span key={g.gap} className="rounded-md border border-border/60 px-2 py-1">
                <span className="capitalize">{g.gap}</span>{" "}
                <span className="font-mono">{g.median?.toFixed(1)}</span>
                <span className="text-muted-foreground"> · {g.n}</span>
              </span>
            ))}
          </div>
        </div>
        {h.missing > 0 && <p className="mt-2 text-[11px] text-muted-foreground">
          {h.missing} runs left out: no traced line across the line of scrimmage, or no spot for the ball (before 2022 only runs from
          under center have one).</p>}
      </div>
    </Panel>
  )
}
