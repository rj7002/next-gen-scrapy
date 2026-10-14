import { Box, CircleDashed, Clipboard, Download, Flame, Image as ImageIcon, MoveUp, MoveVertical, Spline } from "lucide-react"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import type { LegendSpec } from "@/lib/exportChart"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Toggle } from "@/components/ui/toggle"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import type { Kind } from "@/lib/api"
import { type ColorBy, DOWN_COLORS, GAIN, OUTCOME, ROUTE_COLORS, ROUTE_LINE } from "@/lib/colors"
import { COMPARE_A, COMPARE_B, type HeatMetric, METRICS, splitOf, splitsFor, VALUE_RAMP } from "@/lib/heat"
import type { FieldView } from "./Field"

export type ExportFormat = "png" | "svg" | "clipboard"

export function FieldToolbar({ kind, view, setView, colorBy, setColorBy, heatOf, setHeatOf, heatMetric, setHeatMetric, heatSplit, setHeatSplit,
  yac, setYac, threeD, setThreeD, fullDepth, setFullDepth, deeper, onExport, undrawn, showUndrawn, setShowUndrawn }: {
  kind: Kind
  view: FieldView
  setView: (v: FieldView) => void
  colorBy: ColorBy
  setColorBy: (c: ColorBy) => void
  heatOf: "end" | "path"
  setHeatOf: (h: "end" | "path") => void
  heatMetric: HeatMetric
  setHeatMetric: (m: HeatMetric) => void
  heatSplit: string
  setHeatSplit: (s: string) => void
  yac: boolean
  setYac: (b: boolean) => void
  threeD: boolean
  setThreeD: (b: boolean) => void
  fullDepth: boolean
  setFullDepth: (b: boolean) => void
  deeper: number
  onExport: (view: FieldView, format: ExportFormat) => void
  undrawn: number
  showUndrawn: boolean
  setShowUndrawn: (b: boolean) => void
}) {
  const chart = kind === "pass" ? "passing chart" : kind === "route" ? "route chart" : "rushing chart"
  const modes: [ColorBy, string][] = [
    ["outcome", kind === "carry" ? "Gain" : "Outcome"],
    ["epa", "EPA"],
    ...(kind !== "carry" ? ([["route", "Route type"]] as [ColorBy, string][]) : []),
    ["down", "Down"],
    ["confidence", "Match confidence"],
  ]
  return (
    <div className="flex flex-wrap items-center gap-2">
      <ToggleGroup type="single" variant="outline" size="sm" value={view} onValueChange={(v) => v && setView(v as FieldView)}>
        <ToggleGroupItem value="plays" className="gap-1.5 px-2.5 text-xs">
          <Spline className="size-3.5" /> Plays
        </ToggleGroupItem>
        <ToggleGroupItem value="heat" className="gap-1.5 px-2.5 text-xs">
          <Flame className="size-3.5" /> Heatmap
        </ToggleGroupItem>
      </ToggleGroup>
      {view === "plays" ? (
        <Select value={colorBy} onValueChange={(v) => setColorBy(v as ColorBy)}>
          <SelectTrigger size="sm" className="h-8 w-[168px] text-xs">
            <span className="text-muted-foreground">Color:</span>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {modes.map(([v, l]) => (
              <SelectItem key={v} value={v} className="text-xs">
                {l}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <>
          <Select value={heatMetric} onValueChange={(v) => setHeatMetric(v as HeatMetric)}>
            <SelectTrigger size="sm" className="h-8 w-[150px] text-xs" title="What the heatmap shows: where plays go, how well they go there, or a comparison">
              <span className="text-muted-foreground">Heat:</span>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {METRICS[kind].map(([v, l]) => (
                <SelectItem key={v} value={v} className="text-xs">{l}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          {heatMetric === "compare" && (
            <Select value={splitOf(kind, heatSplit).key} onValueChange={setHeatSplit}>
              <SelectTrigger size="sm" className="h-8 w-[196px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {splitsFor(kind).map((sp) => (
                  <SelectItem key={sp.key} value={sp.key} className="text-xs">{sp.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {kind !== "pass" && (
            <ToggleGroup type="single" variant="outline" size="sm" value={heatOf} onValueChange={(v) => v && setHeatOf(v as "end" | "path")}>
              <ToggleGroupItem value="end" className="px-2.5 text-xs">{kind === "route" ? "Catch points" : "End points"}</ToggleGroupItem>
              <ToggleGroupItem value="path" className="px-2.5 text-xs">Whole paths</ToggleGroupItem>
            </ToggleGroup>
          )}
        </>
      )}
      {kind === "pass" && view === "plays" && (
        <Toggle variant="outline" size="sm" pressed={yac} onPressedChange={setYac} className="gap-1.5 px-2.5 text-xs"
          title="Yards after the catch: a line from each completion's catch point to where the play ended (straight downfield - play-by-play has no sideways spot)">
          <MoveUp className="size-3.5" /> YAC
        </Toggle>
      )}
      {(deeper > 0 || fullDepth) && (
        <Toggle variant="outline" size="sm" pressed={fullDepth} onPressedChange={setFullDepth} className="gap-1.5 px-2.5 text-xs"
          title="Show the whole depth of the field">
          <MoveVertical className="size-3.5" /> {fullDepth ? "Full depth" : `+${deeper} deeper`}
        </Toggle>
      )}
      {kind === "pass" && undrawn > 0 && (
        <Toggle variant="outline" size="sm" pressed={showUndrawn} onPressedChange={setShowUndrawn} className="gap-1.5 px-2.5 text-xs"
          title="Throws Next Gen Stats didn't draw (mostly incompletions), placed from play-by-play: air yards + direction">
          <CircleDashed className="size-3.5" /> Undrawn ({undrawn})
        </Toggle>
      )}
      <Toggle variant="outline" size="sm" pressed={threeD} onPressedChange={setThreeD} className="gap-1.5 px-2.5 text-xs">
        <Box className="size-3.5" /> 3D
      </Toggle>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm" className="h-8 gap-1.5 px-2.5 text-xs">
            <Download className="size-3.5" /> Export
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuLabel className="text-xs text-muted-foreground">Save as an image (2D, with your filters)</DropdownMenuLabel>
          <DropdownMenuItem onSelect={() => onExport("plays", "png")}><ImageIcon /> {cap(chart)} · PNG</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => onExport("heat", "png")}><Flame /> Heatmap · PNG</DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => onExport("plays", "svg")}><ImageIcon /> {cap(chart)} · SVG</DropdownMenuItem>
          <DropdownMenuItem onSelect={() => onExport("heat", "svg")}><Flame /> Heatmap · SVG</DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => onExport(view, "clipboard")}><Clipboard /> Copy {view === "heat" ? "heatmap" : chart} to clipboard</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1)

/** The legend for a view / colouring: shared by the on-screen legend and the exported image. */
export function legendSpec(kind: Kind, colorBy: ColorBy, view: FieldView, heat?: { metric: HeatMetric; split: string }): LegendSpec {
  if (view === "heat") {
    const m = heat?.metric ?? "volume"
    if (m === "compare") {
      const sp = splitOf(kind, heat!.split)
      return { gradient: [COMPARE_B, "#3f3f46", COMPARE_A], lo: `more ${sp.b}`, hi: `more ${sp.a}` }
    }
    if (m === "epa") return { gradient: VALUE_RAMP, lo: "−0.6 EPA", hi: "+0.6 EPA" }
    if (m === "cpoe") return { gradient: VALUE_RAMP, lo: "−25% CPOE", hi: "+25% CPOE" }
    if (m === "catch" || m === "success") return { gradient: VALUE_RAMP, lo: `below his ${m} rate`, hi: `above his ${m} rate` }
    return { gradient: ["#5b21b6", "#db2777", "#fde047"], lo: "fewer", hi: "more" }
  }
  if (colorBy === "epa") return { gradient: ["#ef4444", "#a1a1aa", "#10b981"], lo: "−3 EPA", hi: "+3 EPA" }
  if (colorBy === "confidence") return { gradient: ["#f59e0b", "#67c7a7", "#22d3ee"], lo: "unsure", hi: "certain" }
  let items: [string, string][]
  if (colorBy === "down") items = Object.entries(DOWN_COLORS).map(([d, c]) => [["", "1st", "2nd", "3rd", "4th"][Number(d)] + " down", c])
  else if (colorBy === "route") items = Object.entries(ROUTE_COLORS).map(([r, c]) => [cap(r.toLowerCase()), c])
  else if (kind === "pass") items = Object.entries(OUTCOME).map(([k, c]) => [cap(k.toLowerCase()), c])
  else if (kind === "carry") items = [["Loss", GAIN.LOSS], ["0–5 yds", GAIN.SHORT], ["5+ yds", GAIN.LONG], ["Touchdown", ROUTE_LINE.td]]
  else items = [["Complete", ROUTE_LINE.complete], ["Incomplete", ROUTE_LINE.incomplete], ["After catch", ROUTE_LINE.after], ["Touchdown", ROUTE_LINE.td]]
  return { items, dashed: true }
}

export function Legend({ kind, colorBy, view, heat, undrawn = false }: {
  kind: Kind; colorBy: ColorBy; view: FieldView; heat?: { metric: HeatMetric; split: string }; undrawn?: boolean
}) {
  const spec = legendSpec(kind, colorBy, view, heat)
  if ("gradient" in spec) return <GradientLegend from={spec.gradient[0]} via={spec.gradient[1]} to={spec.gradient[2]} lo={spec.lo} hi={spec.hi} />
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      {spec.items.map(([l, c]) => (
        <span key={l} className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-full" style={{ background: c }} />
          {l}
        </span>
      ))}
      <span className="inline-flex items-center gap-1.5">
        <span className="h-0 w-4 border-t border-dashed border-zinc-400" /> unmatched
      </span>
      {undrawn && (
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-full border border-dashed border-zinc-400" /> not drawn by NGS (estimated)
        </span>
      )}
    </div>
  )
}

function GradientLegend({ from, via, to, lo, hi }: { from: string; via: string; to: string; lo: string; hi: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span>{lo}</span>
      <span className="h-2 w-40 rounded-full" style={{ background: `linear-gradient(90deg, ${from}, ${via}, ${to})` }} />
      <span>{hi}</span>
    </div>
  )
}
