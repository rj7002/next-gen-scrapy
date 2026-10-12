import { useMemo } from "react"
import { RotateCcw } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { type FilterSpec, type Game, type Play, weekLabel } from "@/lib/api"
import { distinct, type FilterState, type FilterValue, hasData, numericRange, pretty } from "@/lib/filters"
import { cn } from "@/lib/utils"

interface Props {
  specs: FilterSpec[]
  groups: string[]
  plays: Play[]
  games: Game[]
  state: FilterState
  setState: (s: FilterState) => void
  weeks: Set<string> | null
  setWeeks: (w: Set<string> | null) => void
}

export function FilterPanel(p: Props) {
  const { specs, groups, plays, state, setState } = p
  const usable = useMemo(() => specs.filter((s) => s.col !== "season_type" && hasData(plays, s.col)), [specs, plays])
  const hidden = specs.length - usable.length
  const set = (col: string, v: FilterValue | null) => {
    const next = { ...state }
    if (v == null) delete next[col]
    else next[col] = v
    setState(next)
  }
  const activeIn = (g: string) => usable.filter((s) => s.group === g && state[s.col]).length
  const weekList = p.games.map((g) => String(g.week))

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-4 pt-4 pb-2">
        <span className="text-sm font-medium">Filters</span>
        {(Object.keys(state).length > 0 || p.weeks) && (
          <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs text-muted-foreground"
            onClick={() => (setState({}), p.setWeeks(null))}>
            <RotateCcw className="size-3" /> Reset
          </Button>
        )}
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-4 px-4 pb-6">
          <div>
            <Label className="mb-2 block text-xs text-muted-foreground">Weeks</Label>
            <div className="flex flex-wrap gap-1">
              {weekList.map((w) => {
                const on = !p.weeks || p.weeks.has(w)
                return (
                  <button key={w}
                    onClick={() => {
                      const cur = new Set(p.weeks ?? weekList)
                      if (p.weeks == null) {
                        p.setWeeks(new Set([w]))
                        return
                      }
                      if (cur.has(w)) cur.delete(w)
                      else cur.add(w)
                      p.setWeeks(cur.size === 0 || cur.size === weekList.length ? null : cur)
                    }}
                    className={cn("h-7 min-w-8 rounded-md border px-1.5 font-mono text-[11px] transition",
                      on && p.weeks ? "border-emerald-400/50 bg-emerald-400/15 text-emerald-700 dark:text-emerald-200"
                        : on ? "border-border bg-muted/40 text-foreground" : "border-border/50 text-muted-foreground/60")}>
                    {weekLabel(w)}
                  </button>
                )
              })}
            </div>
          </div>

          <Accordion type="multiple" defaultValue={["Situation", "Result"]} className="w-full">
            {groups.map((g) => {
              const items = usable.filter((s) => s.group === g)
              if (!items.length) return null
              const n = activeIn(g)
              return (
                <AccordionItem key={g} value={g} className="border-border/50">
                  <AccordionTrigger className="py-3 text-sm hover:no-underline">
                    <span className="flex items-center gap-2">
                      {g}
                      {n > 0 && <Badge className="h-5 rounded-full bg-emerald-400/20 px-1.5 text-[10px] text-emerald-700 dark:text-emerald-200">{n}</Badge>}
                    </span>
                  </AccordionTrigger>
                  <AccordionContent className="flex flex-col gap-4 pb-4">
                    {items.map((s) => (
                      <FilterControl key={s.col} spec={s} plays={plays} value={state[s.col]} onChange={(v) => set(s.col, v)} />
                    ))}
                  </AccordionContent>
                </AccordionItem>
              )
            })}
          </Accordion>
          {hidden > 0 && (
            <p className="text-xs text-muted-foreground">
              {hidden} more filters need data this season doesn't have (FTN charting starts in 2022; NGS participation
              is published after the season).
            </p>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

function FilterControl({ spec, plays, value, onChange }: {
  spec: FilterSpec
  plays: Play[]
  value: FilterValue | undefined
  onChange: (v: FilterValue | null) => void
}) {
  const head = (
    <div className="mb-1.5 flex items-center justify-between gap-2">
      <Label className="text-xs font-medium" title={spec.help ?? undefined}>{spec.label}</Label>
      {value && (
        <button className="text-[11px] text-muted-foreground hover:text-foreground" onClick={() => onChange(null)}>
          clear
        </button>
      )}
    </div>
  )

  if (spec.kind === "bool") {
    const v = value?.op === "bool" ? (value.value ? "yes" : "no") : "any"
    return (
      <div>
        {head}
        <ToggleGroup type="single" variant="outline" size="sm" value={v} className="w-full"
          onValueChange={(x) => onChange(!x || x === "any" ? null : { op: "bool", value: x === "yes" })}>
          {["any", "yes", "no"].map((o) => (
            <ToggleGroupItem key={o} value={o} className="flex-1 text-xs capitalize">
              {o}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
    )
  }

  if (spec.kind === "range") {
    const r = numericRange(plays, spec.col)
    if (!r || r[0] === r[1]) return null
    const step = spec.step ?? (r[1] - r[0] > 20 ? 1 : 0.5)
    const cur = value?.op === "range" ? [value.min ?? r[0], value.max ?? r[1]] : r
    return (
      <div>
        {head}
        <Slider min={r[0]} max={r[1]} step={step} value={cur}
          onValueChange={([lo, hi]) => onChange(lo <= r[0] && hi >= r[1] ? null : { op: "range", min: lo > r[0] ? lo : undefined, max: hi < r[1] ? hi : undefined })} />
        <div className="mt-1.5 flex justify-between font-mono text-[11px] text-muted-foreground">
          <span>{fmt(cur[0])}</span>
          <span>{fmt(cur[1])}</span>
        </div>
      </div>
    )
  }

  const options = spec.choices
    ? spec.choices.filter(([v]) => plays.some((p) => String(p[spec.col]) === String(v))).map(([v, l]) => ({ value: String(v), label: l }))
    : distinct(plays, spec.col).slice(0, 40).map((d) => ({ value: d.value, label: pretty(d.value) }))
  if (!options.length) return null

  if (spec.kind === "select" || options.length > 8) {
    const v = value?.op === "in" ? value.values[0] : "__any"
    return (
      <div>
        {head}
        <Select value={v} onValueChange={(x) => onChange(x === "__any" ? null : { op: "in", values: [x] })}>
          <SelectTrigger className="h-8 w-full text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__any">Any</SelectItem>
            {options.map((o) => (
              <SelectItem key={o.value} value={o.value} className="text-xs">
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    )
  }

  const sel = value?.op === "in" ? value.values : []
  return (
    <div>
      {head}
      <ToggleGroup type="multiple" variant="outline" size="sm" value={sel} className="flex w-full flex-wrap justify-start"
        onValueChange={(vals) => onChange(vals.length ? { op: "in", values: vals } : null)}>
        {options.map((o) => (
          <ToggleGroupItem key={o.value} value={o.value} className="h-7 flex-none px-2 text-xs">
            {o.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  )
}

const fmt = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(2))
