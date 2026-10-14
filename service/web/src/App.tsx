import { useEffect, useMemo, useRef, useState } from "react"
import { Clock, MousePointerClick, SlidersHorizontal, X } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toaster } from "@/components/ui/sonner"
import { AskBar } from "@/components/app/AskBar"
import { Field, type FieldHandle, type FieldView } from "@/components/app/Field"
import { type ExportFormat, FieldToolbar, Legend, legendSpec } from "@/components/app/FieldToolbar"
import { FilterPanel } from "@/components/app/FilterPanel"
import { Header } from "@/components/app/Header"
import { Insights } from "@/components/app/Insights"
import { KpiGrid } from "@/components/app/Kpis"
import { Landing } from "@/components/app/Landing"
import { PlayPanel } from "@/components/app/PlayPanel"
import { LoadingCard, PlayerHero } from "@/components/app/PlayerHero"
import { PlayerSearch, primaryKind } from "@/components/app/PlayerSearch"
import { PlayTable } from "@/components/app/PlayTable"
import { api, type Dataset, type FilterSpec, type Job, KIND_NOUN, type Kind, kindFromDataset, type Player, weekLabel } from "@/lib/api"
import type { ColorBy } from "@/lib/colors"
import { exportChart } from "@/lib/exportChart"
import { applyFilters, describe, type FilterState, fromParams } from "@/lib/filters"
import { answerJSON, modelStatus, preload, webgpuSupported } from "@/lib/llm"

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

interface Selection {
  player: Player
  kind: Kind
  season: number
}

export default function App() {
  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem("theme") !== "light"
    } catch {
      return true
    }
  })
  const [seasons, setSeasons] = useState<number[]>([])
  const [season, setSeason] = useState<number | null>(null)
  const [players, setPlayers] = useState<Player[]>([])
  const [searchOpen, setSearchOpen] = useState(false)
  const [sel, setSel] = useState<Selection | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [data, setData] = useState<Dataset | null>(null)
  const [specs, setSpecs] = useState<{ groups: string[]; filters: FilterSpec[] } | null>(null)
  const [filters, setFilters] = useState<FilterState>({})
  const [weeks, setWeeks] = useState<Set<string> | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const url0 = useRef(new URLSearchParams(window.location.search))
  const [colorBy, setColorBy] = useState<ColorBy>(() => (url0.current.get("color") as ColorBy) || "outcome")
  const [view, setView] = useState<FieldView>(() => (url0.current.get("view") as FieldView) || "plays")
  const [heatOf, setHeatOf] = useState<"end" | "path">(() => (url0.current.get("heat") as "end" | "path") || "end")
  const [threeD, setThreeD] = useState(() => url0.current.get("3d") === "1")
  // incompletions NGS didn't draw: always counted in the stats, drawn on the field only when asked for
  const [showUndrawn, setShowUndrawn] = useState(() => url0.current.get("undrawn") === "1")
  const [fullDepth, setFullDepth] = useState(false)
  const [deeper, setDeeper] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [asking, setAsking] = useState(false)
  // the Ask bar's in-browser model: on desktops it loads in the background once the page has settled (phones wait
  // until someone taps the Ask bar, and data-saver connections never preload)
  useEffect(() => {
    const conn = (navigator as any).connection
    if (!webgpuSupported() || /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent) || conn?.saveData) return
    const t = setTimeout(() => preload(), 4000)
    return () => clearTimeout(t)
  }, [])
  const pending = useRef<Record<string, string> | null>(null)

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    try {
      localStorage.setItem("theme", dark ? "dark" : "light")
    } catch {
      /* private mode */
    }
  }, [dark])

  // deep links: ?player=<esb id>&kind=pass|route|carry&season=2024
  const deepLink = useRef<{ player: string; kind: Kind | null; season: number } | null>(null)
  useEffect(() => {
    const q = new URLSearchParams(window.location.search)
    const player = q.get("player")
    api.seasons().then((r) => {
      setSeasons(r.seasons)
      const s = Number(q.get("season")) || r.current
      if (player) deepLink.current = { player, kind: (q.get("kind") as Kind) || null, season: s }
      setSeason((cur) => cur ?? s)
    }).catch((e) => toast.error("Couldn't reach the server", { description: String(e.message ?? e) }))
  }, [])

  useEffect(() => {
    const d = deepLink.current
    if (!d || season !== d.season || !players.length) return
    deepLink.current = null
    const p = players.find((x) => x.esb_id === d.player)
    if (p) setSel({ player: p, kind: d.kind && p.kinds[d.kind] ? d.kind : primaryKind(p), season: d.season })
  }, [players, season])

  useEffect(() => {
    if (!sel) return window.history.replaceState(null, "", window.location.pathname)
    const q = new URLSearchParams({ player: sel.player.esb_id, kind: sel.kind, season: String(sel.season) })
    if (view !== "plays") q.set("view", view)
    if (view === "heat" && heatOf !== "end") q.set("heat", heatOf)
    if (colorBy !== "outcome") q.set("color", colorBy)
    if (threeD) q.set("3d", "1")
    if (showUndrawn) q.set("undrawn", "1")
    window.history.replaceState(null, "", `?${q}`)
  }, [sel, view, heatOf, colorBy, threeD, showUndrawn])

  useEffect(() => {
    if (season == null) return
    api.players(season).then((r) => setPlayers(r.players)).catch((e) => toast.error("Couldn't reach Next Gen Stats", { description: e.message }))
  }, [season])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setSearchOpen((o) => !o)
      }
      if (e.key === "Escape") setSelectedId(null)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  // the live pipeline: start (or reuse) a job, poll it, pull each new version of the data
  useEffect(() => {
    if (!sel) return
    let cancelled = false
    setJob(null)
    setData(null)
    setSelectedId(null)
    ;(async () => {
      let last = 0
      try {
        let j = await api.load(sel.kind, sel.player.esb_id, sel.season)
        while (!cancelled) {
          setJob(j)
          if (j.version > last) {
            const d = await api.data(j.id)
            if (cancelled) return
            if (last > 0) toast.success("Passes re-matched with the receivers' route charts", { description: d.notes.at(-1) })
            last = j.version
            setData(d)
          }
          if (j.status === "done" || j.status === "error") break
          await sleep(last ? 1500 : 700)
          j = await api.job(j.id)
        }
        if (j.status === "error") toast.error(`Couldn't load ${sel.player.name}`, { description: j.error ?? undefined })
      } catch (e: any) {
        if (!cancelled) toast.error(`Couldn't load ${sel.player.name}`, { description: e.message })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [sel])

  useEffect(() => {
    if (!sel) return
    api.filters(sel.kind).then(setSpecs)
  }, [sel?.kind])

  // filters from a question asked before the data arrived
  useEffect(() => {
    if (data && specs && pending.current) {
      setFilters(fromParams(pending.current, specs.filters))
      pending.current = null
    }
  }, [data, specs])

  const pick = (player: Player, kind: Kind, s: number) => {
    setSearchOpen(false)
    setFilters({})
    setWeeks(null)
    if (colorBy === "route" && kind !== "route") setColorBy("outcome")
    setSel({ player, kind, season: s })
  }

  async function ask(q: string) {
    if (season == null) return
    setAsking(true)
    try {
      const kind = sel?.kind ?? "pass"
      // with the in-browser model loaded it reads the leftover words; otherwise the server does (Gemini, if set up)
      let r = await api.ask(q, kind, season, job?.id, modelStatus().state === "ready" ? "client" : "server")
      if (r.refused) {
        // another player named while this chart is open: say so and leave the chart as it is
        toast.error("Stayed on this chart", { description: r.refused })
        return
      }
      if (r.llm_prompt && r.llm_schema) {
        // words the rules couldn't place: the in-browser model reads them (if it's loaded), the server validates
        const answer = await answerJSON(r.llm_prompt, r.llm_schema, r.llm_messages, r.llm_ft).catch(() => null)
        if (answer) r = await api.refine(q, kind, season, answer, job?.id)
      }
      const targetKind = kindFromDataset(r.dataset) ?? kind
      const targetSeason = r.seasons?.[0] ?? sel?.season ?? season
      const meaning = r.understood.map((u) => u.meaning).filter(Boolean)
      if (r.esb_id && (r.esb_id !== sel?.player.esb_id || targetSeason !== sel?.season || targetKind !== sel?.kind)) {
        let list = players
        if (targetSeason !== season) {
          list = (await api.players(targetSeason)).players
          setSeason(targetSeason)
          setPlayers(list)
        }
        const p = list.find((x) => x.esb_id === r.esb_id)
        if (!p) throw new Error(`${r.name} has no ${targetSeason} charts`)
        const k = p.kinds[targetKind] ? targetKind : primaryKind(p)
        pending.current = r.params
        setSearchOpen(false)
        setWeeks(r.weeks?.length ? new Set(r.weeks.map(String)) : null)
        setSel({ player: p, kind: k, season: targetSeason })
      } else if (!sel) {
        toast("Name a player in the question", { description: "e.g. “Josh Allen 3rd down throws under pressure”" })
        return
      } else {
        setFilters(fromParams(r.params, specs?.filters ?? []))
        if (r.weeks?.length) setWeeks(new Set(r.weeks.map(String)))
      }
      toast.success("Got it", { description: meaning.length ? meaning.join(" · ") : "No filters in that question", })
      if (r.notes?.length) toast(r.notes.join(" "))
    } catch (e: any) {
      toast.error("Couldn't answer that", { description: e.message })
    } finally {
      setAsking(false)
    }
  }

  const visible = useMemo(() => (data ? applyFilters(data.plays, filters, { weeks }) : []), [data, filters, weeks])
  const onField = useMemo(() => (showUndrawn ? visible : visible.filter((p) => !p.estimated)), [visible, showUndrawn])
  const selected = useMemo(() => visible.find((p) => p.id === selectedId) ?? data?.plays.find((p) => p.id === selectedId) ?? null, [visible, data, selectedId])
  const specByCol = useMemo(() => new Map((specs?.filters ?? []).map((s) => [s.col, s])), [specs])

  // export the field as an image: always 2D, with whatever filters / colouring are on
  const fieldRef = useRef<FieldHandle>(null)
  const exportField = async (v: FieldView, format: ExportFormat) => {
    if (!data || !sel) return
    if (selectedId) {   // a selected play dims the rest: clear it first and let the field redraw
      setSelectedId(null)
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))
    }
    const svg = fieldRef.current?.svg()
    if (!svg) return
    const what = sel.kind === "pass" ? "Passing" : sel.kind === "route" ? "Route" : "Rushing"
    const parts = [`${onField.length} ${KIND_NOUN[sel.kind][1]}`]
    if (showUndrawn && onField.some((p) => p.estimated)) parts.push(`incl. ${onField.filter((p) => p.estimated).length} NGS didn't draw (estimated spots, dashed)`)
    if (weeks) parts.push(`weeks ${[...weeks].map((w) => weekLabel(w)).join(", ")}`)
    for (const [col, f] of Object.entries(filters)) parts.push(describe(specByCol.get(col), col, f))
    if (v === "plays" && colorBy !== "outcome")
      parts.push(`colored by ${({ epa: "EPA", route: "route type", down: "down", confidence: "match confidence" } as Record<string, string>)[colorBy] ?? colorBy}`)
    if (v === "heat" && sel.kind !== "pass") parts.push(heatOf === "end" ? (sel.kind === "route" ? "catch points" : "end points") : "whole paths")
    const filename = [data.player.name, data.season, sel.kind === "pass" ? "passing" : sel.kind === "route" ? "routes" : "rushing",
      v === "heat" ? "heatmap" : "chart"].join("_").replace(/[^\w.-]+/g, "_")
    try {
      await exportChart(svg, {
        view: v, heat: v === "heat" ? fieldRef.current!.heat() : null, format, filename,
        title: `${data.player.name} · ${data.season} ${what} ${v === "heat" ? "heatmap" : "chart"}`,
        subtitle: parts.join(" · "), legend: legendSpec(sel.kind, colorBy, v),
      })
      toast.success(format === "clipboard" ? "Copied to the clipboard" : `Saved ${filename}.${format}`)
    } catch (e) {
      toast.error("Couldn't export the chart", { description: String((e as Error)?.message ?? e) })
    }
  }
  const loading = sel && !data

  const activeCount = Object.keys(filters).length + (weeks ? 1 : 0)
  const filterPanel = specs && data ? (
    <FilterPanel specs={specs.filters} groups={specs.groups} plays={data.plays} games={data.games} state={filters}
      setState={setFilters} weeks={weeks} setWeeks={setWeeks} />
  ) : null

  return (
    <div className="min-h-svh overflow-x-clip bg-background text-foreground">
      <div className="pointer-events-none fixed inset-0 -z-0 bg-[radial-gradient(circle_at_20%_0%,rgba(52,211,153,0.06),transparent_40%),radial-gradient(circle_at_90%_10%,rgba(96,165,250,0.06),transparent_35%)]" />
      <Header seasons={seasons} season={season ?? 0} onSeason={setSeason} onSearch={() => setSearchOpen(true)} dark={dark} onTheme={() => setDark((d) => !d)} />
      <PlayerSearch open={searchOpen} onOpenChange={setSearchOpen} season={season ?? 0} players={players} onPick={(p, k) => pick(p, k, season!)} />

      {!sel ? (
        <Landing season={season ?? 0} players={players} onSearch={() => setSearchOpen(true)} />
      ) : (
        <main className="relative mx-auto flex max-w-[1680px] flex-col gap-4 px-4 py-5 lg:px-6">
          <PlayerHero player={sel.player} kind={sel.kind} season={sel.season} job={job}
            onKind={(k) => pick(sel.player, k, sel.season)} />
          <AskBar kind={sel.kind} busy={asking} onAsk={ask} />

          {loading ? (
            <LoadingCard job={job} name={sel.player.name} kind={sel.kind} />
          ) : data && (
            <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
              <aside className="hidden overflow-hidden rounded-2xl border border-border/60 bg-card/40 lg:sticky lg:top-20 lg:block lg:h-[calc(100svh-6.5rem)]">
                {filterPanel}
              </aside>
              {/* below lg the filters live in a drawer so the field comes first */}
              <Sheet open={filtersOpen} onOpenChange={setFiltersOpen}>
                <SheetContent side="left" className="w-[88vw] gap-0 p-0 sm:max-w-sm lg:hidden">
                  <SheetTitle className="sr-only">Filters</SheetTitle>
                  <SheetDescription className="sr-only">Narrow the plays on the field</SheetDescription>
                  {filterPanel}
                </SheetContent>
              </Sheet>

              <section className="flex min-w-0 flex-col gap-4">
                {data.pending && data.pending.length > 0 && (
                  <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm">
                    <Clock className="mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400" />
                    <div>
                      <span className="font-medium">
                        {data.pending.map((g) => weekLabel(g.week, true)).join(", ")}: play-by-play isn't out yet.
                      </span>{" "}
                      <span className="text-muted-foreground">
                        Its {data.pending.reduce((n, g) => n + g.plays, 0)} {KIND_NOUN[sel.kind][1]} are on the chart but can't be
                        matched until nflverse publishes the game, usually within a few hours. Reload then.
                      </span>
                    </div>
                  </div>
                )}
                <KpiGrid kind={sel.kind} plays={visible} />

                <Card className="gap-3 overflow-hidden border-border/60 bg-card/50 p-0">
                  <div className="flex flex-wrap items-center justify-between gap-3 px-4 pt-4">
                    <div className="flex flex-wrap items-center gap-2">
                      <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs lg:hidden" onClick={() => setFiltersOpen(true)}>
                        <SlidersHorizontal className="size-3.5" /> Filters
                        {activeCount > 0 && <Badge className="h-4 min-w-4 px-1 text-[10px]">{activeCount}</Badge>}
                      </Button>
                      <span className="text-sm font-medium">
                        {visible.length}
                        <span className="text-muted-foreground"> of {data.plays.length} {KIND_NOUN[sel.kind][1]}</span>
                      </span>
                      {Object.entries(filters).map(([col, f]) => (
                        <Badge key={col} variant="secondary" className="gap-1 pr-1">
                          {describe(specByCol.get(col), col, f)}
                          <button className="rounded-sm p-0.5 hover:bg-white/10" onClick={() => {
                            const n = { ...filters }
                            delete n[col]
                            setFilters(n)
                          }} aria-label="Remove filter">
                            <X className="size-3" />
                          </button>
                        </Badge>
                      ))}
                      {weeks && <Badge variant="secondary">Weeks {[...weeks].map((w) => weekLabel(w)).join(", ")}</Badge>}
                    </div>
                    <FieldToolbar kind={sel.kind} view={view} setView={setView} colorBy={colorBy} setColorBy={setColorBy}
                      heatOf={heatOf} setHeatOf={setHeatOf} threeD={threeD} setThreeD={setThreeD}
                      fullDepth={fullDepth} setFullDepth={setFullDepth} deeper={deeper} onExport={exportField}
                      undrawn={data.counts.undrawn ?? 0} showUndrawn={showUndrawn} setShowUndrawn={setShowUndrawn} />
                  </div>
                  <div className="bg-[#0b0c0f]">
                    <Field ref={fieldRef} kind={sel.kind} plays={onField} allPlays={data.plays} colorBy={colorBy} view={view} heatOf={heatOf}
                      threeD={threeD} selectedId={selectedId} onSelect={setSelectedId} animateKey={`${data.kind}-${data.version}-${sel.player.esb_id}`}
                      fullDepth={fullDepth} onDeeper={setDeeper} />
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-2 px-4 pb-4">
                    <Legend kind={sel.kind} colorBy={colorBy} view={view} undrawn={showUndrawn && sel.kind === "pass"} />
                    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                      <MousePointerClick className="size-3.5" /> Hover or click any play
                    </span>
                  </div>
                </Card>

                <Tabs defaultValue="insights" className="min-w-0">
                  <TabsList>
                    <TabsTrigger value="insights">Insights</TabsTrigger>
                    <TabsTrigger value="plays">Plays</TabsTrigger>
                    <TabsTrigger value="games">Games</TabsTrigger>
                  </TabsList>
                  <TabsContent value="insights" className="mt-3">
                    <Insights kind={sel.kind} plays={visible} fieldPlays={onField} />
                  </TabsContent>
                  <TabsContent value="plays" className="mt-3">
                    <PlayTable kind={sel.kind} plays={visible} selectedId={selectedId} onSelect={setSelectedId} />
                  </TabsContent>
                  <TabsContent value="games" className="mt-3">
                    <Games data={data} visibleIds={new Set(visible.map((p) => p.id))} onWeek={(w) => setWeeks(new Set([w]))} />
                  </TabsContent>
                </Tabs>
              </section>
            </div>
          )}
        </main>
      )}
      <PlayPanel jobId={job?.id} play={selected} kind={sel?.kind ?? "pass"} onClose={() => setSelectedId(null)} />
      <Toaster position="bottom-right" theme={dark ? "dark" : "light"} richColors closeButton />
    </div>
  )
}

function Games({ data, visibleIds, onWeek }: { data: Dataset; visibleIds: Set<string>; onWeek: (w: string) => void }) {
  const k = data.kind
  const line = (g: any) =>
    k === "pass"
      ? `${g.completions}/${g.attempts} · ${g.passingYards} yds · ${g.touchdowns} TD · ${g.interceptions} INT`
      : k === "route"
        ? `${g.receptions} rec · ${g.receivingYards} yds · ${g.touchdowns} TD`
        : `${g.carries} car · ${g.rushingYards} yds · ${g.touchdowns} TD`
  return (
    <Card className="border-border/60 bg-card/50 p-0">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="text-xs">Week</TableHead>
            <TableHead className="text-xs">Opponent</TableHead>
            <TableHead className="text-xs">Box score</TableHead>
            <TableHead className="text-right text-xs">On chart (filtered)</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.games.map((g) => {
            const n = data.plays.filter((p) => String(p.week) === String(g.week) && visibleIds.has(p.id)).length
            return (
              <TableRow key={g.game_id} className="cursor-pointer text-xs" onClick={() => onWeek(String(g.week))}>
                <TableCell className="font-mono">{weekLabel(g.week)}</TableCell>
                <TableCell className="font-mono">{g.opponent ?? "—"}</TableCell>
                <TableCell>{line(g)}</TableCell>
                <TableCell className="text-right font-mono">{n}</TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
    </Card>
  )
}
