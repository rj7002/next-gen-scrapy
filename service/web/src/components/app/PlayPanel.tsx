import { useEffect, useState } from "react"
import { Flag, Timer, Trophy, Users } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { api, type Kind, type Person, type Play, type PlayDetailData, unmatchedText, weekLabel } from "@/lib/api"
import { confColor, GAIN, OUTCOME, ROUTE_COLORS, ROUTE_LINE } from "@/lib/colors"
import { pretty } from "@/lib/filters"
import { cn } from "@/lib/utils"

const ORD = ["", "1st", "2nd", "3rd", "4th"]
const yes = (v: any) => v === true || v === 1 || v === "1" || v === "true"
const no = (v: any) => v === false || v === 0 || v === "0" || v === "false"
const signed = (v: number, d = 1) => `${v > 0 ? "+" : ""}${v.toFixed(d)}`
const pct = (v: number) => `${Math.round(100 * v)}%`
const initials = (n?: string | null) => (n ?? "?").split(" ").map((s) => s[0]).slice(0, 2).join("")

/** The panel that opens when a play is clicked: everything known about it, fetched in full on open. */
export function PlayPanel({ jobId, play, kind, onClose }: { jobId?: string; play: Play | null; kind: Kind; onClose: () => void }) {
  const [d, setD] = useState<PlayDetailData | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    setD(null)
    setErr(null)
    if (!play || !jobId) return
    let live = true
    api.play(jobId, play.id).then((x) => live && setD(x)).catch((e) => live && setErr(String(e.message ?? e)))
    return () => {
      live = false
    }
  }, [jobId, play?.id])

  return (
    <Sheet open={!!play} onOpenChange={(o) => !o && onClose()} modal={false}>
      <SheetContent className="w-full gap-0 overflow-y-auto border-border/60 bg-background/95 p-0 backdrop-blur-xl sm:max-w-lg">
        {play && (
          <>
            <Header play={play} d={d} />
            <div className="flex flex-col gap-6 px-5 pt-4 pb-10">
              <MiniPlay kind={kind} play={play} d={d} />
              {!play.matched ? (
                <p className="text-sm text-muted-foreground">
                  {play.reason === "no_fit" ? unmatchedText(play)
                    : `Drawn on the chart, but no play-by-play row could be matched to it${err ? "" : " with confidence"}.`}
                </p>
              ) : !d ? (
                err ? <p className="text-sm text-rose-500">Couldn't load this play: {err}</p> : <Loading />
              ) : (
                <>
                  <Outcome kind={kind} d={d} />
                  <Numbers kind={kind} d={d} />
                  <WinProb d={d} />
                  <People people={d.people ?? []} />
                  <Scheme kind={kind} d={d} />
                  {kind !== "carry" && <Charting d={d} />}
                  {kind !== "carry" && <GameCenter d={d} play={play} />}
                  {kind === "route" && d.chart.route_guess?.length > 0 && <RouteGuess guess={d.chart.route_guess} ngs={d.scheme?.target_route} />}
                  {play.estimated ? (
                    <Section title="Location">
                      <p className="text-xs leading-relaxed text-muted-foreground">Not drawn by Next Gen Stats. Its spot is estimated from play-by-play air yards and direction: depth is usually within a yard, but across the field it's only known to the side.</p>
                    </Section>
                  ) : (
                    <Match d={d} />
                  )}
                  <Drive d={d} />
                </>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}

function Section({ title, icon: Icon, children, right }: { title: string; icon?: React.ElementType; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section>
      <div className="mb-2.5 flex items-center justify-between gap-2">
        <h4 className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          {Icon && <Icon className="size-3.5" />} {title}
        </h4>
        {right}
      </div>
      {children}
    </section>
  )
}

function Loading() {
  return (
    <div className="flex flex-col gap-3">
      <Skeleton className="h-8 w-2/3" />
      <div className="grid grid-cols-3 gap-2">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-16" />)}</div>
      <Skeleton className="h-24" />
      <Skeleton className="h-40" />
    </div>
  )
}

// ---- header: matchup, scoreboard, situation, description ---------------------------------------------------------
function Header({ play, d }: { play: Play; d: PlayDetailData | null }) {
  const s = d?.situation ?? {}
  const g = d?.game ?? {}
  const pos = s.posteam ?? play.posteam
  const def = s.defteam ?? play.defteam
  const date = g.game_date ? new Date(`${g.game_date}T12:00:00`).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: "numeric" }) : null
  const down = s.down ?? play.down
  const togo = s.ydstogo ?? play.ydstogo
  return (
    <div className="border-b border-border/60 bg-gradient-to-b from-emerald-500/[0.07] to-transparent px-5 pt-5 pb-4">
      <div className="pr-8 text-xs text-muted-foreground">
        {[weekLabel(play.week, true), date, g.away_team && g.home_team ? `${g.away_team} @ ${g.home_team}` : null].filter(Boolean).join(" · ")}
      </div>
      {pos && def && (
        <div className="mt-3 flex items-center gap-3">
          <Score team={pos} score={s.posteam_score} ball />
          <span className="text-muted-foreground">–</span>
          <Score team={def} score={s.defteam_score} />
          {(s.qtr ?? play.qtr) != null && (
            <span className="ml-auto rounded-md border border-border/70 px-2 py-1 font-mono text-xs">
              {s.qtr === 5 ? "OT" : `Q${s.qtr ?? play.qtr}`} {s.time ?? play.time}
            </span>
          )}
        </div>
      )}
      <SheetTitle className="mt-3 text-2xl font-semibold tracking-tight">
        {down ? `${ORD[down]} & ${s.goal_to_go ? "Goal" : togo}` : play.matched ? "Play" : "Unmatched chart play"}
        {s.yrdln && <span className="font-normal text-muted-foreground"> at {s.yrdln}</span>}
      </SheetTitle>
      <SheetDescription className="mt-1.5 text-sm leading-relaxed text-foreground/80">
        {d?.result?.desc ?? play.desc ?? ""}
      </SheetDescription>
    </div>
  )
}

function Score({ team, score, ball }: { team: string; score?: number; ball?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className="rounded-md bg-foreground/10 px-1.5 py-0.5 font-mono text-xs font-semibold">{team}</span>
      <span className="font-mono text-xl font-semibold tabular-nums">{score ?? "–"}</span>
      {ball && <span className="size-1.5 rounded-full bg-amber-500" title="Possession" />}
    </span>
  )
}

// ---- the play drawn on a small field, with the first-down line and the goal line ---------------------------------
function MiniPlay({ kind, play, d }: { kind: Kind; play: Play; d: PlayDetailData | null }) {
  const s = d?.situation ?? {}
  const togo: number | null = s.ydstogo ?? play.ydstogo ?? null
  const goal: number | null = s.yardline_100 ?? play.yardline_100 ?? null
  const xs: number[] = kind === "pass" ? [play.x_coord] : play.path_x ?? []
  const ys: number[] = kind === "pass" ? [play.y_coord] : play.path_y ?? []
  if (!ys.length || ys[0] == null) return null
  const lo = Math.min(-4, Math.min(...ys) - 2)
  let hi = Math.max(12, Math.max(...ys) + 4, togo != null && !s.goal_to_go ? togo + 3 : 0)
  if (goal != null && goal < hi + 10) hi = Math.min(Math.max(hi, goal + 4), goal + 10)
  const H = hi - lo
  const Y = (y: number) => hi - y
  const W = 53.33
  const lines = []
  for (let y = Math.ceil(lo / 5) * 5; y <= hi; y += 5)
    lines.push(<line key={y} x1={-W / 2} x2={W / 2} y1={Y(y)} y2={Y(y)} stroke="white" strokeOpacity={y % 10 ? 0.07 : 0.14} strokeWidth={0.12} />)
  const seg: string[] = play.path_segment ?? []
  const ac = kind === "route" ? seg.indexOf("after_catch") : -1
  const pts = (a: number, b: number) => xs.slice(a, b).map((x, i) => `${x.toFixed(2)},${Y(ys[a + i]).toFixed(2)}`).join(" ")
  const lineColor = kind === "carry" ? (GAIN[play.gain_class] ?? "#a1a1aa") : play.route_type === "INCOMPLETE" ? ROUTE_LINE.incomplete : ROUTE_LINE.complete
  const td = play.chart_touchdown === true || play.chart_touchdown === 1 || play.pass_type === "TOUCHDOWN"
  return (
    <div className="overflow-hidden rounded-xl border border-border/60 bg-[#101215]">
      <svg viewBox={`${-W / 2 - 1} 0 ${W + 2} ${H}`} className="block max-h-64 w-full">
        <rect x={-W / 2} y={0} width={W} height={H} fill="#171a1f" />
        {goal != null && goal <= hi && (
          <>
            <rect x={-W / 2} y={0} width={W} height={Math.max(0, Y(goal))} fill="#10b981" fillOpacity={0.12} />
            <line x1={-W / 2} x2={W / 2} y1={Y(goal)} y2={Y(goal)} stroke="#34d399" strokeOpacity={0.7} strokeWidth={0.25} />
            {Y(goal) > 2.5 && <text x={0} y={Math.min(Y(goal) - 1, 3)} textAnchor="middle" fontSize={1.6} fill="#34d399" fillOpacity={0.6} className="font-mono">END ZONE</text>}
          </>
        )}
        {lines}
        {[-3.08, 3.08].map((h) => (
          <line key={h} x1={h} x2={h} y1={0} y2={H} stroke="white" strokeOpacity={0.05} strokeWidth={0.15} strokeDasharray="0.3 0.7" />
        ))}
        <line x1={-W / 2} x2={W / 2} y1={Y(0)} y2={Y(0)} stroke="#60a5fa" strokeWidth={0.3} />
        {togo != null && !s.goal_to_go && togo > 0 && togo + 0.01 < (goal ?? 999) && (
          <line x1={-W / 2} x2={W / 2} y1={Y(togo)} y2={Y(togo)} stroke="#facc15" strokeWidth={0.3} strokeOpacity={0.85} />
        )}
        {kind === "pass" && play.estimated ? (
          <circle cx={xs[0]} cy={Y(ys[0])} r={0.6} fill="none" stroke={OUTCOME.INCOMPLETE} strokeWidth={0.14} strokeDasharray="0.3 0.2" />
        ) : kind === "pass" ? (
          <>
            <circle cx={xs[0]} cy={Y(ys[0])} r={0.6} fill={OUTCOME[play.pass_type] ?? "#a1a1aa"} fillOpacity={0.2} stroke={OUTCOME[play.pass_type] ?? "#a1a1aa"} strokeWidth={0.16} />
            <circle cx={xs[0]} cy={Y(ys[0])} r={0.2} fill={OUTCOME[play.pass_type] ?? "#a1a1aa"} />
          </>
        ) : (
          <>
            <polyline points={pts(0, ac > 0 ? ac : xs.length)} fill="none" stroke={lineColor} strokeWidth={0.25} strokeLinecap="round" strokeLinejoin="round" />
            {ac > 0 && <polyline points={pts(ac - 1, xs.length)} fill="none" stroke={ROUTE_LINE.after} strokeWidth={0.25} strokeLinecap="round" strokeLinejoin="round" />}
            {ac > 0 && <circle cx={xs[ac - 1]} cy={Y(ys[ac - 1])} r={0.32} fill={ROUTE_LINE.after} />}
            {td && <circle cx={xs[xs.length - 1]} cy={Y(ys[ys.length - 1])} r={0.7} fill="none" stroke={ROUTE_LINE.td} strokeWidth={0.18} />}
          </>
        )}
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-border/60 px-3 py-2 text-[10px] text-zinc-400">
        <span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-3 bg-[#60a5fa]" /> Line of scrimmage</span>
        {togo != null && !s.goal_to_go && <span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-3 bg-[#facc15]" /> First down</span>}
        {goal != null && goal <= hi && <span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-3 bg-[#34d399]" /> Goal line</span>}
        {kind === "pass" && play.x_coord != null && (
          <span className="ml-auto font-mono">{play.estimated ? "≈ " : ""}{play.y_coord?.toFixed(1)} yds downfield · {Math.abs(play.x_coord).toFixed(1)} yds {play.x_coord < 0 ? "left" : "right"} of center{play.estimated ? " (estimated)" : ""}</span>
        )}
      </div>
    </div>
  )
}

// ---- outcome tags ------------------------------------------------------------------------------------------------
function Outcome({ kind, d }: { kind: Kind; d: PlayDetailData }) {
  const r = d.result ?? {}
  const tags: [string, string][] = []
  const what = kind === "carry" ? "Run" : d.chart.pass_type ?? d.chart.route_type
  if (what) tags.push([pretty(String(what)), ""])
  if (r.yards_gained != null) tags.push([`${r.yards_gained > 0 ? "+" : ""}${r.yards_gained} yds`, ""])
  if (yes(r.touchdown)) tags.push(["Touchdown", "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300"])
  else if (yes(r.first_down)) tags.push(["First down", "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"])
  if (yes(r.interception)) tags.push(["Interception", "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300"])
  if (yes(r.fumble_lost)) tags.push(["Fumble lost", "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300"])
  if (yes(r.sack)) tags.push(["Sack", "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300"])
  if (yes(r.qb_hit)) tags.push(["QB hit", "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300"])
  if (yes(r.qb_scramble)) tags.push(["Scramble", ""])
  if (yes(r.penalty)) tags.push([`Penalty${r.penalty_type ? `: ${r.penalty_type}` : ""}${r.penalty_yards ? ` (${r.penalty_yards} yds)` : ""}`, "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300"])
  if (yes(r.success)) tags.push(["Successful play", ""])
  if (r.pass_length || r.pass_location) tags.push([[r.pass_length, r.pass_location].filter(Boolean).join(" ").replace(/^./, (c: string) => c.toUpperCase()), ""])
  if (r.run_location) tags.push([[r.run_location, r.run_gap].filter(Boolean).join(" ").replace(/^./, (c: string) => c.toUpperCase()), ""])
  return (
    <div className="flex flex-wrap gap-1.5">
      {tags.map(([t, cls]) => (
        <Badge key={t} variant="outline" className={cn("rounded-full px-2.5 py-0.5 text-xs font-normal", cls)}>{t}</Badge>
      ))}
    </div>
  )
}

// ---- the numbers -------------------------------------------------------------------------------------------------
function Numbers({ kind, d }: { kind: Kind; d: PlayDetailData }) {
  const r = d.result ?? {}
  const cells: { label: string; value: string; sub?: string; tone?: "good" | "bad" }[] = []
  if (r.epa != null) cells.push({ label: "EPA", value: signed(r.epa, 2), tone: r.epa >= 0 ? "good" : "bad",
    sub: r.air_epa != null && kind !== "carry" ? `${signed(r.air_epa, 2)} air · ${signed(r.yac_epa ?? 0, 2)} YAC` : undefined })
  if (r.wpa != null) cells.push({ label: "Win prob added", value: signed(100 * r.wpa, 1) + "%", tone: r.wpa >= 0 ? "good" : "bad" })
  if (kind === "carry") {
    if (r.yards_gained != null) cells.push({ label: "Gain", value: `${r.yards_gained}`, sub: "yards" })
  } else {
    if (r.air_yards != null) cells.push({ label: "Air yards", value: `${r.air_yards}` })
    if (r.yards_after_catch != null) cells.push({ label: "YAC", value: `${r.yards_after_catch}`,
      sub: r.xyac_mean_yardage != null ? `expected ${Number(r.xyac_mean_yardage).toFixed(1)}` : undefined,
      tone: r.xyac_mean_yardage != null ? (r.yards_after_catch >= r.xyac_mean_yardage ? "good" : "bad") : undefined })
    if (r.cp != null) cells.push({ label: "Completion prob", value: pct(r.cp), sub: r.cpoe != null ? `CPOE ${signed(r.cpoe, 1)}%` : undefined,
      tone: r.cpoe != null ? (r.cpoe >= 0 ? "good" : "bad") : undefined })
  }
  if (r.xpass != null) cells.push({ label: "Pass expected", value: pct(r.xpass), sub: r.pass_oe != null ? `${signed(r.pass_oe, 0)}% over expected` : undefined })
  if (r.xyac_fd != null && kind !== "carry") cells.push({ label: "Chance of 1st after catch", value: pct(r.xyac_fd) })
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {cells.slice(0, 6).map((c) => (
        <div key={c.label} className="rounded-xl border border-border/60 bg-card/50 px-3 py-2.5">
          <div className="text-[11px] text-muted-foreground">{c.label}</div>
          <div className={cn("font-mono text-xl font-semibold tabular-nums", c.tone === "good" && "text-emerald-600 dark:text-emerald-400", c.tone === "bad" && "text-rose-600 dark:text-rose-400")}>{c.value}</div>
          {c.sub && <div className="truncate text-[10px] text-muted-foreground">{c.sub}</div>}
        </div>
      ))}
    </div>
  )
}

function WinProb({ d }: { d: PlayDetailData }) {
  const r = d.result ?? {}
  const s = d.situation ?? {}
  if (r.wp == null) return null
  const before = r.wp
  const after = Math.min(1, Math.max(0, r.wp + (r.wpa ?? 0)))
  return (
    <Section title={`${s.posteam ?? "Offense"} win probability`} icon={Trophy}
      right={<span className="font-mono text-xs text-muted-foreground">{pct(before)} → <span className="text-foreground">{pct(after)}</span></span>}>
      {/* grey up to the lower of before / after, then the change in green (gained) or red (lost) */}
      <div className="relative h-2.5 overflow-hidden rounded-full bg-muted">
        <div className="absolute inset-y-0 left-0 bg-foreground/25" style={{ width: pct(Math.min(before, after)) }} />
        <div className={cn("absolute inset-y-0", after >= before ? "bg-emerald-500" : "bg-rose-500")}
          style={{ left: pct(Math.min(before, after)), width: pct(Math.abs(after - before)) }} />
      </div>
    </Section>
  )
}

// ---- who was involved --------------------------------------------------------------------------------------------
function People({ people }: { people: Person[] }) {
  if (!people.length) return null
  return (
    <Section title="On the play" icon={Users}>
      <div className="flex flex-col gap-1.5">
        {people.map((p) => (
          <div key={`${p.id}-${p.role}`} className="flex items-center gap-3 rounded-lg border border-border/50 bg-card/40 px-2.5 py-2">
            <Avatar className="size-9 ring-1 ring-border">
              <AvatarImage src={p.headshot ?? undefined} className="object-cover" />
              <AvatarFallback className="text-[10px]">{initials(p.name)}</AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium">{p.name ?? "—"}</div>
              <div className="truncate text-[11px] text-muted-foreground">
                {[p.position, p.jersey != null ? `#${p.jersey}` : null, p.team].filter(Boolean).join(" · ")}
              </div>
            </div>
            <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">{p.role}</span>
          </div>
        ))}
      </div>
    </Section>
  )
}

// ---- scheme ------------------------------------------------------------------------------------------------------
const SCHEME: [string, string][] = [
  ["offense_formation", "Formation"], ["offense_personnel", "Offense personnel"], ["defense_personnel", "Defense personnel"],
  ["n_defense_box", "In the box"], ["n_pass_rushers", "Pass rushers"], ["n_blitzers", "Blitzers"],
  ["defense_coverage_type", "Coverage"], ["defense_man_zone_type", "Man / zone"], ["time_to_throw", "Time to throw"],
  ["target_route", "Route (NGS)"], ["qb_location", "QB alignment"], ["starting_hash", "Ball on hash"], ["n_offense_backfield", "In the backfield"],
]
const SCHEME_FLAGS: [string, string][] = [
  ["is_play_action", "Play action"], ["is_rpo", "RPO"], ["is_screen_pass", "Screen"], ["is_motion", "Motion"],
  ["is_no_huddle", "No huddle"], ["was_pressure", "Pressured"], ["is_qb_out_of_pocket", "Out of pocket"], ["is_trick_play", "Trick play"],
]

const QB_LOC: Record<string, string> = { S: "Shotgun", U: "Under center", P: "Pistol" }
const READ: Record<string, string> = { "1": "1st read", "2": "2nd read", "3": "3rd read", "3+": "3rd+ read", SD: "Scramble drill",
  DES: "Designed throw", CHK: "Checkdown" }

function fmtScheme(col: string, v: any) {
  if (col === "time_to_throw") return `${Number(v).toFixed(2)} s`
  if (col === "starting_hash") return ({ L: "Left", R: "Right", M: "Middle" } as Record<string, string>)[v] ?? pretty(v)
  if (col === "qb_location") return QB_LOC[v] ?? pretty(v)
  return pretty(v)
}

function Scheme({ kind, d }: { kind: Kind; d: PlayDetailData }) {
  const sc = { ...(d.scheme ?? {}) }
  const s = d.situation ?? {}
  if (s.shotgun != null && !sc.qb_location) sc.qb_location = yes(s.shotgun) ? "Shotgun" : "Under center"
  const facts = SCHEME.filter(([c]) => sc[c] != null && sc[c] !== "" && !(kind === "carry" && (c === "time_to_throw" || c === "target_route")))
  const flags = SCHEME_FLAGS.filter(([c]) => yes(sc[c]))
  if (!facts.length && !flags.length) return null
  return (
    <Section title="Scheme" icon={Flag}>
      {facts.length > 0 && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-3">
          {facts.map(([c, l]) => (
            <div key={c} className="min-w-0">
              <div className="text-[11px] text-muted-foreground">{l}</div>
              <div className="break-words text-sm">{fmtScheme(c, sc[c])}</div>
            </div>
          ))}
        </div>
      )}
      {flags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {flags.map(([c, l]) => (
            <span key={c} className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 text-xs text-emerald-700 dark:text-emerald-300">
              {l}
            </span>
          ))}
        </div>
      )}
    </Section>
  )
}

// ---- FTN charting ------------------------------------------------------------------------------------------------
const CHARTING: [string, string][] = [
  ["is_catchable_ball", "Catchable"], ["is_contested_ball", "Contested"], ["is_drop", "Dropped"],
  ["is_created_reception", "Created reception"], ["is_interception_worthy", "Interception-worthy"], ["is_throw_away", "Thrown away"],
]

function Charting({ d }: { d: PlayDetailData }) {
  const c = d.charting ?? {}
  const read = c.read_thrown
  const items = CHARTING.filter(([k]) => yes(c[k]))
  const uncatchable = no(c.is_catchable_ball)
  const readOk = read != null && read !== "" && String(read) !== "0"
  if (!items.length && !uncatchable && !readOk) return null
  return (
    <Section title="Charted by FTN">
      <div className="flex flex-wrap gap-1.5">
        {readOk && (
          <span className="rounded-full border border-border/60 px-2.5 py-0.5 text-xs">
            <span className="font-medium">{READ[String(read)] ?? pretty(read)}</span>
          </span>
        )}
        {uncatchable && (
          <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-0.5 text-xs text-amber-700 dark:text-amber-300">Not catchable</span>
        )}
        {items.map(([k, l]) => (
          <span key={k} className={cn("rounded-full border px-2.5 py-0.5 text-xs",
            k === "is_drop" || k === "is_interception_worthy" || k === "is_throw_away"
              ? "border-rose-500/40 bg-rose-500/10 text-rose-700 dark:text-rose-300"
              : "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300")}>
            {l}
          </span>
        ))}
      </div>
    </Section>
  )
}

// ---- NGS Game Center --------------------------------------------------------------------------------------------
function GameCenter({ d, play }: { d: PlayDetailData; play: Play }) {
  const g = d.gamecenter ?? {}
  if (g.ngs_receiver_targets == null) return null
  const who = d.people?.find((p) => p.role === "Target")?.name ?? play.receiver_player_name ?? "The target"
  return (
    <Section title="Next Gen Stats, this game">
      <p className="text-sm">
        <span className="font-medium">{who}</span>: {g.ngs_receiver_targets} targets, averaging{" "}
        <span className="font-mono">{Number(g.ngs_receiver_avg_air_yards).toFixed(1)}</span> air yards (NGS-measured).
      </p>
    </Section>
  )
}

function RouteGuess({ guess, ngs }: { guess: [string, number][]; ngs?: string }) {
  return (
    <Section title="Route type, from the line's shape" right={ngs ? <span className="text-xs text-muted-foreground">NGS says <span className="font-medium text-foreground">{pretty(ngs)}</span></span> : undefined}>
      <div className="flex flex-col gap-1.5">
        {guess.map(([r, p]) => (
          <div key={r} className="flex items-center gap-2 text-xs">
            <span className="w-16 font-medium">{r}</span>
            <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
              <span className="block h-full rounded-full" style={{ width: `${100 * p}%`, background: ROUTE_COLORS[r] }} />
            </span>
            <span className="w-9 text-right font-mono text-muted-foreground">{Math.round(100 * p)}%</span>
          </div>
        ))}
      </div>
    </Section>
  )
}

// ---- match -------------------------------------------------------------------------------------------------------
function Match({ d }: { d: PlayDetailData }) {
  const m = d.match ?? {}
  return (
    <Section title="Location confidence">
      <Bar p={m.location_prob} />
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
        Chance this is within 3 yds of where the play really went.{" "}
        {m.n_candidates === 1
          ? "The only play with this player and outcome in the game, so it's certain."
          : `Chosen among ${m.n_candidates} plays with the same player and outcome in this game.`}
        {m.swap_yds ? ` The closest rival match would move it ${Number(m.swap_yds).toFixed(1)} yds.` : ""}
      </p>
    </Section>
  )
}

function Bar({ p }: { p?: number | null }) {
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
        <span className="block h-full rounded-full" style={{ width: `${100 * (p ?? 0)}%`, background: confColor(p ?? null) }} />
      </span>
      <span className="w-10 text-right font-mono text-sm">{p == null ? "—" : pct(p)}</span>
    </div>
  )
}

// ---- the drive ---------------------------------------------------------------------------------------------------
function Drive({ d }: { d: PlayDetailData }) {
  const dr = d.drive
  if (!dr || !dr.plays?.length) return null
  const meta = [dr.plays_count != null && `${dr.plays_count} plays`, dr.time, dr.first_downs != null && `${dr.first_downs} first downs`].filter(Boolean).join(" · ")
  return (
    <Section title={`Drive${dr.number != null ? ` ${dr.number}` : ""}`} icon={Timer}
      right={dr.result ? <Badge variant="outline" className="rounded-full text-[11px] font-normal">{dr.result}</Badge> : undefined}>
      {meta && <div className="-mt-1 mb-2 text-xs text-muted-foreground">{meta}{dr.start ? ` · from ${dr.start}` : ""}</div>}
      <ol className="relative flex flex-col border-l border-border/70 pl-3">
        {dr.plays.map((p) => (
          <li key={p.play_id} className={cn("relative py-1.5 text-xs", p.current && "font-medium")}>
            <span className={cn("absolute top-3 -left-[17px] size-2 rounded-full border border-background", p.current ? "bg-emerald-500" : "bg-muted-foreground/40")} />
            <div className="flex items-baseline gap-2">
              <span className="w-[86px] shrink-0 font-mono text-[11px] text-muted-foreground">
                {p.down ? `${ORD[p.down]}&${p.ydstogo}` : pretty(p.play_type ?? "")} {p.yrdln ? `· ${p.yrdln}` : ""}
              </span>
              <span className={cn("line-clamp-2", p.current ? "text-foreground" : "text-muted-foreground")}>{p.desc}</span>
              {p.epa != null && (
                <span className={cn("ml-auto shrink-0 font-mono text-[11px]", p.epa >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400")}>
                  {signed(p.epa, 1)}
                </span>
              )}
            </div>
          </li>
        ))}
      </ol>
    </Section>
  )
}
