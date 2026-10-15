import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { type Kind, type Play, weekLabel } from "@/lib/api"
import { lineNote } from "@/lib/stats"
import { confColor, ROUTE_COLORS } from "@/lib/colors"
import { pretty } from "@/lib/filters"
import { cn } from "@/lib/utils"

const FIELDS: [string, string][] = [
  ["defense_coverage_type", "Coverage"],
  ["defense_man_zone_type", "Man / zone"],
  ["was_pressure", "Pressured"],
  ["time_to_throw", "Time to throw"],
  ["offense_formation", "Formation"],
  ["offense_personnel", "Personnel"],
  ["is_play_action", "Play action"],
  ["is_motion", "Motion"],
  ["n_pass_rushers", "Rushers"],
  ["n_defense_box", "Box"],
  ["target_route", "NGS route"],
  ["run_location", "Run direction"],
  ["run_gap", "Gap"],
  ["receiver_player_name", "Target"],
  ["passer_player_name", "Passer"],
  ["wp", "Win prob"],
]

// 0/1 columns that are really yes/no flags; any other number is shown as a number (0 yards is "0", not "No")
const FLAGS = new Set(["was_pressure", "is_play_action", "is_motion", "success", "first_down"])

function show(v: any, col?: string): string {
  if (v === true || (col && FLAGS.has(col) && v === 1)) return "Yes"
  if (v === false || (col && FLAGS.has(col) && v === 0)) return "No"
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2)
  return pretty(v)
}

/** "Week 3 · vs BAL · Q2 11:04" */
export function playWhen(play: Play) {
  return [weekLabel(play.week, true), play.defteam && `vs ${play.defteam}`, play.qtr && `Q${play.qtr} ${play.time ?? ""}`]
    .filter(Boolean)
    .join(" · ")
}

/** Everything known about one play: shared by the click-to-pin sheet and the hover card on the field. */
export function PlayFacts({ play, kind, compact = false }: { play: Play; kind: Kind; compact?: boolean }) {
  if (!play.matched) return null
  const h4 = cn("font-medium uppercase tracking-wide text-muted-foreground", compact ? "mb-1.5 text-[10px]" : "mb-2 text-xs")
  const fields = FIELDS.filter(([c]) => play[c] != null && play[c] !== "")
  return (
    <div className={cn("flex flex-col", compact ? "gap-3" : "gap-5")}>
      <div className="grid grid-cols-3 gap-2">
        <Metric compact={compact} label="EPA" value={play.epa == null ? "—" : `${play.epa > 0 ? "+" : ""}${Number(play.epa).toFixed(2)}`}
          cls={play.epa == null ? undefined : play.epa >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"} />
        <Metric compact={compact} label={kind === "carry" ? "Gain" : "Air yds"} value={show(kind === "carry" ? play.yards_gained : play.air_yards)} />
        <Metric compact={compact} label={kind === "carry" ? "Success" : "YAC"} value={kind === "carry" ? show(play.success, "success") : show(play.yards_after_catch)} />
      </div>

      <section>
        <h4 className={h4}>Location confidence</h4>
        {play.estimated ? <p className="text-xs leading-snug text-muted-foreground">{lineNote(kind, play)}</p> : <Bar p={play.location_prob} />}
        {!compact && (
          <p className="mt-2 text-xs text-muted-foreground">
            {play.n_candidates === 1
              ? "The only play with this player and outcome in the game: certain."
              : `Chosen among ${play.n_candidates} plays with the same player and outcome in this game.`}
            {play.swap_yds ? ` A rival match would move it ${Number(play.swap_yds).toFixed(1)} yds.` : ""}
          </p>
        )}
      </section>

      {play.route_guess?.length > 0 && (
        <section>
          <h4 className={h4}>Route type (from the line's shape)</h4>
          <div className={cn("flex flex-col", compact ? "gap-1" : "gap-1.5")}>
            {play.route_guess.slice(0, compact ? 3 : undefined).map(([r, p]: [string, number]) => (
              <div key={r} className="flex items-center gap-2 text-xs">
                <span className="w-14 font-medium">{r}</span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                  <span className="block h-full rounded-full" style={{ width: `${100 * p}%`, background: ROUTE_COLORS[r] }} />
                </span>
                <span className="w-9 text-right font-mono text-muted-foreground">{Math.round(100 * p)}%</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {fields.length > 0 && (
        <>
          <Separator />
          <section className={cn("grid grid-cols-2", compact ? "gap-x-3 gap-y-1.5" : "gap-x-4 gap-y-2.5")}>
            {fields.map(([c, l]) => (
              <div key={c} className="min-w-0">
                <div className={cn("text-muted-foreground", compact ? "text-[10px]" : "text-[11px]")}>{l}</div>
                <div className={cn("break-words", compact ? "text-xs" : "text-sm")}>{show(play[c], c)}</div>
              </div>
            ))}
          </section>
        </>
      )}
      {play.gap && (
        <Badge variant="outline" className="w-fit whitespace-normal border-amber-400/30 text-amber-700 dark:text-amber-300">
          Joined to its TD ring by a straight jump (NGS drew no line into it)
        </Badge>
      )}
    </div>
  )
}

function Metric({ label, value, cls, compact }: { label: string; value: string; cls?: string; compact?: boolean }) {
  return (
    <div className={cn("rounded-xl border border-border/60 bg-card/50", compact ? "px-2.5 py-1.5" : "p-3")}>
      <div className={cn("text-muted-foreground", compact ? "text-[10px]" : "text-[11px]")}>{label}</div>
      <div className={cn("font-mono font-semibold", compact ? "text-base" : "text-xl", cls)}>{value}</div>
    </div>
  )
}

function Bar({ p }: { p: number | null }) {
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <span className="block h-full rounded-full transition-all" style={{ width: `${100 * (p ?? 0)}%`, background: confColor(p) }} />
      </span>
      <span className="w-9 text-right font-mono">{p == null ? "—" : `${Math.round(100 * p)}%`}</span>
    </div>
  )
}
