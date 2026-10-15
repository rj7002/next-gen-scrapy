import { Card } from "@/components/ui/card"
import type { Kind, Play } from "@/lib/api"
import { inLog, kpis } from "@/lib/stats"
import { cn } from "@/lib/utils"

export function KpiGrid({ kind, plays }: { kind: Kind; plays: Play[] }) {
  return (
    <div className="flex flex-col gap-1.5">
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4 xl:grid-cols-7">
      {kpis(kind, plays).map((k) => (
        <Card key={k.label} className="gap-0.5 border-border/60 bg-card/50 px-4 py-3">
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{k.label}</div>
          <div className={cn("font-mono text-2xl font-semibold tabular-nums tracking-tight",
            k.tone === "good" && "text-emerald-600 dark:text-emerald-400", k.tone === "bad" && "text-rose-600 dark:text-rose-400")}>
            {k.value}
          </div>
          {k.sub && <div className="truncate text-[11px] text-muted-foreground">{k.sub}</div>}
        </Card>
      ))}
    </div>
    {notes(kind, plays).length > 0 && (
      <p className="px-1 text-[11px] text-muted-foreground">{notes(kind, plays).join(" ")}</p>
    )}
    </div>
  )
}

/** what's in the totals besides drawn, matched plays - and what's drawn but not in them */
function notes(kind: Kind, plays: Play[]): string[] {
  const out: string[] = []
  const est = plays.filter((p) => p.estimated && !p.lateral).length
  const noun = kind === "pass" ? ["throw", "throws"] : kind === "route" ? ["target", "targets"] : ["carry", "carries"]
  const n = (k: number) => `${k} ${noun[k === 1 ? 0 : 1]}`
  if (kind === "pass" && est)
    out.push(`Includes ${n(est)} Next Gen Stats didn't draw on its chart (mostly incompletions with no intended receiver, plus spikes). Their spots are estimated from play-by-play; turn on "Undrawn" to see them.`)
  else if (est)
    out.push(`Includes ${n(est)} with no chart line (not drawn by Next Gen Stats, or not told apart from the others), from play-by-play, so the totals are his game log's.`)
  const lines = plays.filter((p) => !p.matched).length
  if (lines) out.push(`${lines} chart ${kind === "pass" ? "dot" : "line"}${lines === 1 ? "" : "s"} no play fits ${lines === 1 ? "isn't" : "aren't"} counted.`)
  const two = plays.filter((p) => p.matched && !p.estimated && !inLog(p)).length
  if (two) out.push(`${two} two-point tr${two === 1 ? "y" : "ies"} on the chart ${two === 1 ? "isn't" : "aren't"} counted, as in the game log.`)
  return out
}
