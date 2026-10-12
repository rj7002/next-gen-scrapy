import { Card } from "@/components/ui/card"
import type { Kind, Play } from "@/lib/api"
import { kpis } from "@/lib/stats"
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
    {kind === "pass" && plays.some((p) => p.estimated) && (
      <p className="px-1 text-[11px] text-muted-foreground">
        Includes {plays.filter((p) => p.estimated).length} throws Next Gen Stats didn't draw on its chart (mostly incompletions with
        no intended receiver, plus spikes). Their spots are estimated from play-by-play; turn on "Undrawn" to see them.
      </p>
    )}
    </div>
  )
}
