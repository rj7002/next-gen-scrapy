import { ShieldCheck } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import type { Kind, Play } from "@/lib/api"
import { kpis, matchQuality } from "@/lib/stats"
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

export function MatchQuality({ plays, kind }: { plays: Play[]; kind: Kind }) {
  const q = matchQuality(plays)
  const conf = q.matched ? q.confident / q.matched : 0
  const accuracy: Record<Kind, string> = {
    pass: "A random pass lands within 3 yds of its real spot ~97% of the time (2018–22) / ~90% (2023+, incl. 2026); confident ones 97–100%",
    route: "~98% (2018–22) / ~92–93% (2023+) of routes matched to the right spot, pinned by the QB's pass charts; confident ones ~99%",
    carry: "~78% (2018–21) / ~84% (2022+) of runs matched to the right spot, ~86–89% for 2023–25 with Big Data Bowl tracking; look-alike runs make the exact play harder; confident ones ~99%",
  }
  return (
    <Card className="gap-3 border-border/60 bg-card/50 p-4">
      <div className="flex items-center gap-2 text-sm font-medium">
        <ShieldCheck className="size-4 text-emerald-700 dark:text-emerald-300" /> Match quality
      </div>
      <div className="grid grid-cols-3 gap-3 text-center">
        <Stat label="Location" value={q.avgLocation == null ? "—" : `${Math.round(100 * q.avgLocation)}%`} />
        <Stat label="Confident" value={`${Math.round(100 * conf)}%`} />
        <Stat label="Unmatched" value={String(q.unmatched)} />
      </div>
      <Progress value={100 * conf} className="h-1.5" />
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        {accuracy[kind]}. Measured against independent checks.
        {q.undrawn > 0 && ` Not counted here: ${q.undrawn} throws NGS didn't draw.`}
      </p>
    </Card>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="font-mono text-lg font-semibold tabular-nums">{value}</div>
      <div className="text-[11px] text-muted-foreground">{label}</div>
    </div>
  )
}
