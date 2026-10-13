import { Check, Download, Loader2, Radio } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type Job, KIND_LABEL, type Kind, type Player } from "@/lib/api"
import { cn } from "@/lib/utils"
import { initials, statLine } from "./PlayerSearch"

export function PlayerHero({ player, kind, season, job, onKind }: {
  player: Player
  kind: Kind
  season: number
  job: Job | null
  onKind: (k: Kind) => void
}) {
  const kinds = (Object.keys(player.kinds) as Kind[]).sort((a, b) => ["pass", "route", "carry"].indexOf(a) - ["pass", "route", "carry"].indexOf(b))
  return (
    <div className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-card via-card to-emerald-950/20">
      <div className="pointer-events-none absolute -right-20 -top-24 size-80 rounded-full bg-emerald-400/10 blur-3xl" />
      <div className="relative flex flex-col gap-5 p-5 sm:flex-row sm:items-center">
        <Avatar className="size-20 shrink-0 rounded-2xl ring-1 ring-white/10 sm:size-24">
          <AvatarImage src={player.headshot ?? undefined} className="object-cover" />
          <AvatarFallback className="rounded-2xl text-xl">{initials(player.name)}</AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="secondary" className="font-mono">{player.team}</Badge>
            <span>{player.position}</span>
            <span>·</span>
            <span className="font-mono">{season}</span>
            {job && <JobBadge job={job} />}
          </div>
          <h2 className="mt-1 truncate text-3xl font-semibold tracking-tight sm:text-4xl">{player.name}</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {player.kinds[kind]?.games} games · {statLine(player, kind)}
          </p>
        </div>
        <div className="flex flex-col items-start gap-3 sm:items-end">
          {kinds.length > 1 && (
            <Tabs value={kind} onValueChange={(v) => onKind(v as Kind)}>
              <TabsList>
                {kinds.map((k) => (
                  <TabsTrigger key={k} value={k}>
                    {KIND_LABEL[k]}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          )}
          {job && job.version > 0 && (
            <Button asChild variant="outline" size="sm" className="gap-2">
              <a href={api.exportUrl(job.id)}>
                <Download className="size-4" /> Export CSV
              </a>
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

function JobBadge({ job }: { job: Job }) {
  if (job.status === "refining")
    return (
      <Badge variant="outline" className="gap-1.5 border-sky-400/30 text-sky-700 dark:text-sky-300">
        <Loader2 className="size-3 animate-spin" />
        {job.stage}
        {job.total ? ` ${job.done}/${job.total}` : "…"}
      </Badge>
    )
  if (job.status === "done")
    return (
      <Badge variant="outline" className="gap-1 border-emerald-400/30 text-emerald-700 dark:text-emerald-300">
        <Check className="size-3" /> Live · matched
      </Badge>
    )
  return null
}

// in the order engine._run does them; "Loading season data" (stored charts) counts as reading charts
const STAGES = ["Loading play-by-play", "Reading charts", "Matching to play-by-play"]

export function LoadingCard({ job, name, kind }: { job: Job | null; name: string; kind: Kind }) {
  const stage = job?.stage === "Loading season data" ? "Reading charts" : job?.stage
  const idx = job ? Math.max(0, STAGES.indexOf(stage ?? "")) : 0
  const pct = !job ? 4 : idx === 0 ? 8 : stage === "Reading charts" && job.total ? 15 + (70 * job.done) / job.total : idx === 1 ? 15 : 90
  return (
    <Card className="mx-auto mt-6 w-full max-w-xl gap-5 border-border/60 bg-card/60 p-6">
      <div className="flex items-center gap-3">
        <span className="relative flex size-9 items-center justify-center rounded-xl bg-emerald-400/10 text-emerald-700 dark:text-emerald-300">
          <Radio className="size-4" />
          <span className="absolute inset-0 animate-ping rounded-xl bg-emerald-400/10" />
        </span>
        <div>
          <div className="font-medium">Reading {name}’s {KIND_LABEL[kind].toLowerCase()} charts</div>
          <div className="text-xs text-muted-foreground">Live from Next Gen Stats, read and matched on the fly</div>
        </div>
      </div>
      <Progress value={pct} className="h-1.5" />
      <ol className="flex flex-col gap-2.5 text-sm">
        {STAGES.map((s, i) => {
          const state = i < idx ? "done" : i === idx ? "active" : "todo"
          return (
            <li key={s} className={cn("flex items-center gap-2.5", state === "todo" && "text-muted-foreground")}>
              <span className={cn("flex size-5 items-center justify-center rounded-full border",
                state === "done" && "border-emerald-400/40 bg-emerald-400/15 text-emerald-700 dark:text-emerald-300",
                state === "active" && "border-sky-400/40 text-sky-700 dark:text-sky-300", state === "todo" && "border-border")}>
                {state === "done" ? <Check className="size-3" /> : state === "active" ? <Loader2 className="size-3 animate-spin" /> : null}
              </span>
              {s}
              {s === "Reading charts" && job?.total ? (
                <span className="ml-auto font-mono text-xs text-muted-foreground">
                  {job.done}/{job.total} charts
                </span>
              ) : null}
            </li>
          )
        })}
      </ol>
    </Card>
  )
}
