import { useMemo, useState } from "react"
import { ArrowDown, ArrowUp } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { type Kind, type Play, weekLabel } from "@/lib/api"
import { confColor, playRoute } from "@/lib/colors"
import { situation } from "@/lib/stats"
import { cn } from "@/lib/utils"

type Col = { key: string; label: string; get: (p: Play) => any; render?: (p: Play) => React.ReactNode; cls?: string }

const weekNum = (p: Play) => (String(p.week).startsWith("post") ? 100 + Number(String(p.week).split("-")[1]) : Number(p.week))

export function PlayTable({ kind, plays, selectedId, onSelect }: { kind: Kind; plays: Play[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: "week", dir: 1 })
  const cols: Col[] = useMemo(() => {
    const outcome: Col =
      kind === "pass"
        ? { key: "outcome", label: "Result", get: (p) => p.pass_type, render: (p) => <Outcome v={p.pass_type} /> }
        : kind === "route"
          ? { key: "outcome", label: "Result", get: (p) => p.route_type, render: (p) => <Outcome v={p.chart_touchdown ? "TOUCHDOWN" : p.route_type} /> }
          : { key: "outcome", label: "Gain", get: (p) => p.yards_gained, render: (p) => <span className="font-mono">{p.yards_gained ?? "—"}</span> }
    const list: Col[] = [
      { key: "week", label: "Wk", get: weekNum, render: (p) => <span className="font-mono text-muted-foreground">{weekLabel(p.week)}</span> },
      { key: "opp", label: "Opp", get: (p) => p.defteam ?? "", render: (p) => <span className="font-mono">{p.defteam ?? "—"}</span> },
      { key: "q", label: "Q", get: (p) => (p.qtr ?? 0) * 10000 - (Number(String(p.time ?? "0:0").split(":")[0]) * 60 + Number(String(p.time ?? "0:0").split(":")[1] ?? 0)), render: (p) => <span className="font-mono text-muted-foreground">{p.qtr ? `Q${p.qtr} ${p.time ?? ""}` : "—"}</span> },
      { key: "sit", label: "Down", get: (p) => (p.down ?? 9) * 100 + (p.ydstogo ?? 0), render: (p) => situation(p) || "—" },
      outcome,
    ]
    if (kind !== "carry") list.push({ key: "route", label: "Route", get: (p) => playRoute(p) ?? "", render: (p) => playRoute(p) ?? "—" })
    if (kind !== "carry") list.push({ key: "air", label: "Air", get: (p) => p.air_yards ?? -99, render: (p) => <span className="font-mono">{p.air_yards ?? "—"}</span> })
    list.push(
      { key: "epa", label: "EPA", get: (p) => p.epa ?? -99, render: (p) => (p.epa == null ? "—" : <span className={cn("font-mono", p.epa >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400")}>{p.epa > 0 ? "+" : ""}{Number(p.epa).toFixed(2)}</span>) },
      { key: "conf", label: "Match", get: (p) => p.location_prob ?? -1,
        render: (p) => (p.estimated ? <span className="text-muted-foreground">not drawn</span> : <Conf p={p.location_prob} />) },
      { key: "desc", label: "Play", get: (p) => p.desc ?? "", render: (p) => <span className="line-clamp-1 text-muted-foreground">{p.desc ?? "unmatched chart line"}</span>, cls: "max-w-[420px]" },
    )
    return list
  }, [kind])

  const rows = useMemo(() => {
    const c = cols.find((x) => x.key === sort.key) ?? cols[0]
    return [...plays].sort((a, b) => {
      const va = c.get(a)
      const vb = c.get(b)
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir
    })
  }, [plays, cols, sort])

  return (
    <ScrollArea className="h-[520px] rounded-xl border border-border/60">
      <Table>
        <TableHeader className="sticky top-0 z-10 bg-card/95 backdrop-blur">
          <TableRow>
            {cols.map((c) => (
              <TableHead key={c.key} className="cursor-pointer select-none text-xs whitespace-nowrap"
                onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? ((-s.dir) as 1 | -1) : 1 }))}>
                <span className="inline-flex items-center gap-1">
                  {c.label}
                  {sort.key === c.key && (sort.dir === 1 ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />)}
                </span>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((p) => (
            <TableRow key={p.id} onClick={() => onSelect(p.id)} data-state={p.id === selectedId ? "selected" : undefined}
              className={cn("cursor-pointer text-xs", !p.matched && "opacity-60")}>
              {cols.map((c) => (
                <TableCell key={c.key} className={cn("py-2", c.cls)}>
                  {c.render ? c.render(p) : String(c.get(p))}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </ScrollArea>
  )
}

function Outcome({ v }: { v: string }) {
  const map: Record<string, string> = {
    COMPLETE: "bg-emerald-400/15 text-emerald-700 dark:text-emerald-300",
    INCOMPLETE: "bg-zinc-400/15 text-zinc-700 dark:text-zinc-300",
    TOUCHDOWN: "bg-sky-400/15 text-sky-700 dark:text-sky-300",
    INTERCEPTION: "bg-rose-400/15 text-rose-700 dark:text-rose-300",
  }
  return <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-medium", map[v] ?? "bg-muted")}>{(v ?? "—").toLowerCase()}</span>
}

function Conf({ p }: { p: number | null }) {
  if (p == null) return <span className="text-amber-700 dark:text-amber-300">—</span>
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-1.5 w-10 overflow-hidden rounded-full bg-muted">
        <span className="block h-full rounded-full" style={{ width: `${100 * p}%`, background: confColor(p) }} />
      </span>
      <span className="font-mono text-muted-foreground">{Math.round(100 * p)}</span>
    </span>
  )
}
