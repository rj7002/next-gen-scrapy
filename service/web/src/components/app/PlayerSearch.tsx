import { useEffect, useMemo, useState } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Command, CommandDialog, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { KIND_LABEL, type Kind, type Player } from "@/lib/api"

export function primaryKind(p: Player): Kind {
  const pos = (p.position ?? "").toUpperCase()
  if (pos === "QB" && p.kinds.pass) return "pass"
  if (["RB", "HB", "FB"].includes(pos) && p.kinds.carry) return "carry"
  if (p.kinds.route) return "route"
  return (Object.keys(p.kinds)[0] as Kind) ?? "pass"
}

export function statLine(p: Player, kind: Kind): string {
  const k = p.kinds[kind]
  if (!k) return ""
  if (kind === "pass") return `${k.completions}/${k.attempts} · ${k.passingYards?.toLocaleString()} yds · ${k.touchdowns} TD · ${k.interceptions} INT`
  if (kind === "route") return `${k.receptions} rec · ${k.receivingYards?.toLocaleString()} yds · ${k.touchdowns} TD`
  return `${k.carries} car · ${k.rushingYards?.toLocaleString()} yds · ${k.touchdowns} TD`
}

export function initials(name: string) {
  return name
    .split(" ")
    .map((s) => s[0])
    .slice(0, 2)
    .join("")
}

export function PlayerSearch({ open, onOpenChange, season, players, onPick }: {
  open: boolean
  onOpenChange: (o: boolean) => void
  season: number
  players: Player[]
  onPick: (p: Player, kind: Kind) => void
}) {
  // Filter ourselves and render only the best few per group: cmdk would otherwise mount every player
  // (and fetch every headshot) each time the palette opens.
  const [q, setQ] = useState("")
  useEffect(() => {
    if (!open) setQ("")
  }, [open])
  const groups = useMemo(() => {
    const norm = (t: string) => t.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f.'’-]/g, "")
    const terms = norm(q).split(/\s+/).filter(Boolean)
    const score = (p: Player) => {
      const hay = norm(`${p.name} ${p.team ?? ""} ${p.position ?? ""}`)
      const words = hay.split(/\s+/)
      let n = 1
      for (const t of terms) {
        if (words.some((w) => w.startsWith(t))) n += 2
        else if (hay.includes(t)) n += 1
        else return 0
      }
      return n
    }
    const volume = (p: Player, k: Kind) => {
      const v = p.kinds[k]
      return (k === "pass" ? v?.attempts : k === "route" ? v?.receivingYards : v?.rushingYards) ?? 0
    }
    const cap = terms.length ? 12 : 5
    return ([["Quarterbacks", "pass"], ["Receivers", "route"], ["Running backs", "carry"]] as const).map(([label, k]) => {
      const list = players
        .filter((p) => primaryKind(p) === k)
        .map((p) => [p, score(p)] as const)
        .filter(([, n]) => n > 0)
        .sort((x, y) => y[1] - x[1] || volume(y[0], k) - volume(x[0], k))
        .slice(0, cap)
        .map(([p]) => p)
      return [terms.length ? label : `Top ${label.toLowerCase()}`, list] as const
    })
  }, [players, q])

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange} title="Find a player" description={`Every player with a ${season} chart`}
      className="sm:max-w-2xl">
      <Command shouldFilter={false}>
      <CommandInput value={q} onValueChange={setQ} placeholder={`Search ${players.length} players with ${season} charts...`} />
      <CommandList className="max-h-[60vh]">
        <CommandEmpty>No player with a {season} chart by that name.</CommandEmpty>
        {groups.map(([label, list]) =>
          list.length ? (
            <CommandGroup key={label} heading={label}>
              {list.map((p) => {
                const kind = primaryKind(p)
                return (
                  <CommandItem key={p.esb_id} value={`${p.name} ${p.team} ${p.position}`} onSelect={() => onPick(p, kind)}
                    className="gap-3 py-2">
                    <Avatar className="size-9 ring-1 ring-white/10">
                      <AvatarImage src={p.headshot ?? undefined} alt={p.name} className="object-cover" />
                      <AvatarFallback>{initials(p.name)}</AvatarFallback>
                    </Avatar>
                    <div className="flex min-w-0 flex-1 flex-col">
                      <span className="truncate font-medium">{p.name}</span>
                      <span className="truncate text-xs text-muted-foreground">{statLine(p, kind)}</span>
                    </div>
                    <div className="flex shrink-0 items-center gap-1.5">
                      {(Object.keys(p.kinds) as Kind[]).map((k) => (
                        <Badge key={k} variant={k === kind ? "secondary" : "outline"} className="text-[10px]">
                          {KIND_LABEL[k]}
                        </Badge>
                      ))}
                      <Badge variant="outline" className="font-mono text-[10px]">
                        {p.team} · {p.position}
                      </Badge>
                    </div>
                  </CommandItem>
                )
              })}
            </CommandGroup>
          ) : null
        )}
      </CommandList>
      </Command>
    </CommandDialog>
  )
}
