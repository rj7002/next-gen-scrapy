import { useMemo } from "react"
import { Search } from "lucide-react"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import type { Player } from "@/lib/api"
import { OUTCOME } from "@/lib/colors"

export function Landing({ season, players, onSearch }: { season: number; players: Player[]; onSearch: () => void }) {
  return (
    <section className="relative isolate flex min-h-[calc(100svh-3.5rem)] flex-col items-center overflow-hidden px-4">
      <FieldBackdrop />
      <div className="relative z-10 flex w-full max-w-2xl flex-1 flex-col items-center justify-center pb-[22vh] text-center">
        <h1 className="text-5xl font-semibold tracking-tight sm:text-7xl">
          <span className="text-emerald-600 dark:text-emerald-400">NFL Field</span> Explorer
        </h1>
        <p className="mt-4 text-balance text-muted-foreground sm:text-lg">
          Throws, routes and runs from Next Gen Stats charts, mapped and matched to play-by-play.
        </p>
        <button onClick={onSearch}
          className="mt-9 flex h-14 w-full max-w-xl items-center gap-3 rounded-2xl border border-border bg-card/80 px-5 text-left text-muted-foreground shadow-2xl shadow-black/20 backdrop-blur-md transition hover:border-emerald-500/40 hover:bg-card">
          <Search className="size-5 shrink-0" />
          <span className="flex-1 truncate">Search {players.length ? `${players.length} ` : ""}players with {season} charts</span>
          <KbdGroup className="hidden sm:inline-flex">
            <Kbd>⌘</Kbd>
            <Kbd>K</Kbd>
          </KbdGroup>
        </button>
      </div>
    </section>
  )
}

// ---- the backdrop: a QB's pass chart seen from behind the offense; throws arc out one by one and leave their ring ----
// Projected in code (a pinhole camera behind the line of scrimmage) rather than with a CSS 3D tilt, so every line
// stays sharp, and only opacity / stroke offsets are animated.

const CAM_BACK = 30     // yards behind the line of scrimmage
const CAM_UP = 20       // yards above the field
const FOCAL = 40
const NEAR = -10
const FAR = 55
const HALF = 26.67
const QB: [number, number] = [0, -7]

function project(x: number, y: number, z = 0): [number, number] {
  const d = y + CAM_BACK
  return [(x * FOCAL) / d, ((CAM_UP - z) * FOCAL) / d]
}

function path(pts: [number, number, number?][]) {
  return pts.map(([x, y, z], i) => {
    const [u, v] = project(x, y, z ?? 0)
    return `${i ? "L" : "M"}${u.toFixed(2)},${v.toFixed(2)}`
  }).join("")
}

// a deterministic, believable game of throws: mostly short, some intermediate, a few shots downfield
function passes() {
  let seed = 7
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  const out: { x: number; y: number; type: string }[] = []
  for (let i = 0; i < 46; i++) {
    const r = rnd()
    const y = r < 0.12 ? -4 + rnd() * 4 : r < 0.62 ? rnd() * 10 : r < 0.9 ? 10 + rnd() * 10 : 20 + rnd() * 25
    const spread = Math.min(HALF - 2, 9 + y * 0.45)
    const x = (rnd() * 2 - 1) * spread
    const t = rnd()
    const type = y > 18 && t < 0.18 ? "TOUCHDOWN" : t < 0.03 ? "INTERCEPTION" : t < (y > 15 ? 0.55 : 0.33) ? "INCOMPLETE" : "COMPLETE"
    out.push({ x, y, type })
  }
  return out
}

function FieldBackdrop() {
  const geo = useMemo(() => {
    const lines: { d: string; major: boolean }[] = []
    for (let y = -10; y <= FAR; y += 5) if (y) lines.push({ d: path([[-HALF, y], [HALF, y]]), major: y % 10 === 0 })
    const sides = [path([[-HALF, NEAR], [-HALF, FAR]]), path([[HALF, NEAR], [HALF, FAR]])]
    const hashes: string[] = []
    for (let y = NEAR + 1; y < FAR; y++) for (const hx of [-3.08, 3.08]) hashes.push(path([[hx - 0.35, y], [hx + 0.35, y]]))
    const turf = path([[-HALF, NEAR], [HALF, NEAR], [HALF, FAR], [-HALF, FAR]]) + "Z"
    const throws = passes().map((p, i) => {
      const dist = Math.hypot(p.x - QB[0], p.y - QB[1])
      const apex = 1.5 + dist / 7
      const arc: [number, number, number][] = []
      for (let k = 0; k <= 24; k++) {
        const t = k / 24
        arc.push([QB[0] + (p.x - QB[0]) * t, QB[1] + (p.y - QB[1]) * t, 1.8 + 4 * apex * t * (1 - t) - 1.8 * t])
      }
      const [u, v] = project(p.x, p.y)
      const d = p.y + CAM_BACK
      const rx = (0.72 * FOCAL) / d
      return { ...p, arc: path(arc), u, v, rx, ry: rx * (CAM_UP / Math.hypot(CAM_UP, d)), w: (0.13 * FOCAL) / d, delay: i * 0.26 }
    })
    return { lines, sides, hashes, turf, los: path([[-HALF, 0], [HALF, 0]]), throws }
  }, [])
  const top = project(0, FAR)[1] - 2
  const bottom = project(0, NEAR)[1]
  return (
    <div aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-[72%]
      [mask-image:linear-gradient(to_bottom,transparent,black_30%,black_85%,transparent)]">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_100%,rgba(16,185,129,0.12),transparent_65%)]" />
      <svg viewBox={`-50 ${top} 100 ${bottom - top}`} preserveAspectRatio="xMidYMax slice" className="absolute inset-0 size-full text-foreground">
        <path d={geo.turf} className="fill-foreground/[0.03]" />
        {geo.lines.map((ln, i) => (
          <path key={i} d={ln.d} stroke="currentColor" strokeOpacity={ln.major ? 0.13 : 0.06} strokeWidth={0.08} fill="none" />
        ))}
        {geo.hashes.map((d, i) => <path key={`h${i}`} d={d} stroke="currentColor" strokeOpacity={0.1} strokeWidth={0.07} />)}
        {geo.sides.map((d, i) => <path key={`s${i}`} d={d} stroke="currentColor" strokeOpacity={0.18} strokeWidth={0.12} />)}
        <path d={geo.los} stroke="#60a5fa" strokeOpacity={0.75} strokeWidth={0.16} />
        {geo.throws.map((t, i) => {
          const incomplete = t.type === "INCOMPLETE"
          const color = incomplete ? "currentColor" : OUTCOME[t.type]
          return (
            <g key={i} className={incomplete ? "text-zinc-400 dark:text-zinc-300" : undefined}>
              <path d={t.arc} pathLength={1} fill="none" stroke="currentColor" strokeOpacity={0.35} strokeWidth={t.w * 0.6}
                strokeLinecap="round" className="hero-throw text-foreground" style={{ animationDelay: `${t.delay}s` }} />
              <g className="hero-ring" style={{ animationDelay: `${t.delay}s` }}>
                <ellipse cx={t.u} cy={t.v + t.ry * 0.35} rx={t.rx * 1.05} ry={t.ry * 1.05} className="fill-black/10 dark:fill-black/40" />
                <ellipse cx={t.u} cy={t.v} rx={t.rx} ry={t.ry} fill={color} fillOpacity={0.18} stroke={color} strokeWidth={t.w} />
                <ellipse cx={t.u} cy={t.v} rx={t.rx * 0.4} ry={t.ry * 0.4} fill={color} fillOpacity={0.9} />
              </g>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
