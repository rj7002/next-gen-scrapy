import { useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { type Kind, type Play, unmatchedText } from "@/lib/api"
import type { HeatImage } from "@/lib/exportChart"
import { type ColorBy, heat, playColor, ROUTE_LINE } from "@/lib/colors"
import { situation } from "@/lib/stats"
import { cn } from "@/lib/utils"
import { PlayFacts, playWhen } from "./PlayDetail"

const XS = 26.67 // sideline
const OOB = 4.6 // drawn out-of-bounds margin
const HASH = 3.08 // NFL hash marks: 6.17 yd apart
const XL = -XS - OOB
const W = 2 * (XS + OOB)

export type FieldView = "plays" | "heat"

/** What the export needs from the field: its SVG, and the heatmap for the plays on it (made on demand). */
export interface FieldHandle {
  svg: () => SVGSVGElement | null
  heat: () => HeatImage | null
}

interface Props {
  kind: Kind
  plays: Play[]
  allPlays: Play[]
  colorBy: ColorBy
  view: FieldView
  heatOf: "end" | "path"
  threeD: boolean
  selectedId: string | null
  onSelect: (id: string | null) => void
  animateKey: string
  fullDepth: boolean
  onDeeper?: (n: number) => void
  ref?: React.Ref<FieldHandle>
}

function ys(p: Play, kind: Kind): number[] {
  if (kind === "pass") return p.y_coord != null ? [p.y_coord] : []
  return (p.path_y as number[]) ?? []
}

/** the point a play "lands" at: pass dot, route catch point (or end), carry end */
export function landing(p: Play, kind: Kind): [number, number] | null {
  if (kind === "pass") return p.x_coord != null ? [p.x_coord, p.y_coord] : null
  const xs: number[] = p.path_x ?? []
  const yv: number[] = p.path_y ?? []
  if (!xs.length) return null
  if (kind === "route") {
    const seg: string[] = p.path_segment ?? []
    let i = xs.length - 1
    const first = seg.indexOf("after_catch")
    if (first > 0) i = first - 1
    return [xs[i], yv[i]]
  }
  return [xs[xs.length - 1], yv[yv.length - 1]]
}

function polyline(xs: number[], yv: number[], top: number, from = 0, to = xs.length) {
  let d = ""
  for (let i = from; i < to; i++) d += `${i === from ? "M" : "L"}${xs[i].toFixed(2)} ${(top - yv[i]).toFixed(2)}`
  return d
}

const DEFAULT_CAM = { tilt: 40, yaw: 0, zoom: 1, panX: 0, panY: 0 }
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

export function Field(props: Props) {
  const { kind, plays, allPlays, colorBy, view, heatOf, threeD, selectedId, onSelect, animateKey, fullDepth } = props
  const wrap = useRef<HTMLDivElement>(null)
  // hover position is in viewport coordinates: the card is portalled to <body> so the field never clips it
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null)
  const canHover = useMemo(() => window.matchMedia("(hover: hover)").matches, [])
  // 3D tilt: rotate about the bottom edge with the perspective on the field itself, then pull the
  // field up by however much the tilt shortened it so no empty band is left above.
  const svgRef = useRef<SVGSVGElement>(null)
  const [svgH, setSvgH] = useState(0)
  const [svgW, setSvgW] = useState(0)
  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => (setSvgH(e.contentRect.height), setSvgW(e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const persp = 1.9 * (svgH || 600)
  const tilt0 = (DEFAULT_CAM.tilt * Math.PI) / 180
  const lift = threeD && svgH ? svgH - (svgH * Math.cos(tilt0) * persp) / (persp + svgH * Math.sin(tilt0)) - 10 : 0

  // 3D camera: drag to orbit (sideways spins the field all the way round, up/down tilts it), shift-drag to pan,
  // pinch or ⌘/ctrl + scroll to zoom, double-click to reset
  const [cam, setCam] = useState(DEFAULT_CAM)
  const [dragging, setDragging] = useState(false)
  const drag = useRef<{ x: number; y: number; cam: typeof DEFAULT_CAM; pan: boolean; moved: boolean } | null>(null)
  const justDragged = useRef(false)
  useEffect(() => {
    if (!threeD) setCam(DEFAULT_CAM)
  }, [threeD])
  useEffect(() => {
    const el = wrap.current
    if (!el || !threeD) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return       // plain scrolling still scrolls the page
      e.preventDefault()
      setCam((c) => ({ ...c, zoom: clamp(c.zoom * Math.exp(-e.deltaY * 0.01), 0.5, 3) }))
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [threeD])
  const camHandlers = threeD
    ? {
        onPointerDown: (e: React.PointerEvent) => {
          if (e.button !== 0) return
          drag.current = { x: e.clientX, y: e.clientY, cam, pan: e.shiftKey, moved: false }
        },
        onPointerMove: (e: React.PointerEvent) => {
          const d = drag.current
          if (!d) return
          const dx = e.clientX - d.x
          const dy = e.clientY - d.y
          if (!d.moved && Math.hypot(dx, dy) < 4) return
          if (!d.moved) {
            d.moved = true
            setDragging(true)
            setHover(null)
            ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
          }
          setCam(d.pan
            ? { ...d.cam, panX: d.cam.panX + dx, panY: d.cam.panY + dy }
            : { ...d.cam, yaw: d.cam.yaw - dx * 0.36, tilt: clamp(d.cam.tilt + dy * 0.2, 0, 80) })   // yaw: no limit, 360°
        },
        onPointerUp: () => {
          if (drag.current?.moved) {
            justDragged.current = true          // the click that ends a drag isn't a click on a play
            setTimeout(() => (justDragged.current = false), 0)
          }
          drag.current = null
          setDragging(false)
        },
        // back to straight on - from the nearest full turn, so it doesn't unwind every lap it was spun
        onDoubleClick: () => setCam((c) => ({ ...DEFAULT_CAM, yaw: Math.round(c.yaw / 360) * 360 })),
        onClickCapture: (e: React.MouseEvent) => {
          if (justDragged.current) {
            e.stopPropagation()
            e.preventDefault()
          }
        },
      }
    : {}

  // Bounds come from the whole dataset, so light filtering doesn't rescale the field. By default the view covers
  // where 95% of the plays end (a single 80-yard run shouldn't squash every other carry into a sliver); a filter
  // that keeps mostly deep plays stretches it to fit them, and "full depth" shows everything.
  const [ymin, ymax, deeper] = useMemo(() => {
    const extent = (list: Play[]) => {
      const tops: number[] = []
      const lows: number[] = []
      for (const p of list) {
        const v = ys(p, kind)
        if (!v.length) continue
        tops.push(Math.max(...v))
        lows.push(Math.min(...v))
      }
      return [tops.sort((a, b) => a - b), lows.sort((a, b) => a - b)]
    }
    const [tops, lows] = extent(allPlays)
    if (!tops.length) return [-10, 25, 0]
    const q = (a: number[], f: number) => a[Math.min(a.length - 1, Math.max(0, Math.floor(f * (a.length - 1))))]
    const lo = Math.max(-15, Math.floor(Math.min(-3, q(lows, 0.01)) - 2))
    const full = Math.min(80, Math.max(22, Math.ceil(tops[tops.length - 1] + 3.5)))
    let focus = Math.min(full, Math.max(22, Math.ceil(q(tops, 0.95) + 6)))
    const [shown] = plays.length < allPlays.length ? extent(plays) : [tops]
    if (shown !== tops && shown.length)
      focus = Math.min(full, Math.max(focus, Math.ceil(q(shown, shown.length < 20 ? 1 : 0.95) + 6)))
    const hi = fullDepth ? full : focus
    return [lo, hi, shown.filter((t) => t > hi).length]
  }, [allPlays, plays, kind, fullDepth])
  const reported = useRef(-1)
  if (reported.current !== deeper) {
    reported.current = deeper
    queueMicrotask(() => props.onDeeper?.(deeper))
  }
  const H = ymax - ymin
  const Y = (y: number) => ymax - y

  const heatImg = useMemo(() => (view === "heat" ? heatmap(plays, kind, heatOf, ymin, ymax) : null), [view, plays, kind, heatOf, ymin, ymax])
  // in 3D the heatmap rises off the field: stacked slices, each lifted by the density it marks
  const heatStack = useMemo(() => (view === "heat" && threeD ? heatSlices(plays, kind, heatOf, ymin, ymax) : null),
    [view, threeD, plays, kind, heatOf, ymin, ymax])
  // the SVG letterboxes its viewBox (max height): the slices sit exactly over the drawn field
  const fit = Math.min(svgW / W, svgH / (ymax - ymin)) || 0
  const box = { w: W * fit, h: (ymax - ymin) * fit, x: (svgW - W * fit) / 2, y: (svgH - (ymax - ymin) * fit) / 2 }
  const peak = 0.12 * box.w
  useImperativeHandle(props.ref, () => ({
    svg: () => svgRef.current,
    heat: () => heatImg ?? heatmap(plays, kind, heatOf, ymin, ymax),
  }), [heatImg, plays, kind, heatOf, ymin, ymax])

  const byId = useMemo(() => new Map(plays.map((p) => [p.id, p])), [plays])
  const hovered = hover ? byId.get(hover.id) : null
  const focus = selectedId ?? hover?.id ?? null
  const many = plays.length > 150
  const ordered = useMemo(() => {
    // incompletes / unmatched underneath, the focused play on top
    const rank = (p: Play) => (p.id === focus ? 3 : !p.matched ? 0 : p.route_type === "INCOMPLETE" || p.pass_type === "INCOMPLETE" ? 1 : 2)
    return [...plays].sort((a, b) => rank(a) - rank(b))
  }, [plays, focus])

  const yardLines = []
  for (let y = Math.ceil(ymin); y <= Math.floor(ymax); y++) {
    const major = y % 5 === 0
    if (major) {
      yardLines.push(
        <line key={`l${y}`} x1={-XS} x2={XS} y1={Y(y)} y2={Y(y)} stroke="white"
          strokeOpacity={y % 10 === 0 ? 0.2 : 0.11} vectorEffect="non-scaling-stroke" strokeWidth={1} />
      )
    }
    for (const hx of [-XS + 0.9, -HASH, HASH, XS - 0.9]) {
      yardLines.push(
        <line key={`h${y}${hx}`} x1={hx - 0.35} x2={hx + 0.35} y1={Y(y)} y2={Y(y)} stroke="white" strokeOpacity={0.16}
          vectorEffect="non-scaling-stroke" strokeWidth={1} />
      )
    }
    if (y % 10 === 0 && y !== 0 && y - ymin > 1 && ymax - y > 1) {
      for (const side of [-1, 1]) {
        yardLines.push(
          <text key={`n${y}${side}`} x={side * (XS + 2.45)} y={Y(y) + 0.34} fill="white" fillOpacity={0.3}
            fontSize={0.95} fontWeight={500} textAnchor="middle" className="select-none font-mono">
            {y > 0 ? `+${y}` : y}
          </text>
        )
      }
    }
  }

  return (
    <div
      ref={wrap}
      className={cn("relative w-full overflow-hidden", threeD && ["select-none", dragging ? "cursor-grabbing" : "cursor-grab"])}
      style={{ touchAction: threeD ? "none" : undefined }}
      {...camHandlers}
      // functional update: this mousemove fires after the new play's mouseenter in the same event, and a stale
      // `hover` here would put the previous play back
      onMouseMove={(e) => !dragging && setHover((h) => (h ? { ...h, x: e.clientX, y: e.clientY } : h))}
      onMouseLeave={() => setHover(null)}
    >
      {/* outer: pan, then tilt about the bottom edge (keeps the default framing); inner: turn and zoom about the
          middle of the field, in its own plane, so it spins in place */}
      <div
        className={cn(!dragging && "transition-all duration-700 ease-[cubic-bezier(.2,.8,.2,1)]")}
        style={{
          transform: threeD
            ? `perspective(${persp}px) translate(${cam.panX}px, ${cam.panY}px) rotateX(${cam.tilt}deg)`
            : `perspective(${persp}px) translate(0px, 0px) rotateX(0deg)`,
          transformOrigin: "50% 100%", marginTop: -lift, transformStyle: "preserve-3d",
        }}
      >
       <div
        className={cn("relative", !dragging && "transition-transform duration-700 ease-[cubic-bezier(.2,.8,.2,1)]")}
        style={{ transform: threeD ? `rotate(${cam.yaw}deg) scale(${cam.zoom})` : "rotate(0deg) scale(1)", transformOrigin: "50% 60%",
          transformStyle: "preserve-3d" }}
       >
        <svg ref={svgRef} viewBox={`${XL} 0 ${W} ${H}`} className="block h-auto max-h-[72vh] w-full" onClick={() => onSelect(null)}>
          <defs>
            <linearGradient id="turf" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#15181d" />
              <stop offset="1" stopColor="#1d2128" />
            </linearGradient>
            <linearGradient id="oob" x1="0" x2="1">
              <stop offset="0" stopColor="#0b0c0f" />
              <stop offset="1" stopColor="#111317" />
            </linearGradient>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="0.45" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <rect x={XL} y={0} width={W} height={H} fill="url(#oob)" />
          <rect x={-XS} y={0} width={2 * XS} height={H} fill="url(#turf)" />
          {/* sideline bands */}
          <rect x={-XS - 1.1} y={0} width={1.1} height={H} fill="#3f4450" opacity={0.55} />
          <rect x={XS} y={0} width={1.1} height={H} fill="#3f4450" opacity={0.55} />
          {yardLines}
          {/* line of scrimmage */}
          {ymin < 0 && ymax > 0 && (
            <>
              <line x1={-XS} x2={XS} y1={Y(0)} y2={Y(0)} stroke="#3b82f6" strokeWidth={6} strokeOpacity={0.25}
                vectorEffect="non-scaling-stroke" />
              <line x1={-XS} x2={XS} y1={Y(0)} y2={Y(0)} stroke="#60a5fa" strokeWidth={2.2} vectorEffect="non-scaling-stroke" />
              {[-1, 1].map((s) => (
                <text key={s} x={s * (XS + 2.45)} y={Y(0) + 0.34} fill="#93c5fd" fillOpacity={0.8} fontSize={0.95}
                  fontWeight={700} textAnchor="middle" className="select-none">
                  LOS
                </text>
              ))}
            </>
          )}

          {heatImg && (
            <image data-layer="heat" href={heatImg.url} x={heatImg.x} y={heatImg.y} width={heatImg.w} height={heatImg.h}
              preserveAspectRatio="none" style={{ mixBlendMode: "screen" }} />
          )}

          <g key={animateKey} data-layer="plays" opacity={view === "heat" ? 0.18 : 1}>
            {ordered.map((p, i) => (
              <PlayMark key={p.id} play={p} kind={kind} colorBy={colorBy} top={ymax} index={i}
                dim={focus != null && p.id !== focus} focused={p.id === focus} many={many}
                onEnter={(e) => setHover({ id: p.id, x: e.clientX, y: e.clientY })}
                onLeave={() => setHover((h) => (h?.id === p.id ? null : h))}
                onClick={() => onSelect(p.id === selectedId ? null : p.id)}
                animate={!threeD}
              />
            ))}
          </g>
        </svg>
        {heatStack && box.w > 0 && (
          <div className="pointer-events-none absolute" aria-hidden
            style={{ left: box.x, top: box.y, width: box.w, height: box.h, transformStyle: "preserve-3d" }}>
            {heatStack.map((url, i) => (
              <img key={i} src={url} alt="" draggable={false} className="absolute inset-0 size-full"
                style={{ transform: `translateZ(${((i + 1) / heatStack.length) * peak}px)` }} />
            ))}
          </div>
        )}
       </div>
      </div>

      {threeD && (
        <div className="pointer-events-none absolute right-3 bottom-3 rounded-md bg-black/50 px-2 py-1 text-[10px] text-zinc-400 backdrop-blur-sm">
          Drag to spin 360° and tilt · Shift-drag to pan · Pinch or ⌘-scroll to zoom · Double-click to reset
        </div>
      )}
      {hovered && hover && canHover && !dragging && <PlayHoverCard play={hovered} kind={kind} x={hover.x} y={hover.y} />}
    </div>
  )
}

function PlayMark({ play, kind, colorBy, top, index, dim, focused, many, onEnter, onLeave, onClick, animate }: {
  play: Play
  kind: Kind
  colorBy: ColorBy
  top: number
  index: number
  dim: boolean
  focused: boolean
  many: boolean
  onEnter: (e: React.MouseEvent) => void
  onLeave: () => void
  onClick: () => void
  animate: boolean
}) {
  const color = playColor(play, kind, colorBy)
  const opacity = dim ? 0.12 : focused ? 1 : many ? 0.62 : 0.85
  const width = focused ? 0.3 : many ? 0.12 : 0.16 // yards
  const delay = `${Math.min(index * 6, 900)}ms`
  const handlers = {
    onMouseEnter: onEnter,
    onMouseLeave: onLeave,
    onClick: (e: React.MouseEvent) => {
      e.stopPropagation()
      onClick()
    },
    className: "cursor-pointer",
  }
  const dash = play.matched ? undefined : "0.5 0.4"

  if (kind === "pass" && play.estimated) {
    // not drawn by NGS: a faint dashed ring at the spot estimated from play-by-play
    if (play.x_coord == null || play.y_coord == null) return null
    return (
      <g {...handlers} opacity={dim ? 0.12 : focused ? 1 : 0.55}>
        <circle cx={play.x_coord} cy={top - play.y_coord} r={1.4} fill="transparent" />
        <circle cx={play.x_coord} cy={top - play.y_coord} r={focused ? 0.85 : many ? 0.42 : 0.55} fill="none" stroke={color}
          strokeWidth={focused ? 0.18 : 0.1} strokeDasharray="0.22 0.16" />
      </g>
    )
  }

  if (kind === "pass") {
    if (play.x_coord == null) return null
    const cx = play.x_coord
    const cy = top - play.y_coord
    return (
      <g {...handlers} opacity={opacity} filter={focused ? "url(#glow)" : undefined}>
        <circle cx={cx} cy={cy} r={1.4} fill="transparent" />
        <circle cx={cx} cy={cy} r={focused ? 0.85 : many ? 0.42 : 0.55} fill={color} fillOpacity={0.2} stroke={color}
          strokeWidth={focused ? 0.22 : 0.13} strokeDasharray={dash ? "0.25 0.2" : undefined}
          className={animate ? "pop-in" : undefined} style={{ animationDelay: delay }}
          onAnimationEnd={(e) => e.currentTarget.classList.remove("pop-in")} />
      </g>
    )
  }

  const xs: number[] = play.path_x ?? []
  const yv: number[] = play.path_y ?? []
  if (xs.length < 2) return null
  const seg: string[] = play.path_segment ?? []
  const firstAc = kind === "route" ? seg.indexOf("after_catch") : -1
  const split = firstAc > 0 ? firstAc : xs.length
  const pre = polyline(xs, yv, top, 0, split)
  const post = firstAc > 0 ? polyline(xs, yv, top, firstAc - 1, xs.length) : null
  const end = [xs[xs.length - 1], top - yv[yv.length - 1]]
  const td = play.chart_touchdown === true || play.chart_touchdown === 1
  const afterColor = colorBy === "outcome" ? ROUTE_LINE.after : color
  const line = (d: string, c: string, key: string, w = width) => (
    <path key={key} d={d} fill="none" stroke={c} strokeWidth={w} strokeLinecap="round" strokeLinejoin="round"
      pathLength={dash ? undefined : 1} strokeDasharray={dash} className={dash ? undefined : "draw-in"}
      style={{ animationDelay: delay }} />
  )
  return (
    <g {...handlers} opacity={opacity} filter={focused ? "url(#glow)" : undefined}>
      <path d={polyline(xs, yv, top)} fill="none" stroke="transparent" strokeWidth={12} vectorEffect="non-scaling-stroke" />
      {line(pre, color, "pre")}
      {post && line(post, afterColor, "post", width * 0.92)}
      {kind === "route" && firstAc > 0 && (
        <circle cx={xs[firstAc - 1]} cy={top - yv[firstAc - 1]} r={0.32} fill={afterColor} />
      )}
      {td && <circle cx={end[0]} cy={end[1]} r={0.85} fill="none" stroke={ROUTE_LINE.td} strokeWidth={0.2} />}
      {kind === "carry" && (play.chart_fumble_lost === true || play.chart_fumble_lost === 1) && (
        <circle cx={end[0]} cy={end[1]} r={0.85} fill="none" stroke="#f87171" strokeWidth={0.2} />
      )}
    </g>
  )
}

/** The whole play, the moment the pointer is on it. Sits beside the cursor, flipping sides near the window edges. */
function PlayHoverCard({ play, kind, x, y }: { play: Play; kind: Kind; x: number; y: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const [h, setH] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (el && el.offsetHeight !== h) setH(el.offsetHeight)
  })
  const w = 340
  const gap = 18
  const pad = 8
  const left = x + gap + w + pad <= window.innerWidth ? x + gap : Math.max(pad, x - gap - w)
  const top = Math.min(Math.max(pad, y - 48), Math.max(pad, window.innerHeight - h - pad))
  const td = play.chart_touchdown === true || play.chart_touchdown === 1 || play.pass_type === "TOUCHDOWN"
  const outcome =
    kind === "pass" ? play.pass_type : kind === "route" ? (td ? "touchdown" : play.route_type) : td ? "touchdown" : null
  const yards = play.yards_gained
  return createPortal(
    <div ref={ref} style={{ left, top, width: w, visibility: h ? "visible" : "hidden" }}
      className="dark pointer-events-none fixed z-50 max-h-[calc(100vh-16px)] overflow-hidden rounded-xl border border-white/10 bg-zinc-950/95 p-3.5 text-xs text-foreground shadow-2xl backdrop-blur-md">
      <div className="mb-1 flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
        <span className="truncate">{playWhen(play)}</span>
        {!play.matched && <span className="shrink-0 text-amber-300">{play.reason === "no_fit" ? "no play fits" : "unmatched"}</span>}
        {play.estimated && <span className="shrink-0 text-zinc-400">not drawn by NGS</span>}
      </div>
      {play.matched ? (
        <>
          <div className="mb-1 flex flex-wrap items-center gap-1.5 text-sm font-medium">
            {situation(play) && <span>{situation(play)}</span>}
            {outcome && (
              <span className="rounded bg-white/10 px-1.5 py-0.5 text-[11px] font-normal">{String(outcome).toLowerCase().replace("_", " ")}</span>
            )}
            {yards != null && (
              <span className="font-mono text-[11px] font-normal text-muted-foreground">
                {yards > 0 ? "+" : ""}
                {yards} yds
              </span>
            )}
          </div>
          {play.desc && <p className="mb-3 line-clamp-3 leading-snug text-zinc-300">{play.desc}</p>}
          <PlayFacts play={play} kind={kind} compact />
          <div className="mt-3 text-[10px] text-muted-foreground">Click to pin it open</div>
        </>
      ) : (
        <p className="text-zinc-400">{unmatchedText(play)}</p>
      )}
    </div>,
    document.body,
  )
}

function heatmap(plays: Play[], kind: Kind, heatOf: "end" | "path", ymin: number, ymax: number) {
  const d = density(plays, kind, heatOf, ymin, ymax)
  if (!d) return null
  const { t, nx, ny } = d
  const canvas = document.createElement("canvas")
  canvas.width = nx
  canvas.height = ny
  const ctx = canvas.getContext("2d")!
  const img = ctx.createImageData(nx, ny)
  for (let i = 0; i < t.length; i++) {
    const [r, g, b] = heat(t[i])
    img.data[4 * i] = r
    img.data[4 * i + 1] = g
    img.data[4 * i + 2] = b
    img.data[4 * i + 3] = Math.round(255 * Math.min(1, t[i] * 1.6) * 0.92)
  }
  ctx.putImageData(img, 0, 0)
  return { url: canvas.toDataURL(), x: XL, y: 0, w: W, h: ymax - ymin }
}

/** The 3D heatmap: SLICES images, slice k showing where the density reaches its level, coloured for that level
 * and soft at its edge - stacked at heights in proportion they make a contoured mound of the density. The faint
 * spread below FLOOR stays on the flat heatmap underneath, so the field's lines still show through. */
const SLICES = 36
const FLOOR = 0.14
function heatSlices(plays: Play[], kind: Kind, heatOf: "end" | "path", ymin: number, ymax: number): string[] | null {
  const d = density(plays, kind, heatOf, ymin, ymax)
  if (!d) return null
  const { t, nx, ny } = d
  const canvas = document.createElement("canvas")
  canvas.width = nx
  canvas.height = ny
  const ctx = canvas.getContext("2d")!
  const out: string[] = []
  for (let k = 1; k <= SLICES; k++) {
    const level = FLOOR + ((1 - FLOOR) * k) / SLICES
    const [r, g, b] = heat(level)
    const shade = 0.72 + 0.28 * level              // lower slices a touch darker: reads as depth from the side
    const opacity = 0.5 + 0.45 * (k / SLICES)      // and the lower ones partly see-through
    const img = ctx.createImageData(nx, ny)
    let any = false
    for (let i = 0; i < t.length; i++) {
      const a = Math.min(1, Math.max(0, ((t[i] - level) * SLICES) / (1 - FLOOR) + 0.5))
      if (a <= 0) continue
      any = true
      img.data[4 * i] = r * shade
      img.data[4 * i + 1] = g * shade
      img.data[4 * i + 2] = b * shade
      img.data[4 * i + 3] = Math.round(255 * a * opacity)
    }
    if (!any) break
    ctx.putImageData(img, 0, 0)
    out.push(canvas.toDataURL())
  }
  return out
}

/** Smoothed density of the plays' end points (or whole paths) on a half-yard grid, scaled 0-1. */
function density(plays: Play[], kind: Kind, heatOf: "end" | "path", ymin: number, ymax: number) {
  const res = 0.5
  const nx = Math.round(W / res)
  const ny = Math.round((ymax - ymin) / res)
  const grid = new Float32Array(nx * ny)
  const sigma = heatOf === "path" ? 1.2 : 1.7
  const rad = Math.ceil((3 * sigma) / res)
  const ker: number[] = []
  for (let dy = -rad; dy <= rad; dy++)
    for (let dx = -rad; dx <= rad; dx++) ker.push(Math.exp(-((dx * res) ** 2 + (dy * res) ** 2) / (2 * sigma * sigma)))
  const add = (x: number, y: number, w: number) => {
    const cx = Math.round((x - XL) / res)
    const cy = Math.round((ymax - y) / res)
    let k = 0
    for (let dy = -rad; dy <= rad; dy++)
      for (let dx = -rad; dx <= rad; dx++, k++) {
        const gx = cx + dx
        const gy = cy + dy
        if (gx >= 0 && gx < nx && gy >= 0 && gy < ny) grid[gy * nx + gx] += ker[k] * w
      }
  }
  for (const p of plays) {
    if (heatOf === "path" && kind !== "pass") {
      const xs: number[] = p.path_x ?? []
      const yv: number[] = p.path_y ?? []
      const step = 2
      const w = 1 / Math.max(1, xs.length / step)
      for (let i = 0; i < xs.length; i += step) add(xs[i], yv[i], w)
    } else {
      const l = landing(p, kind)
      if (l) add(l[0], l[1], 1)
    }
  }
  let max = 0
  for (const v of grid) max = Math.max(max, v)
  if (max <= 0) return null
  const t = new Float32Array(grid.length)
  for (let i = 0; i < grid.length; i++) t[i] = Math.pow(grid[i] / max, 0.65)
  return { t, nx, ny }
}
