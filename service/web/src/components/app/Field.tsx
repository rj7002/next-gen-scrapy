import { useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { type Kind, type Play, unmatchedText } from "@/lib/api"
import type { HeatImage } from "@/lib/exportChart"
import { type ColorBy, playColor, ROUTE_LINE } from "@/lib/colors"
import { COMPARE_A, COMPARE_B, type HeatMetric, heatField, heatImage, heatSlices, isVersus, splitOf } from "@/lib/heat"
import { kpis, situation } from "@/lib/stats"
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
  heatMetric: HeatMetric
  heatSplit: string
  /** route charts: draw the run after the catch (off: each route stops at the catch) */
  routeYac?: boolean
  /** compare against another player / season: their plays (same filters) and both sides' names */
  versus?: { mine: string; other: string; plays: Play[] } | null
  yac: boolean
  threeD: boolean
  selectedId: string | null
  onSelect: (id: string | null) => void
  animateKey: string
  fullDepth: boolean
  onDeeper?: (n: number) => void
  ref?: React.Ref<FieldHandle>
}

function ys(p: Play, kind: Kind, yac = false): number[] {
  if (kind === "pass") {
    if (p.y_coord == null) return []
    const end = yac ? yacEnd(p) : null
    return end != null ? [p.y_coord, end] : [p.y_coord]
  }
  return (p.path_y as number[]) ?? []
}

/** where a completion's run after the catch ended (straight downfield from the catch: pbp has no sideways spot) */
export function yacEnd(p: Play): number | null {
  if (p.y_coord == null || !(p.pass_type === "COMPLETE" || p.pass_type === "TOUCHDOWN")) return null
  const yac = p.yards_after_catch
  if (yac != null && yac !== "" && !Number.isNaN(Number(yac))) return p.y_coord + Number(yac)
  return p.yards_gained != null && !Number.isNaN(Number(p.yards_gained)) ? Number(p.yards_gained) : null
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
  const { kind, plays, allPlays, colorBy, view, heatOf, heatMetric, heatSplit, versus, yac, routeYac = true, threeD, selectedId, onSelect, animateKey, fullDepth } = props
  const vsPlays = heatMetric === "compare" && isVersus(heatSplit) ? versus?.plays ?? null : null
  const showYac = yac && kind === "pass" && view === "plays"
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
        const v = ys(p, kind, showYac)
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
  }, [allPlays, plays, kind, fullDepth, showYac])
  const reported = useRef(-1)
  if (reported.current !== deeper) {
    reported.current = deeper
    queueMicrotask(() => props.onDeeper?.(deeper))
  }
  const H = ymax - ymin
  const Y = (y: number) => ymax - y

  const heatF = useMemo(() => (view === "heat" ? heatField(plays, kind, heatOf, heatMetric, heatSplit, XL, W, ymin, ymax, landing, vsPlays) : null),
    [view, plays, kind, heatOf, heatMetric, heatSplit, vsPlays, ymin, ymax])
  const heatImg = useMemo(() => (heatF ? { url: heatImage(heatF), x: XL, y: 0, w: W, h: ymax - ymin } : null), [heatF, ymin, ymax])
  // in 3D the heatmap rises off the field: stacked slices, each lifted by the density it marks
  const heatStack = useMemo(() => (heatF && threeD ? heatSlices(heatF) : null), [heatF, threeD])
  // the SVG letterboxes its viewBox (max height): the slices sit exactly over the drawn field
  const fit = Math.min(svgW / W, svgH / (ymax - ymin)) || 0
  const box = { w: W * fit, h: (ymax - ymin) * fit, x: (svgW - W * fit) / 2, y: (svgH - (ymax - ymin) * fit) / 2 }
  const peak = 0.12 * box.w

  // heatmap hover: the plays around the spot under the cursor, summed up. The cursor is mapped back onto the field
  // through wherever the field's corners are on screen (a homography), so it holds at any 3D spin, tilt and zoom.
  const corners = useRef<(HTMLDivElement | null)[]>([])
  const [spot, setSpot] = useState<{ x: number; y: number; cx: number; cy: number } | null>(null)
  const radius = heatOf === "path" && kind !== "pass" ? 2.5 : 4
  const fieldPoint = (cx: number, cy: number): [number, number] | null => {
    const pts = corners.current.map((el) => el?.getBoundingClientRect())
    if (pts.some((r) => !r)) return null
    const toUnit = homography(pts.map((r) => [r!.left, r!.top] as [number, number]), [[0, 0], [1, 0], [1, 1], [0, 1]])
    if (!toUnit) return null
    const [u, v] = toUnit(cx, cy)
    if (!(u >= 0 && u <= 1 && v >= 0 && v <= 1)) return null
    return [XL + u * W, ymax - v * (ymax - ymin)]
  }
  const around = (list: Play[]) => {
    if (!spot) return []
    const near = (x: number, y: number) => (x - spot.x) ** 2 + (y - spot.y) ** 2 <= radius * radius
    return list.filter((p) => {
      if (heatOf === "path" && kind !== "pass") {
        const xs: number[] = p.path_x ?? []
        const yv: number[] = p.path_y ?? []
        return xs.some((x, i) => near(x, yv[i]))
      }
      const l = landing(p, kind)
      return l != null && near(l[0], l[1])
    })
  }
  const region = useMemo(() => (spot && view === "heat" ? around(plays) : null), [spot, view, plays, heatOf, kind, radius])
  // compare mode: the two sides of the hovered spot - a split of these plays, or these vs the other player's / season's
  const compare = useMemo(() => {
    if (!region || heatMetric !== "compare") return undefined
    if (vsPlays && versus) return { a: versus.mine, b: versus.other, sides: [region, around(vsPlays)] as [Play[], Play[]] }
    if (isVersus(heatSplit)) return undefined
    const sp = splitOf(kind, heatSplit)
    return { a: sp.a, b: sp.b, sides: [region.filter((p) => sp.side(p) === "a"), region.filter((p) => sp.side(p) === "b")] as [Play[], Play[]] }
  }, [region, heatMetric, heatSplit, vsPlays, versus, kind])
  useEffect(() => {
    if (view !== "heat") setSpot(null)
  }, [view])
  useImperativeHandle(props.ref, () => ({
    svg: () => svgRef.current,
    heat: () => {
      if (heatImg) return heatImg
      const f = heatField(plays, kind, heatOf, heatMetric, heatSplit, XL, W, ymin, ymax, landing, vsPlays)
      return f ? { url: heatImage(f), x: XL, y: 0, w: W, h: ymax - ymin } : null
    },
  }), [heatImg, plays, kind, heatOf, heatMetric, heatSplit, vsPlays, ymin, ymax])

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
      onMouseMove={(e) => {
        if (dragging) return
        if (view === "heat") {
          const f = canHover ? fieldPoint(e.clientX, e.clientY) : null
          setSpot(f ? { x: f[0], y: f[1], cx: e.clientX, cy: e.clientY } : null)
          return
        }
        setHover((h) => (h ? { ...h, x: e.clientX, y: e.clientY } : h))
      }}
      onMouseLeave={() => canHover && (setHover(null), setSpot(null))}
      onClick={(e) => {
        if (canHover || view !== "heat" || justDragged.current) return
        const f = fieldPoint(e.clientX, e.clientY)
        setSpot(f ? { x: f[0], y: f[1], cx: e.clientX, cy: e.clientY } : null)
      }}
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

          {spot && region && !dragging && (
            <circle data-layer="spot" cx={spot.x} cy={Y(spot.y)} r={radius} fill="white" fillOpacity={0.07} stroke="white"
              strokeOpacity={0.85} strokeWidth={1.5} strokeDasharray="4 3" vectorEffect="non-scaling-stroke" pointerEvents="none" />
          )}

          {showYac && (
            <g data-layer="yac" pointerEvents="none">
              <defs>
                <marker id="yac-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                  <path d="M0,0 L10,5 L0,10 z" fill="context-stroke" />
                </marker>
              </defs>
              {plays.map((p) => {
                const end = yacEnd(p)
                if (end == null || p.x_coord == null || Math.abs(end - p.y_coord) < 0.5) return null
                return (
                  <line key={p.id} x1={p.x_coord} y1={Y(p.y_coord)} x2={p.x_coord} y2={Y(end)} stroke={playColor(p, kind, colorBy)}
                    strokeOpacity={focus != null && p.id !== focus ? 0.15 : 0.7} strokeWidth={1.4} vectorEffect="non-scaling-stroke"
                    markerEnd="url(#yac-head)" />
                )
              })}
            </g>
          )}

          <g key={animateKey} data-layer="plays" opacity={view === "heat" ? 0.18 : 1}
            style={view === "heat" ? { pointerEvents: "none" } : undefined}>
            {ordered.map((p, i) => (
              <PlayMark key={p.id} play={p} kind={kind} colorBy={colorBy} top={ymax} index={i}
                dim={focus != null && p.id !== focus} focused={p.id === focus} many={many}
                onEnter={(e) => setHover({ id: p.id, x: e.clientX, y: e.clientY })}
                onLeave={() => setHover((h) => (h?.id === p.id ? null : h))}
                onClick={() => onSelect(p.id === selectedId ? null : p.id)}
                animate={!threeD} noYac={kind === "route" && !routeYac}
              />
            ))}
          </g>
        </svg>
        {[[0, 0], [1, 0], [1, 1], [0, 1]].map(([u, v], i) => (
          <div key={i} ref={(el) => { corners.current[i] = el }} aria-hidden className="pointer-events-none absolute size-0"
            style={{ left: box.x + u * box.w, top: box.y + v * box.h }} />
        ))}
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
      {hovered && hover && canHover && !dragging && view !== "heat" && <PlayHoverCard play={hovered} kind={kind} x={hover.x} y={hover.y} />}
      {spot && region && !dragging && (
        <RegionCard kind={kind} plays={region} total={plays.length} radius={radius} paths={heatOf === "path" && kind !== "pass"}
          compare={compare} versus={!!vsPlays} touch={!canHover} x={spot.cx} y={spot.cy} />
      )}
    </div>
  )
}

function PlayMark({ play, kind, colorBy, top, index, dim, focused, many, onEnter, onLeave, onClick, animate, noYac = false }: {
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
  noYac?: boolean
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
  const post = firstAc > 0 && !noYac ? polyline(xs, yv, top, firstAc - 1, xs.length) : null
  // with the run after the catch hidden, the route ends at the catch - and so does its touchdown ring
  const last = firstAc > 0 && noYac ? firstAc - 1 : xs.length - 1
  const end = [xs[last], top - yv[last]]
  const td = play.chart_touchdown === true || play.chart_touchdown === 1
  const afterColor = colorBy === "outcome" ? ROUTE_LINE.after : color
  const line = (d: string, c: string, key: string, w = width) => (
    <path key={key} d={d} fill="none" stroke={c} strokeWidth={w} strokeLinecap="round" strokeLinejoin="round"
      pathLength={dash ? undefined : 1} strokeDasharray={dash} className={dash ? undefined : "draw-in"}
      style={{ animationDelay: delay }} />
  )
  return (
    <g {...handlers} opacity={opacity} filter={focused ? "url(#glow)" : undefined}>
      <path d={polyline(xs, yv, top, 0, last + 1)} fill="none" stroke="transparent" strokeWidth={12} vectorEffect="non-scaling-stroke" />
      {line(pre, color, "pre")}
      {post && line(post, afterColor, "post", width * 0.92)}
      {kind === "route" && firstAc > 0 && (
        <circle cx={xs[firstAc - 1]} cy={top - yv[firstAc - 1]} r={0.32} fill={noYac ? color : afterColor} />
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

/** The heatmap's hover card: the same numbers as the cards above the field, for the plays around the cursor. */
function RegionCard({ kind, plays, total, radius, paths, compare, versus, touch, x, y }: {
  kind: Kind; plays: Play[]; total: number; radius: number; paths: boolean
  compare?: { a: string; b: string; sides: [Play[], Play[]] }; versus: boolean; touch: boolean; x: number; y: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [h, setH] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (el && el.offsetHeight !== h) setH(el.offsetHeight)
  })
  const w = 300
  const gap = 18
  const pad = 8
  // beside the cursor; on a touch screen above or below the finger instead, so the tapped spot stays in view
  const left = touch ? Math.max(pad, Math.min(x - w / 2, window.innerWidth - w - pad))
    : x + gap + w + pad <= window.innerWidth ? x + gap : Math.max(pad, x - gap - w)
  const top = touch ? (y + 56 + h + pad <= window.innerHeight ? y + 56 : Math.max(pad, y - 56 - h))
    : Math.min(Math.max(pad, y - 48), Math.max(pad, window.innerHeight - h - pad))
  const [one, many] = kind === "pass" ? ["throw", "throws"] : kind === "route" ? ["target", "targets"] : ["carry", "carries"]
  const stats = plays.length && !compare ? kpis(kind, plays).slice(1) : []
  const sides = compare && (compare.sides[0].length || compare.sides[1].length) ? compare.sides : null
  const n = versus && compare ? compare.sides[0].length + compare.sides[1].length : plays.length
  return createPortal(
    <div ref={ref} style={{ left, top, width: w, visibility: h ? "visible" : "hidden" }}
      className="dark pointer-events-none fixed z-50 rounded-xl border border-white/10 bg-zinc-950/95 p-3.5 text-xs text-foreground shadow-2xl backdrop-blur-md">
      <div className="text-sm font-medium">
        {n ? `${n} ${n === 1 ? one : many}` : `No ${many} here`}
      </div>
      <div className="mb-2.5 text-[11px] text-muted-foreground">
        {paths ? `paths within ${radius} yds of here` : `ending within ${radius} yds of here`}
        {!versus && plays.length > 0 && ` · ${Math.round((100 * plays.length) / Math.max(1, total))}% of the ${total} on the field`}
      </div>
      {sides && compare && (
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-muted-foreground">
              <th />
              <th className="pb-1 text-right font-medium" style={{ color: COMPARE_A }}>{compare.a} · {sides[0].length}</th>
              <th className="pb-1 text-right font-medium" style={{ color: COMPARE_B }}>{compare.b} · {sides[1].length}</th>
            </tr>
          </thead>
          <tbody>
            {(sides[0].length ? kpis(kind, sides[0]) : kpis(kind, sides[1])).slice(1).map((k, i) => (
              <tr key={k.label} className="border-t border-white/5">
                <td className="py-1 text-muted-foreground">{k.label}</td>
                {sides.map((side, j) => (
                  <td key={j} className="py-1 text-right font-mono">{side.length ? kpis(kind, side)[i + 1].value : "—"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {stats.length > 0 && (
        <div className="grid grid-cols-2 gap-1.5">
          {stats.map((k) => (
            <div key={k.label} className="rounded-lg border border-white/10 px-2.5 py-1.5">
              <div className="text-[10px] tracking-wide text-muted-foreground uppercase">{k.label}</div>
              <div className={cn("font-mono text-sm font-medium", k.tone === "good" && "text-emerald-400", k.tone === "bad" && "text-rose-400")}>
                {k.value}
              </div>
              {k.sub && <div className="truncate text-[10px] text-muted-foreground">{k.sub}</div>}
            </div>
          ))}
        </div>
      )}
    </div>,
    document.body,
  )
}

/** Maps points through the projective transform taking the four `from` corners to the four `to` corners. */
function homography(from: [number, number][], to: [number, number][]): ((x: number, y: number) => [number, number]) | null {
  // solve for h0..h7 (h8 = 1): u = (h0 x + h1 y + h2) / (h6 x + h7 y + 1), v likewise with h3..h5
  const A: number[][] = []
  for (let i = 0; i < 4; i++) {
    const [x, y] = from[i]
    const [u, v] = to[i]
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v])
  }
  for (let c = 0; c < 8; c++) {
    let piv = c
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r
    if (Math.abs(A[piv][c]) < 1e-12) return null
    ;[A[c], A[piv]] = [A[piv], A[c]]
    for (let r = 0; r < 8; r++) {
      if (r === c) continue
      const f = A[r][c] / A[c][c]
      for (let k = c; k < 9; k++) A[r][k] -= f * A[c][k]
    }
  }
  const h = A.map((row, i) => row[8] / row[i])
  return (x, y) => {
    const d = h[6] * x + h[7] * y + 1
    return [(h[0] * x + h[1] * y + h[2]) / d, (h[3] * x + h[4] * y + h[5]) / d]
  }
}

