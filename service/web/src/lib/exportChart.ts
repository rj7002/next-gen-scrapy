// Export the field as a standalone, shareable image: a dark card with a title, the 2D field (whatever 3D tilt is on
// screen), a legend and a small wordmark - as PNG (2x), SVG, or straight to the clipboard.

const NS = "http://www.w3.org/2000/svg"
const SANS = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

export type LegendSpec =
  | { items: [string, string][]; dashed?: boolean }
  | { gradient: [string, string, string]; lo: string; hi: string }

export interface HeatImage { url: string; x: number; y: number; w: number; h: number }

export interface ExportOptions {
  view: "plays" | "heat"
  heat?: HeatImage | null
  title: string
  subtitle: string
  legend: LegendSpec
  filename: string
  format: "png" | "svg" | "clipboard"
}

function el(name: string, attrs: Record<string, string | number>, text?: string) {
  const e = document.createElementNS(NS, name)
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v))
  if (text != null) e.textContent = text
  return e
}

/** The card as SVG markup, plus its pixel size. */
export function buildChartSvg(src: SVGSVGElement, o: ExportOptions): { markup: string; width: number; height: number } {
  const vb = src.viewBox.baseVal
  const field = src.cloneNode(true) as SVGSVGElement
  field.removeAttribute("class")
  field.removeAttribute("style")
  // the two layers: plays on their own, or the heatmap over faint plays
  const plays = field.querySelector('[data-layer="plays"]')
  let heat = field.querySelector('[data-layer="heat"]')
  if (o.view === "plays") {
    heat?.remove()
    plays?.setAttribute("opacity", "1")
  } else {
    if (!heat && o.heat) {
      heat = el("image", { href: o.heat.url, x: o.heat.x, y: o.heat.y, width: o.heat.w, height: o.heat.h, preserveAspectRatio: "none",
        style: "mix-blend-mode: screen", "data-layer": "heat" })
      plays?.parentNode?.insertBefore(heat, plays)
    }
    plays?.setAttribute("opacity", "0.18")
  }
  // nothing interactive or half-animated in a still image
  field.querySelectorAll("[class]").forEach((n) => n.removeAttribute("class"))
  field.querySelectorAll("[style]").forEach((n) => {
    if (n.getAttribute("data-layer") !== "heat") n.removeAttribute("style")
  })

  const W = 1600
  const pad = 56
  const headH = 150
  const fw = W - 2 * pad
  const fh = Math.round((fw * vb.height) / vb.width)
  // lay the legend out first (it may wrap) so the card can be sized to fit
  const rows: { label: string; color: string; x: number; row: number }[] = []
  if ("items" in o.legend) {
    let x = pad
    let row = 0
    const entries: [string, string][] = [...o.legend.items, ...(o.legend.dashed ? [["unmatched", ""] as [string, string]] : [])]
    for (const [label, color] of entries) {
      const w = 24 + label.length * 10.5 + 34
      if (x + w > W - pad && x > pad) {
        x = pad
        row += 1
      }
      rows.push({ label, color, x, row })
      x += w
    }
  }
  const legendRows = rows.length ? Math.max(...rows.map((r) => r.row)) + 1 : 1
  const footH = 60 + 34 * legendRows
  const H = headH + fh + footH
  const out = el("svg", { xmlns: NS, width: W, height: H, viewBox: `0 0 ${W} ${H}`, "font-family": SANS })
  out.append(el("rect", { x: 0, y: 0, width: W, height: H, fill: "#0b0c0f" }))
  out.append(el("rect", { x: 0, y: 0, width: W, height: 4, fill: "#10b981" }))
  out.append(el("text", { x: pad, y: 78, fill: "#fafafa", "font-size": 44, "font-weight": 600, "letter-spacing": -0.5 }, o.title))
  out.append(el("text", { x: pad, y: 118, fill: "#a1a1aa", "font-size": 21 }, o.subtitle))

  field.setAttribute("x", String(pad))
  field.setAttribute("y", String(headH))
  field.setAttribute("width", String(fw))
  field.setAttribute("height", String(fh))
  field.setAttribute("font-family", MONO)
  out.append(el("rect", { x: pad - 1, y: headH - 1, width: fw + 2, height: fh + 2, rx: 14, fill: "#101215", stroke: "#27272a" }))
  out.append(field)

  // legend
  const ly = headH + fh + 54
  const lg = el("g", { "font-size": 19, fill: "#d4d4d8" })
  if ("items" in o.legend) {
    for (const { label, color, x, row } of rows) {
      const y = ly + row * 34
      if (color) lg.append(el("circle", { cx: x + 8, cy: y - 6, r: 8, fill: color }))
      else lg.append(el("line", { x1: x - 4, x2: x + 18, y1: y - 6, y2: y - 6, stroke: "#a1a1aa", "stroke-width": 3, "stroke-dasharray": "6 5" }))
      lg.append(el("text", { x: x + 24, y }, label))
    }
  } else {
    const [a, b, c] = o.legend.gradient
    const defs = el("defs", {})
    const grad = el("linearGradient", { id: "export-legend", x1: 0, x2: 1, y1: 0, y2: 0 })
    grad.append(el("stop", { offset: 0, "stop-color": a }), el("stop", { offset: 0.5, "stop-color": b }), el("stop", { offset: 1, "stop-color": c }))
    defs.append(grad)
    out.append(defs)
    lg.append(el("text", { x: pad, y: ly }, o.legend.lo))
    const gx = pad + o.legend.lo.length * 10.5 + 18
    lg.append(el("rect", { x: gx, y: ly - 14, width: 260, height: 14, rx: 7, fill: "url(#export-legend)" }))
    lg.append(el("text", { x: gx + 278, y: ly }, o.legend.hi))
  }
  out.append(lg)
  // wordmark, top right beside the title: "NFL Field" in green, "Explorer" muted
  out.append(el("text", { x: W - pad, y: 76, "text-anchor": "end", "font-size": 22, fill: "#71717a" }, "Explorer"))
  out.append(el("text", { x: W - pad - 98, y: 76, "text-anchor": "end", "font-size": 22, "font-weight": 600, fill: "#10b981" }, "NFL Field"))
  return { markup: new XMLSerializer().serializeToString(out), width: W, height: H }
}

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.append(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

async function rasterize(markup: string, width: number, height: number, scale = 2): Promise<Blob> {
  const url = URL.createObjectURL(new Blob([markup], { type: "image/svg+xml;charset=utf-8" }))
  try {
    const img = new Image()
    img.decoding = "async"
    await new Promise<void>((ok, fail) => {
      img.onload = () => ok()
      img.onerror = () => fail(new Error("couldn't render the chart"))
      img.src = url
    })
    const canvas = document.createElement("canvas")
    canvas.width = width * scale
    canvas.height = height * scale
    const ctx = canvas.getContext("2d")!
    ctx.scale(scale, scale)
    ctx.drawImage(img, 0, 0, width, height)
    return await new Promise<Blob>((ok, fail) => canvas.toBlob((b) => (b ? ok(b) : fail(new Error("couldn't encode PNG"))), "image/png"))
  } finally {
    URL.revokeObjectURL(url)
  }
}

export async function exportChart(src: SVGSVGElement, o: ExportOptions) {
  const { markup, width, height } = buildChartSvg(src, o)
  if (o.format === "svg") return download(new Blob([markup], { type: "image/svg+xml" }), `${o.filename}.svg`)
  const png = await rasterize(markup, width, height)
  if (o.format === "png") return download(png, `${o.filename}.png`)
  await navigator.clipboard.write([new ClipboardItem({ "image/png": png })])
}
