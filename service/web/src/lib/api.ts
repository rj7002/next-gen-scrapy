export type Kind = "pass" | "route" | "carry"

export const KIND_LABEL: Record<Kind, string> = { pass: "Passing", route: "Receiving", carry: "Rushing" }
export const KIND_NOUN: Record<Kind, [string, string]> = {
  pass: ["pass", "passes"],
  route: ["route", "routes"],
  carry: ["carry", "carries"],
}

export interface PlayerKindStats {
  games: number
  [stat: string]: number
}

export interface Player {
  esb_id: string
  name: string
  position?: string
  team: string
  headshot?: string | null
  kinds: Partial<Record<Kind, PlayerKindStats>>
}

export type JobStatus = "queued" | "running" | "ready" | "refining" | "done" | "error"

export interface Job {
  id: string
  kind: Kind
  name: string
  season: number
  status: JobStatus
  stage: string
  done: number
  total: number
  version: number
  error: string | null
}

// One pass / route / carry. Matched plays carry every curated pbp field; unmatched ones only the chart's own.
export interface Play {
  id: string
  matched: boolean
  week?: number | string | null
  [field: string]: any
}

export interface Game {
  week: string
  game_id: number
  opponent: string | null
  season_type: string
  [stat: string]: any
}

export interface Dataset {
  kind: Kind
  table: string
  season: number
  player: { esb_id: string; name: string; position?: string; team: string; headshot?: string | null }
  games: Game[]
  plays: Play[]
  notes: string[]
  pending?: { week: string; game_id: number; plays: number }[]   // charted games nflverse hasn't published yet
  counts: { charts: number; matched: number; unmatched: number; undrawn?: number }
  version: number
}

export type FilterKind = "choice" | "select" | "bool" | "range"

export interface FilterSpec {
  col: string
  label: string
  group: string
  kind: FilterKind
  tables: string[]
  choices: [string | number, string][] | null
  since: string | null
  step: number | null
  help: string | null
}

export interface AskResult {
  dataset: string | null
  seasons: number[]
  weeks: number[]
  team: string | null
  name: string | null
  esb_id?: string | null
  params: Record<string, string>
  understood: { text: string; meaning: string }[]
  ignored: string[]
  notes: string[]
  engine: string
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = r.statusText
    try {
      msg = (await r.json()).detail ?? msg
    } catch {
      /* not json */
    }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

/** fetch for /api: the deployed server answers only this page's session; if that has expired (a tab left open,
 * the server restarted), get a new one and try once more. */
async function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  const r = await fetch(url, { credentials: "same-origin", ...init })
  if (r.status !== 401) return r
  await fetch("/session", { credentials: "same-origin" })
  return fetch(url, { credentials: "same-origin", ...init })
}

export const api = {
  seasons: () => apiFetch("/api/seasons").then(json<{ seasons: number[]; current: number }>),
  players: (season: number) => apiFetch(`/api/players?season=${season}`).then(json<{ players: Player[] }>),
  load: (kind: Kind, esb_id: string, season: number) =>
    apiFetch("/api/load", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind, esb_id, season }),
    }).then(json<Job>),
  job: (id: string) => apiFetch(`/api/jobs/${id}`).then(json<Job>),
  data: (id: string) => apiFetch(`/api/data/${id}`).then(json<Dataset>),
  filters: (kind: Kind) => apiFetch(`/api/filters?kind=${kind}`).then(json<{ groups: string[]; filters: FilterSpec[] }>),
  ask: (q: string, kind: Kind, season: number, job?: string) =>
    apiFetch(`/api/ask?${new URLSearchParams({ q, kind, season: String(season), ...(job ? { job } : {}) })}`).then(
      json<AskResult>
    ),
  play: (job: string, id: string) => apiFetch(`/api/play/${job}/${encodeURIComponent(id)}`).then(json<PlayDetailData>),
  exportUrl: (id: string) => `/api/export/${id}`,
}

export interface Person { id: string; role: string; name?: string | null; position?: string | null; headshot?: string | null; jersey?: number | null; team?: string | null }
export interface DrivePlay { play_id: number; qtr?: number; time?: string; down?: number; ydstogo?: number; yrdln?: string; play_type?: string; yards?: number; epa?: number; desc?: string; current: boolean }
/** Everything about one play (GET /api/play/{job}/{id}); sections are absent when unknown. */
export interface PlayDetailData {
  id: string
  kind: Kind
  matched: boolean
  week?: string
  chart: Record<string, any>
  game?: Record<string, any>
  situation?: Record<string, any>
  result?: Record<string, any>
  scheme?: Record<string, any>
  charting?: Record<string, any>
  gamecenter?: Record<string, any>
  match?: Record<string, any>
  people?: Person[]
  drive?: { number?: number; result?: string; plays_count?: number; time?: string; first_downs?: number; start?: string; end?: string; plays: DrivePlay[] }
}

export function kindFromDataset(ds: string | null | undefined): Kind | null {
  return ds === "passes" ? "pass" : ds === "routes" ? "route" : ds === "carries" ? "carry" : null
}

/** "1".."18" for the regular season; NGS labels playoff games post-19 (wild card) .. post-23 (Super Bowl). */
export function weekLabel(w: string | number | null | undefined, long = false): string {
  const s = String(w ?? "")
  if (!s.startsWith("post")) return long ? `Week ${s}` : s
  const n = Number(s.split("-")[1])
  const name = n <= 19 ? "WC" : n === 20 ? "DIV" : n === 21 ? "CONF" : "SB"
  return long ? { WC: "Wild Card", DIV: "Divisional", CONF: "Conference", SB: "Super Bowl" }[name]! : name
}

/** Why a drawn line has no play: never matched, or matched to nothing that fits (likely a misread line). */
export function unmatchedText(play: Play) {
  if (play.reason !== "no_fit") return "Drawn on the chart, but no play-by-play row could be matched to it."
  const end = Array.isArray(play.path_y) && play.path_y.length ? play.path_y[play.path_y.length - 1] : null
  return `No run in play-by-play fits this line${end != null ? ` (it ends ${Number(end).toFixed(1)} yds past the line of scrimmage)` : ""} - most likely a misread chart: a short run hidden under a longer one, read as a branch of it.`
}
