// A small LLM that runs in the visitor's own browser (WebLLM over WebGPU) to read the words of an Ask-bar question
// the rule parser couldn't place: no API key, no server, and the question never leaves the device. The model
// (~280 MB) downloads once and the browser keeps it; until it's ready, or where WebGPU isn't available, the rules'
// answer stands on its own.
import type { MLCEngineInterface } from "@mlc-ai/web-llm"

export const MODELS = {
  small: { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 1.5B", size: "~0.9 GB" },
  tiny: { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 0.5B", size: "~0.3 GB" },
  mid: { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 3B", size: "~1.7 GB" },
  big: { id: "Qwen2.5-7B-Instruct-q4f16_1-MLC", label: "Qwen 2.5 7B", size: "~4.3 GB" },
  ft: { id: "ngs-ask-0.5b-q4f16_1", label: "NGS Ask 0.5B (fine-tuned)", size: "~280 MB" },
} as const
export type ModelKey = keyof typeof MODELS

export type ModelStatus = { state: "unsupported" | "idle" | "loading" | "ready" | "error"; progress: number; text: string }

export function webgpuSupported(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator
}

let status: ModelStatus = { state: webgpuSupported() ? "idle" : "unsupported", progress: 0, text: "" }
const listeners = new Set<(s: ModelStatus) => void>()
let engine: MLCEngineInterface | null = null
let started: Promise<MLCEngineInterface | null> | null = null

function set(s: Partial<ModelStatus>) {
  status = { ...status, ...s }
  for (const f of listeners) f(status)
}

// read at load, before the app tidies the address bar
const QS = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "")

export function modelKey(): ModelKey {
  const q = QS.get("nlm")
  return q === "small" || q === "tiny" || q === "mid" || q === "big" ? q : "ft"
}

// Qwen 2.5 0.5B fine-tuned on the Ask bar's own task: same architecture as WebLLM's prebuilt 0.5B, so it runs on
// that model's compiled library and only the weights (~270 MB) are ours
const FT_URL = QS.get("nlurl") ?? "https://huggingface.co/rj7002/ngs-ask-0.5b-MLC"
const FT_LIB = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm"

export function modelStatus(): ModelStatus {
  return status
}

export function onModelStatus(f: (s: ModelStatus) => void): () => void {
  listeners.add(f)
  return () => {
    listeners.delete(f)
  }
}

/** Start downloading / loading the model in the background (once); resolves with the engine, or null. */
export function preload(): Promise<MLCEngineInterface | null> {
  if (started) return started
  if (!webgpuSupported()) return Promise.resolve(null)
  set({ state: "loading", progress: 0, text: "" })
  started = (async () => {
    try {
      const web = await import("@mlc-ai/web-llm")          // the runtime itself is only fetched here
      const appConfig = modelKey() !== "ft" ? undefined : { model_list: [...web.prebuiltAppConfig.model_list,
        { model: FT_URL, model_id: MODELS.ft.id, model_lib: FT_LIB, low_resource_required: true,
          overrides: { context_window_size: 4096 } }] }
      engine = await web.CreateWebWorkerMLCEngine(
        new Worker(new URL("./llm.worker.ts", import.meta.url), { type: "module" }),
        MODELS[modelKey()].id,
        { initProgressCallback: (r) => set({ progress: r.progress, text: r.text }), appConfig },
        { context_window_size: 4096 },
      )
      set({ state: "ready", progress: 1 })
      return engine
    } catch (e) {
      set({ state: "error", text: e instanceof Error ? e.message : String(e) })
      return null
    }
  })()
  return started
}

type Msg = { role: string; content: string }

/** The model's answer to the server's prompt, held to the server's JSON schema - or null if the model isn't ready
 * (the caller keeps the rules' answer). The fine-tuned model gets the compact prompt it was trained on and writes
 * its JSON unconstrained (the server validates every field either way). */
export async function answerJSON(prompt: string, schema: object, messages?: Msg[] | null, ft?: Msg[] | null):
    Promise<Record<string, unknown> | null> {
  if (status.state !== "ready" || !engine) return null
  const tuned = modelKey() === "ft" && !!ft
  const free = tuned || QS.get("nlschema") === "0"
  const useMsgs = messages && QS.get("nlmsgs") !== "0"
  const reply = await engine.chat.completions.create({
    messages: (tuned ? ft : useMsgs ? messages : [{ role: "user", content: prompt }]) as any,
    temperature: 0,
    max_tokens: 400,
    ...(free ? {} : { response_format: { type: "json_object" as const, schema: JSON.stringify(schema) } }),
  })
  const text = reply.choices[0]?.message?.content ?? ""
  try {
    return JSON.parse(free ? (text.match(/\{[\s\S]*\}/)?.[0] ?? "") : text)
  } catch {
    return null
  }
}

// test hook for the Ask-bar accuracy check (only with ?nltest=1 in the URL)
if (typeof window !== "undefined" && QS.has("nltest")) {
  ;(window as any).__nl = { preload, answerJSON, modelStatus, modelKey,
    schemaOn: () => QS.get("nlschema") !== "0",
    raw: async (prompt: string) => (await engine!.chat.completions.create({ messages: [{ role: "user", content: prompt }], temperature: 0, max_tokens: 300 })).choices[0]?.message?.content }
}
