import { useSyncExternalStore } from "react"

const q = typeof window !== "undefined" ? window.matchMedia("(prefers-reduced-motion: reduce)") : null

/** true when the user (or the screenshot tooling) asks for no animation */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    (cb) => {
      q?.addEventListener("change", cb)
      return () => q?.removeEventListener("change", cb)
    },
    () => q?.matches ?? false,
    () => false
  )
}
