import { useEffect, useState } from "react"
import { ArrowUp, Loader2, Sparkles } from "lucide-react"
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group"

const HINTS: Record<string, string[]> = {
  pass: ["3rd down throws under pressure", "deep passes vs zone in the 4th quarter", "play action in the red zone"],
  route: ["targets on 3rd and long", "deep routes vs man coverage", "red zone targets"],
  carry: ["runs to the right on 1st down", "carries in the red zone", "shotgun runs with 8 in the box"],
}

export function AskBar({ kind, busy, onAsk }: { kind: string; busy: boolean; onAsk: (q: string) => void }) {
  const [q, setQ] = useState("")
  const [hint, setHint] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setHint((h) => h + 1), 3500)
    return () => clearInterval(t)
  }, [])
  const hints = HINTS[kind] ?? HINTS.pass
  const submit = () => q.trim() && onAsk(q.trim())
  return (
    <InputGroup className="h-11 rounded-xl border-border/70 bg-card/60 shadow-sm">
      <InputGroupAddon>
        <Sparkles className="size-4 text-emerald-700 dark:text-emerald-300" />
      </InputGroupAddon>
      <InputGroupInput value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()}
        placeholder={`Ask: “${hints[hint % hints.length]}”`} className="text-sm" />
      <InputGroupAddon align="inline-end">
        <InputGroupButton size="icon-xs" variant="default" className="rounded-lg" onClick={submit} disabled={busy || !q.trim()} aria-label="Ask">
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <ArrowUp className="size-3.5" />}
        </InputGroupButton>
      </InputGroupAddon>
    </InputGroup>
  )
}
