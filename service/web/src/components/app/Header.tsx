import { Moon, Search, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

/** The NFL Field Explorer mark (same drawing as public/favicon.svg): a route off the line of scrimmage into a catch ring. */
export function Logo({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <defs>
        <linearGradient id="ngs-logo-tile" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#111a2e" />
          <stop offset="1" stopColor="#0a0f1c" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="8" fill="url(#ngs-logo-tile)" />
      <rect x="1.5" y="1.5" width="29" height="29" rx="7.5" fill="none" stroke="#ffffff" strokeOpacity=".12" />
      <path d="M5 9.5H27M5 15.5H27" stroke="#ffffff" strokeOpacity=".10" strokeWidth="1" />
      <path d="M5 22H27" stroke="#3b82f6" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M10 26.5V16.5Q10 14.5 12 13.6L19.2 10.4" fill="none" stroke="#f8fafc" strokeWidth="2.4"
        strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="22.4" cy="9" r="3.2" fill="none" stroke="#34d399" strokeWidth="2" />
      <circle cx="22.4" cy="9" r="1" fill="#34d399" />
    </svg>
  )
}

export function Header({ seasons, season, onSeason, onSearch, dark, onTheme }: {
  seasons: number[]
  season: number
  onSeason: (s: number) => void
  onSearch: () => void
  dark: boolean
  onTheme: () => void
}) {
  return (
    <header className="sticky top-0 z-40 border-b border-border/60 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-[1680px] items-center gap-3 px-4 lg:px-6">
        <a href="/" className="flex items-center gap-2.5">
          <Logo className="size-7" />
          <span className="text-[15px] font-semibold tracking-tight">
            NFL Field <span className="text-muted-foreground">Explorer</span>
          </span>
        </a>
        <div className="ml-auto flex items-center gap-1.5 sm:gap-2">
          <Button variant="outline" onClick={onSearch}
            className="h-9 w-9 justify-center gap-2 bg-muted/40 px-0 text-muted-foreground shadow-none sm:w-72 sm:justify-start sm:px-3"
            aria-label="Search players">
            <Search className="size-4" />
            <span className="hidden flex-1 text-left sm:inline">Search players…</span>
            <KbdGroup className="hidden sm:inline-flex">
              <Kbd>⌘</Kbd>
              <Kbd>K</Kbd>
            </KbdGroup>
          </Button>
          <Select value={String(season)} onValueChange={(v) => onSeason(Number(v))}>
            <SelectTrigger className="h-9 w-[92px] bg-muted/40 font-mono sm:w-[104px]" aria-label="Season">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {seasons.map((s) => (
                <SelectItem key={s} value={String(s)} className="font-mono">
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="ghost" size="icon" onClick={onTheme} aria-label="Toggle theme">
            {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
        </div>
      </div>
    </header>
  )
}
