import { useEffect, useState } from 'react'

export type ThemeMode = 'light' | 'dark'
export type ThemePref = ThemeMode | 'system'
export type RGBA = [number, number, number, number]

/** Canvas (deck.gl) colours per mode. Mirrors the CSS roles in index.css. */
export const CANVAS: Record<
  ThemeMode,
  {
    surface: RGBA
    point: RGBA
    pointDim: RGBA
    pointProbed: RGBA
    centroid: RGBA
    ink: RGBA
    series1: RGBA
    series2: RGBA
    /** Sequential blue ramp (light -> dark) for magnitude. */
    ramp: string[]
  }
> = {
  light: {
    surface: hex('#fcfcfb'),
    point: hex('#898781', 150),
    pointDim: hex('#c3c2b7', 110),
    pointProbed: hex('#52514e', 200),
    centroid: hex('#0b0b0b', 230),
    ink: hex('#0b0b0b'),
    series1: hex('#2a78d6'),
    series2: hex('#eb6834'),
    ramp: ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
  },
  dark: {
    surface: hex('#1a1a19'),
    point: hex('#898781', 150),
    pointDim: hex('#383835', 160),
    pointProbed: hex('#c3c2b7', 200),
    centroid: hex('#ffffff', 230),
    ink: hex('#ffffff'),
    series1: hex('#3987e5'),
    series2: hex('#d95926'),
    // Dark mode: the ramp still runs "less -> more", from near-surface to bright.
    ramp: ['#104281', '#184f95', '#1c5cab', '#2a78d6', '#5598e7', '#86b6ef', '#b7d3f6'],
  },
}

export function hex(h: string, alpha = 255): RGBA {
  const n = parseInt(h.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255, alpha]
}

/** Map t in [0, 1] onto the sequential ramp. */
export function rampColor(mode: ThemeMode, t: number, alpha = 220): RGBA {
  const ramp = CANVAS[mode].ramp
  const i = Math.min(ramp.length - 1, Math.max(0, Math.round(t * (ramp.length - 1))))
  return hex(ramp[i], alpha)
}

const STORAGE_KEY = 'faissight-theme'

function readPref(): ThemePref {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch {
    // storage unavailable (private mode, sandboxed iframe): fall back to the OS
  }
  return 'system'
}

function systemMode(): ThemeMode {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

/** Theme preference with OS fallback; keeps <html data-theme> in sync. */
export function useTheme(): { mode: ThemeMode; pref: ThemePref; setPref: (p: ThemePref) => void } {
  const [pref, setPrefState] = useState<ThemePref>(readPref)
  const [system, setSystem] = useState<ThemeMode>(systemMode)

  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    if (!mq) return
    const onChange = () => setSystem(mq.matches ? 'dark' : 'light')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  const mode = pref === 'system' ? system : pref
  useEffect(() => {
    document.documentElement.dataset.theme = mode
  }, [mode])

  const setPref = (p: ThemePref) => {
    setPrefState(p)
    try {
      localStorage.setItem(STORAGE_KEY, p)
    } catch {
      // ignore: the choice just won't persist
    }
  }
  return { mode, pref, setPref }
}
