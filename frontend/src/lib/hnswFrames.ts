import type { UserId } from '../api/types'
import type { HnswTrace, HnswVisit } from '../api/types'

/** One animation frame: scanning the links of one node. */
export interface Frame {
  level: number
  /** 0-based index of this step within its level. */
  levelStep: number
  levelSteps: number
  expanded: UserId
  visits: HnswVisit[]
}

/** Frames in execution order: top level first, level 0 last. */
export function buildFrames(trace: HnswTrace): Frame[] {
  const frames: Frame[] = []
  for (const lv of trace.levels) {
    lv.steps.forEach((s, i) =>
      frames.push({
        level: lv.level,
        levelStep: i,
        levelSteps: lv.steps.length,
        expanded: s.expanded,
        visits: s.visits,
      }),
    )
  }
  return frames
}

export interface FrameState {
  frame: Frame
  /** Nodes evaluated on the current level so far (including the level's entry node). */
  visited: Set<UserId>
  /** Nodes whose links were scanned on the current level so far. */
  expanded: Set<UserId>
  /** Level 0: queued candidates not yet expanded. Upper levels: the current nearest node. */
  frontier: Set<UserId>
  /** Where the search entered each level up to now (the descent path). */
  entries: { level: number; node: UserId }[]
  isLast: boolean
}

/** Search state after playing frames[0..index]. */
export function stateAt(trace: HnswTrace, frames: Frame[], index: number): FrameState {
  const i = Math.max(0, Math.min(index, frames.length - 1))
  const frame = frames[i]
  const entry = trace.levels.find((l) => l.level === frame.level)?.entry ?? frame.expanded
  const visited = new Set<UserId>([entry])
  const expanded = new Set<UserId>()
  const queued = new Set<UserId>([entry])
  let nearest = entry
  for (let j = 0; j <= i; j++) {
    const f = frames[j]
    if (f.level !== frame.level) continue
    expanded.add(f.expanded)
    queued.delete(f.expanded)
    for (const v of f.visits) {
      visited.add(v.node)
      if (v.accepted) {
        queued.add(v.node)
        nearest = v.node
      }
    }
  }
  const frontier = frame.level === 0 ? queued : new Set([nearest])
  const entries = trace.levels
    .filter((l) => l.level >= frame.level)
    .map((l) => ({ level: l.level, node: l.entry }))
  return { frame, visited, expanded, frontier, entries, isLast: i === frames.length - 1 }
}

/** Label like "Level 0 · step 12 of 40". */
export function frameLabel(f: Frame): string {
  return `Level ${f.level} · step ${f.levelStep + 1} of ${f.levelSteps}`
}
