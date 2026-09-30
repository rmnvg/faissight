import { useEffect, useMemo, useState } from 'react'
import type { HnswTrace } from '../../api/types'
import { buildFrames, stateAt, type FrameState } from '../../lib/hnswFrames'

/** Step-through state for a trace: the current frame, play/pause and speed. A new trace
 * starts from the top and plays. */
export function usePlayback(trace: HnswTrace | null) {
  const frames = useMemo(() => (trace ? buildFrames(trace) : []), [trace])
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2)
  const [traceSeen, setTraceSeen] = useState<HnswTrace | null>(null)
  if (trace !== traceSeen) {
    // New trace: start from the top and play.
    setTraceSeen(trace)
    setFrame(0)
    setPlaying(trace !== null)
  }
  useEffect(() => {
    if (!playing || frames.length === 0) return
    const id = window.setInterval(() => {
      setFrame((f) => {
        if (f + 1 >= frames.length) {
          setPlaying(false)
          return f
        }
        return f + 1
      })
    }, 600 / speed)
    return () => window.clearInterval(id)
  }, [playing, speed, frames.length])

  const state: FrameState | null = trace && frames.length ? stateAt(trace, frames, frame) : null
  const finished = state?.isLast && !playing
  return { frames, frame, setFrame, playing, setPlaying, speed, setSpeed, state, finished }
}

export type Playback = ReturnType<typeof usePlayback>
export type Camera = 'auto' | 'layers' | 'level0'
