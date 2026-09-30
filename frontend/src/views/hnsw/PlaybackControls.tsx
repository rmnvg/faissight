import { Segmented } from '../../components/ui'
import { frameLabel } from '../../lib/hnswFrames'
import { SPEEDS } from './constants'
import type { Camera, Playback } from './usePlayback'

/** Play/pause, step, scrub, speed and camera controls over the scene. */
export function PlaybackControls({
  playback,
  camera,
  onCamera,
}: {
  playback: Playback
  camera: Camera
  onCamera: (c: Camera) => void
}) {
  const { frames, frame, setFrame, playing, setPlaying, speed, setSpeed, state } = playback
  if (!state) return null
  return (
    <div className="absolute right-3 bottom-3 left-3 flex flex-wrap items-center gap-3 rounded-xl border border-line bg-surface/95 px-3 py-2 text-xs text-ink-2 shadow-sm">
      <button
        onClick={() => {
          if (state.isLast) setFrame(0)
          setPlaying(!playing)
        }}
        className="w-16 rounded-md bg-series-1 px-2 py-1 font-medium text-white"
        aria-label={playing ? 'Pause' : 'Play'}
      >
        {playing ? 'Pause' : state.isLast ? 'Replay' : 'Play'}
      </button>
      <button onClick={() => { setPlaying(false); setFrame(Math.max(0, frame - 1)) }} className="rounded-md border border-line px-2 py-1 hover:text-ink" aria-label="Previous step">◀</button>
      <button onClick={() => { setPlaying(false); setFrame(Math.min(frames.length - 1, frame + 1)) }} className="rounded-md border border-line px-2 py-1 hover:text-ink" aria-label="Next step">▶</button>
      <input
        type="range"
        min={0}
        max={frames.length - 1}
        value={frame}
        onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)) }}
        className="min-w-40 flex-1 accent-series-1"
        aria-label="Search step"
      />
      <span className="tabular w-44 text-ink">{frameLabel(state.frame)}</span>
      <Segmented
        label="Speed"
        value={speed}
        onChange={setSpeed}
        options={SPEEDS.map((s) => ({ value: s, label: `${s}×` }))}
      />
      <Segmented
        label="Camera"
        value={camera}
        onChange={onCamera}
        options={[
          { value: 'auto', label: 'Auto', title: 'Layers, then close in on level 0' },
          { value: 'layers', label: 'Layers' },
          { value: 'level0', label: 'Level 0' },
        ]}
      />
    </div>
  )
}
