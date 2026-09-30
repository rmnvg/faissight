import type { CompareMeasurement, UserId } from '../api/types'

/** "12.4 MB" style size, binary units (serialized index bytes). */
export function fmtBytes(n: number): string {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${i === 0 ? v : v.toFixed(v < 10 ? 2 : 1)} ${units[i]}`
}

/** "2.3× faster" / "1.4× larger" wording for right vs left; "same" within `tolerance`. */
export function relative(
  left: number,
  right: number,
  words: [smaller: string, larger: string],
  tolerance = 0.05,
): { text: string; better: boolean | null } {
  if (left <= 0 || right <= 0) return { text: '—', better: null }
  const r = right / left
  if (Math.abs(r - 1) <= tolerance) return { text: 'about the same', better: null }
  return r < 1
    ? { text: `${(1 / r).toFixed(1)}× ${words[0]}`, better: true }
    : { text: `${r.toFixed(1)}× ${words[1]}`, better: false }
}

/** True when the two recall intervals overlap, so the recall gap may be sampling noise. */
export function recallIntervalsOverlap(a: CompareMeasurement, b: CompareMeasurement): boolean {
  if (a.recall_ci_low === null || a.recall_ci_high === null) return false
  if (b.recall_ci_low === null || b.recall_ci_high === null) return false
  return a.recall_ci_low <= b.recall_ci_high && b.recall_ci_low <= a.recall_ci_high
}

/** Signed recall change with 3 decimals; changes that round to zero show as "0.000". */
export function signedRecall(d: number): string {
  if (Math.abs(d) < 0.0005) return '0.000'
  return `${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(3)}`
}

/** One-sentence verdict: what switching from the left index to the right one does. */
export function verdict(left: CompareMeasurement, right: CompareMeasurement, k: number): string {
  const d = right.recall - left.recall
  const recall = Math.abs(d) < 0.0005 ? `the same recall@${k}` : `${signedRecall(d)} recall@${k}`
  const speed = relative(left.latency_p95_ms, right.latency_p95_ms, ['faster', 'slower'])
  const size = relative(left.serialized_bytes, right.serialized_bytes, ['smaller', 'larger'])
  const parts = [
    recall,
    speed.text === 'about the same' ? 'about the same p95 latency' : `${speed.text} at p95`,
    size.text === 'about the same' ? 'about the same size' : size.text,
  ]
  return `${right.name} vs ${left.name}: ${parts.join(', ')}.`
}

/** First few ids, then "+N more". */
export function idList(ids: UserId[], max = 5): string {
  if (ids.length === 0) return '—'
  const shown = ids.slice(0, max).join(', ')
  return ids.length > max ? `${shown} +${ids.length - max} more` : shown
}
