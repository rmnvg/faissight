import type { Suggestion, SweepParam, SweepPoint } from '../api/types'
import type { View } from './route'

/** Whether a point meets the target: its mean recall, or with `confident` the lower end of
 * its 95% interval (mirrors core.sweep). */
export function meets(p: SweepPoint, target: number, confident = false): boolean {
  const recall = confident && p.recall_ci_low !== null ? p.recall_ci_low : p.recall
  return recall >= target - 1e-9
}

/** Smallest parameter value that meets the target (mirrors core.sweep). */
export function recommend(points: SweepPoint[], target: number, confident = false): SweepPoint | null {
  const ok = points.filter((p) => meets(p, target, confident))
  if (ok.length === 0) return null
  return ok.reduce((a, b) => (b.value < a.value ? b : a))
}

/** Measured-fastest point (mean latency) that meets the target (mirrors core.sweep). */
export function fastest(points: SweepPoint[], target: number, confident = false): SweepPoint | null {
  const ok = points.filter((p) => meets(p, target, confident))
  if (ok.length === 0) return null
  return ok.reduce((a, b) =>
    b.latency_mean_ms < a.latency_mean_ms || (b.latency_mean_ms === a.latency_mean_ms && b.value < a.value)
      ? b
      : a,
  )
}

/** Share of queries whose own recall is below the target; null without a distribution. */
export function fractionBelow(p: SweepPoint, target: number): number | null {
  const total = p.recall_distribution.reduce((s, [, n]) => s + n, 0)
  if (total === 0) return null
  const below = p.recall_distribution.reduce((s, [r, n]) => (r < target - 1e-9 ? s + n : s), 0)
  return below / total
}

/** "3.2x faster than the largest value tried", or null when not meaningful. */
export function speedup(points: SweepPoint[], chosen: SweepPoint): number | null {
  const slowest = points.reduce((a, b) => (b.value > a.value ? b : a))
  if (slowest.value === chosen.value || chosen.latency_mean_ms <= 0) return null
  return slowest.latency_mean_ms / chosen.latency_mean_ms
}

/** Parse "1, 2, 4 8" into sorted unique positive integers; null if anything is invalid. */
export function parseValues(text: string, max: number | null): number[] | null {
  const parts = text.split(/[\s,]+/).filter(Boolean)
  if (parts.length === 0) return null
  const nums = parts.map(Number)
  if (nums.some((n) => !Number.isInteger(n) || n < 1 || (max !== null && n > max))) return null
  return [...new Set(nums)].sort((a, b) => a - b)
}

/** Copy-pasteable Python for applying the chosen value. */
export function codeSnippet(param: SweepParam, value: number, hasRefine: boolean): string {
  const cls = param === 'nprobe' ? 'SearchParametersIVF' : 'SearchParametersHNSW'
  let perCall = `faiss.${cls}(${param}=${value})`
  if (hasRefine) {
    perCall = `faiss.IndexRefineSearchParameters(\n    k_factor=index.k_factor,\n    base_index_params=${perCall},\n)`
  }
  return [
    '# Per search call (leaves the index unchanged):',
    `params = ${perCall}`,
    'D, I = index.search(xq, k, params=params)',
    '',
    '# Or set it on the index (works through IDMap / PreTransform wrappers):',
    `faiss.ParameterSpace().set_index_parameter(index, "${param}", ${value})`,
  ].join('\n')
}

/** True when the values are spread enough that a log2 x-axis reads better. */
export function prefersLogAxis(values: number[]): boolean {
  if (values.length < 3) return false
  const min = Math.min(...values)
  const max = Math.max(...values)
  return min > 0 && max / min >= 8
}

/** Where a suggestion's "show me" button goes, or null when it has none. */
export function suggestionLink(
  s: Suggestion,
  param: SweepParam,
  k: number,
): { view: View; params: Record<string, string | number>; label: string } | null {
  switch (s.view) {
    case 'query': {
      if (s.query_id === null) return null
      const params: Record<string, string | number> = { id: String(s.query_id), k }
      if (s.at_value !== null) params[param === 'nprobe' ? 'nprobe' : 'ef'] = s.at_value
      return { view: 'query', params, label: `Explain query id ${s.query_id}` }
    }
    case 'overview':
      return { view: 'overview', params: {}, label: 'See list sizes' }
    case 'quantization':
      return { view: 'quantization', params: {}, label: 'See quantization error' }
    case 'compare':
      return { view: 'compare', params: {}, label: 'Compare indexes' }
    default:
      return null
  }
}
