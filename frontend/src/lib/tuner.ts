import type { SweepParam, SweepPoint } from '../api/types'

/** Smallest parameter value whose mean recall meets the target (mirrors core.sweep). */
export function recommend(points: SweepPoint[], target: number): SweepPoint | null {
  const ok = points.filter((p) => p.recall >= target - 1e-9)
  if (ok.length === 0) return null
  return ok.reduce((a, b) => (b.value < a.value ? b : a))
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
