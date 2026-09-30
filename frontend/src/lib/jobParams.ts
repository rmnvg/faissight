import type { CompareRequest, SweepRequest } from '../api/types'
import { intParam } from './route'

type Params = Record<string, string | number | null>

/** Settings shared by sweeps and comparisons, as they appear in the URL. */
export interface RunSettings {
  k: number
  nQueries: number
  repeats: number
  seed: number
}

const DEFAULT_RUN: RunSettings = { k: 10, nQueries: 200, repeats: 3, seed: 0 }

/** `k`, `n`, `repeats` and `seed` from the URL, falling back to defaults when absent or invalid. */
export function runSettingsFromParams(params: URLSearchParams): RunSettings {
  const pos = (key: string, fallback: number, min = 1) => {
    const v = intParam(params, key)
    return v !== null && v >= min ? v : fallback
  }
  return {
    k: pos('k', DEFAULT_RUN.k),
    nQueries: pos('n', DEFAULT_RUN.nQueries),
    repeats: pos('repeats', DEFAULT_RUN.repeats),
    seed: pos('seed', DEFAULT_RUN.seed, 0),
  }
}

/** URL params for a sweep: its job id plus the settings needed to run it again. */
export function sweepParams(jobId: string, req: SweepRequest): Params {
  return {
    job: jobId,
    values: req.values?.join(',') ?? null,
    k: req.k,
    n: req.n_queries,
    repeats: req.repeats ?? null,
    seed: req.seed ?? null,
  }
}

/** The values list from a sweep URL, as the form's text ("1, 2, 4"), or null if absent. */
export function sweepValuesText(params: URLSearchParams): string | null {
  const v = params.get('values')
  if (!v) return null
  const nums = v.split(',').map(Number)
  return nums.every((n) => Number.isInteger(n) && n > 0) ? nums.join(', ') : null
}

/** URL params for a comparison: its job id plus the settings needed to run it again. */
export function compareParams(jobId: string, req: CompareRequest): Params {
  return {
    job: jobId,
    candidate: req.candidate,
    k: req.k,
    n: req.n_queries,
    repeats: req.repeats ?? null,
    seed: req.seed ?? null,
    left: req.left_nprobe ?? req.left_ef_search ?? null,
    right: req.right_nprobe ?? req.right_ef_search ?? null,
  }
}
