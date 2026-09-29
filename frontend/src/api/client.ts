import type { UserId } from './types'
import type {
  ApiErrorBody,
  Dims,
  HnswGraph,
  HnswStats,
  HnswTrace,
  Info,
  ListMembers,
  ListSizes,
  MetadataRow,
  PqErrorOrStatus,
  ProjectionMethod,
  ProjectionOrStatus,
  SearchRequest,
  SearchResponse,
  SweepJob,
  SweepRequest,
} from './types'

/** A structured API error ({error_code, message, hint}) or a network failure. */
export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly hint: string | null

  constructor(status: number, body: ApiErrorBody) {
    super(body.message)
    this.name = 'ApiError'
    this.status = status
    this.code = body.error_code
    this.hint = body.hint
  }
}

// Relative URLs so the UI works under any path prefix (Jupyter, Colab, HF Spaces).
const BASE = 'api'

/** Every request gets a deadline so a hung server surfaces as an error, not an endless spinner.
 * Long work (projections, sweeps, PQ analysis) runs as polled jobs, so this only bounds one call. */
export const REQUEST_TIMEOUT_MS = 60_000

/** True for requests cancelled by their caller (superseded or unmounted), which aren't failures. */
export function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = REQUEST_TIMEOUT_MS,
): Promise<{ status: number; body: T }> {
  // One controller carries both the caller's cancellation and the deadline.
  const ctrl = new AbortController()
  const caller = init.signal
  const onCallerAbort = () => ctrl.abort()
  if (caller?.aborted) ctrl.abort()
  else caller?.addEventListener('abort', onCallerAbort, { once: true })
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    ctrl.abort()
  }, timeoutMs)

  let res: Response
  let text: string
  try {
    res = await fetch(`${BASE}/${path}`, { ...init, signal: ctrl.signal })
    text = await res.text()
  } catch {
    if (caller?.aborted) throw new DOMException('Request cancelled.', 'AbortError')
    throw new ApiError(0, timedOut
      ? {
          error_code: 'TIMEOUT',
          message: `The faissight server did not answer within ${timeoutMs / 1000} s.`,
          hint: 'Check the terminal running `faissight serve`; try a smaller k or nprobe.',
        }
      : {
          error_code: 'NETWORK_ERROR',
          message: 'Could not reach the faissight server.',
          hint: 'Is `faissight serve` still running?',
        })
  } finally {
    clearTimeout(timer)
    caller?.removeEventListener('abort', onCallerAbort)
  }
  let body: unknown = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = null
  }
  if (!res.ok) {
    const err = body as Partial<ApiErrorBody> | null
    throw new ApiError(res.status, {
      error_code: err?.error_code ?? `HTTP_${res.status}`,
      message: err?.message ?? (text || res.statusText),
      hint: err?.hint ?? null,
    })
  }
  return { status: res.status, body: body as T }
}

const get = async <T>(path: string, signal?: AbortSignal) => (await request<T>(path, { signal })).body
const post = async <T>(path: string, payload: unknown, signal?: AbortSignal) =>
  (
    await request<T>(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    })
  ).body

// Reads take an optional AbortSignal (React Query passes one to queryFn) so obsolete
// requests are cancelled rather than left to finish in the background.
export const api = {
  info: (signal?: AbortSignal) => get<Info>('info', signal),
  ivfLists: (signal?: AbortSignal) => get<ListSizes>('ivf/lists', signal),
  listMembers: (listNo: number, offset: number, limit: number, signal?: AbortSignal) =>
    get<ListMembers>(`ivf/list/${listNo}?offset=${offset}&limit=${limit}`, signal),
  /** Resolves to a JobStatus (HTTP 202) while the projection is still computing. */
  projection: (
    kind: 'points' | 'centroids',
    method: ProjectionMethod,
    dims: Dims,
    signal?: AbortSignal,
  ) => get<ProjectionOrStatus>(`projection?kind=${kind}&method=${method}&dims=${dims}`, signal),
  search: (req: SearchRequest, signal?: AbortSignal) => post<SearchResponse>('search', req, signal),
  metadata: (id: UserId, signal?: AbortSignal) => get<MetadataRow>(`metadata/${id}`, signal),
  /** Starts (or reuses) a sweep; the job may still be running (HTTP 202). */
  startSweep: (req: SweepRequest) => post<SweepJob>('sweep', req),
  cancelSweep: async (jobId: string) =>
    (await request<SweepJob>(`sweep/${encodeURIComponent(jobId)}`, { method: 'DELETE' })).body,
  sweep: (jobId: string, signal?: AbortSignal) =>
    get<SweepJob>(`sweep/${encodeURIComponent(jobId)}`, signal),
  hnswStats: (signal?: AbortSignal) => get<HnswStats>('hnsw/stats', signal),
  hnswGraph: (level: number, limit: number, around: UserId | null, signal?: AbortSignal) =>
    get<HnswGraph>(
      `hnsw/graph?level=${level}&limit=${limit}${around === null ? '' : `&around=${around}`}`,
      signal,
    ),
  traceHnsw: (req: SearchRequest, signal?: AbortSignal) =>
    post<HnswTrace>('trace/hnsw', req, signal),
  /** Resolves to a JobStatus (HTTP 202) while the analysis is still computing. */
  pqError: (signal?: AbortSignal) => get<PqErrorOrStatus>('pq/error', signal),
}
