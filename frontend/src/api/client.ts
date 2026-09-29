import type { UserId } from './types'
import type {
  ApiErrorBody,
  Dims,
  HnswGraph,
  HnswStats,
  HnswTrace,
  Info,
  IvfTrace,
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

async function request<T>(path: string, init?: RequestInit): Promise<{ status: number; body: T }> {
  let res: Response
  try {
    res = await fetch(`${BASE}/${path}`, init)
  } catch {
    throw new ApiError(0, {
      error_code: 'NETWORK_ERROR',
      message: 'Could not reach the faissight server.',
      hint: 'Is `faissight serve` still running?',
    })
  }
  const text = await res.text()
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

const get = async <T>(path: string) => (await request<T>(path)).body
const post = async <T>(path: string, payload: unknown) =>
  (
    await request<T>(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    })
  ).body

export const api = {
  info: () => get<Info>('info'),
  ivfLists: () => get<ListSizes>('ivf/lists'),
  listMembers: (listNo: number, offset: number, limit: number) =>
    get<ListMembers>(`ivf/list/${listNo}?offset=${offset}&limit=${limit}`),
  /** Resolves to a JobStatus (HTTP 202) while the projection is still computing. */
  projection: (kind: 'points' | 'centroids', method: ProjectionMethod, dims: Dims) =>
    get<ProjectionOrStatus>(`projection?kind=${kind}&method=${method}&dims=${dims}`),
  search: (req: SearchRequest) => post<SearchResponse>('search', req),
  traceIvf: (req: SearchRequest) => post<IvfTrace>('trace/ivf', req),
  metadata: (id: UserId) => get<MetadataRow>(`metadata/${id}`),
  /** Starts (or reuses) a sweep; the job may still be running (HTTP 202). */
  startSweep: (req: SweepRequest) => post<SweepJob>('sweep', req),
  cancelSweep: async (jobId: string) =>
    (await request<SweepJob>(`sweep/${encodeURIComponent(jobId)}`, { method: 'DELETE' })).body,
  sweep: (jobId: string) => get<SweepJob>(`sweep/${encodeURIComponent(jobId)}`),
  hnswStats: () => get<HnswStats>('hnsw/stats'),
  hnswGraph: (level: number, limit: number, around: UserId | null) =>
    get<HnswGraph>(
      `hnsw/graph?level=${level}&limit=${limit}${around === null ? '' : `&around=${around}`}`,
    ),
  traceHnsw: (req: SearchRequest) => post<HnswTrace>('trace/hnsw', req),
  /** Resolves to a JobStatus (HTTP 202) while the analysis is still computing. */
  pqError: () => get<PqErrorOrStatus>('pq/error'),
}
