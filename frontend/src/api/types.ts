// Mirrors src/faissight/server/schemas.py.

export type Metric = 'L2' | 'IP' | 'OTHER'
export type IndexKind =
  | 'FLAT'
  | 'IVF_FLAT'
  | 'IVF_PQ'
  | 'IVF_SQ'
  | 'HNSW_FLAT'
  | 'HNSW_OTHER'
  | 'UNSUPPORTED'
export type MissReason = 'FOUND' | 'CELL_NOT_PROBED' | 'QUANTIZATION' | 'TRANSFORM' | 'RANKED_OUT'
export type ProjectionMethod = 'pca' | 'umap'
export type Dims = 2 | 3

export interface ApiErrorBody {
  error_code: string
  message: string
  hint: string | null
}

export interface Info {
  version: string
  name: string
  kind: IndexKind
  class_chain: string[]
  transforms: { name: string; d_in: number; d_out: number }[]
  d: number
  core_d: number
  ntotal: number
  metric: Metric
  higher_is_closer: boolean
  is_trained: boolean
  params: Record<string, number | boolean | string>
  has_id_map: boolean
  has_refine: boolean
  supported: boolean
  unsupported_reason: string | null
  ground_truth_source: 'raw' | 'reconstructed' | null
  max_points: number
  inputs: {
    raw_vectors: boolean
    metadata_rows: number | null
    metadata_columns: string[]
    metadata_coverage: number | null
    embedder: string | null
    embedder_status: 'loading' | 'ready' | 'failed' | null
    queries: number | null
  }
}

export interface ListSizes {
  nlist: number
  sizes: number[]
  n_empty: number
  imbalance_factor: number
  min: number
  median: number
  max: number
  top_lists: { list_no: number; size: number }[]
  top_5pct_share: number
}

export interface Snippet {
  title?: string
  text?: string
}

export interface ListMembers {
  list_no: number
  size: number
  offset: number
  limit: number
  members: { id: number; snippet: Snippet | null }[]
}

export interface JobStatus {
  status: 'running' | 'done' | 'failed'
  progress: number
  message: string
  error: string | null
}

export interface Projection {
  status: 'done'
  kind: 'points' | 'centroids'
  method: ProjectionMethod
  dims: Dims
  ids: number[]
  x: number[]
  y: number[]
  z: number[] | null
  list_nos: number[] | null
  n_total: number
  sampled: boolean
  explained_variance: number[] | null
}

export type ProjectionOrStatus = Projection | JobStatus

export interface QueryIn {
  id?: number
  vector?: number[]
  text?: string
}

export interface SearchRequest {
  query: QueryIn
  k: number
  nprobe?: number
  efSearch?: number
  compare: boolean
  projection?: { method: ProjectionMethod; dims: Dims }
}

export interface ResultRow {
  rank: number
  id: number
  distance: number
  list_no: number | null
  in_truth: boolean | null
  snippet: Snippet | null
}

export interface TruthRow {
  rank: number
  id: number
  distance: number
  list_no: number | null
  probe_rank: number | null
  reason: MissReason | null
  found_rank: number | null
  snippet: Snippet | null
}

export interface SearchResponse {
  query_kind: 'id' | 'vector' | 'text'
  metric: Metric
  higher_is_closer: boolean
  k: number
  params: Record<string, number>
  latency_ms: number
  results: ResultRow[]
  truth: TruthRow[] | null
  recall: number | null
  truth_source: 'raw' | 'reconstructed' | null
  min_nprobe: number | null
  reason_counts: Record<MissReason, number> | null
  query_coords: number[] | null
}

export interface ProbeRow {
  probe_rank: number
  list_no: number
  centroid_distance: number
  probed: boolean
  size: number
  n_true_neighbours: number
}

export interface IvfTrace {
  nprobe: number
  nlist: number
  min_nprobe: number
  recall: number | null
  metric: Metric
  higher_is_closer: boolean
  probes: ProbeRow[]
  neighbours: TruthRow[]
}

export interface MetadataRow {
  id: number
  row: Record<string, unknown>
}
