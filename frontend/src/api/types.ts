// Mirrors src/faissight/server/schemas.py.

/** Small ids are numbers; larger int64 ids are lossless decimal strings. */
export type UserId = number | string

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
  sweep: { param: SweepParam; values: number[]; max_value: number | null } | null
  demo_limits: {
    max_k: number
    max_ef_search: number
    max_sweep_queries: number
    max_sweep_values: number
    max_sweep_k: number
    umap_from_cache_only: boolean
  } | null
  memory: {
    /** Raw vectors (n x d x 4); null without --vectors. */
    vectors_bytes: number | null
    /** Raw vectors are read from a memory-mapped file (--mmap), not held in RAM. */
    vectors_mapped: boolean
    /** Why --mmap could not keep the vectors mapped, if so. */
    mmap_note: string | null
    /** Memory needed to decode every stored vector from the index. */
    reconstruct_bytes: number
    /** The decoded vectors are already in memory. */
    reconstructed: boolean
  }
  /** Indexes given with --compare; empty when there are none. */
  compare: CompareCandidate[]
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
  members: { id: UserId; snippet: Snippet | null }[]
}

export interface JobStatus {
  status: 'running' | 'done' | 'failed' | 'cancelled'
  progress: number
  message: string
  error: string | null
}

export interface Projection {
  status: 'done'
  kind: 'points' | 'centroids'
  method: ProjectionMethod
  dims: Dims
  ids: UserId[]
  x: number[]
  y: number[]
  z: number[] | null
  list_nos: number[] | null
  n_total: number
  sampled: boolean
  explained_variance: number[] | null
  /** Set when the projection couldn't be saved to the disk cache (it is still valid). */
  cache_warning?: string | null
}

export type ProjectionOrStatus = Projection | JobStatus

export interface QueryIn {
  id?: UserId
  vector?: number[]
  text?: string
  /** Row of the evaluation query set given with --queries (as sweeps number them). */
  row?: number
}

export interface SearchRequest {
  query: QueryIn
  k: number
  nprobe?: number
  efSearch?: number
  compare: boolean
  /** IVF only, needs compare: include the probe trace in the search response. */
  trace?: boolean
  projection?: { method: ProjectionMethod; dims: Dims }
}

export interface ResultRow {
  rank: number
  id: UserId
  distance: number
  list_no: number | null
  in_truth: boolean | null
  snippet: Snippet | null
}

export interface TruthRow {
  rank: number
  id: UserId
  distance: number
  list_no: number | null
  probe_rank: number | null
  reason: MissReason | null
  found_rank: number | null
  snippet: Snippet | null
}

export interface SearchResponse {
  query_kind: 'id' | 'vector' | 'text' | 'row'
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
  ivf_trace: IvfTrace | null
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
  id: UserId
  row: Record<string, unknown>
}

export type SweepParam = 'nprobe' | 'efSearch'

export interface WorstQuery {
  /** Row in the query set. */
  query_no: number
  /** The query's own stored id (sampled queries); null for given queries. */
  id: UserId | null
  recall: number
  /** IVF: share of its true neighbours in the probed lists (caps its recall). */
  probe_coverage: number | null
}

export interface SweepPoint {
  value: number
  recall: number
  latency_mean_ms: number
  latency_p95_ms: number
  /** Approximate 95% interval for the mean recall. */
  recall_ci_low: number | null
  recall_ci_high: number | null
  /** Exact per-query recall distribution as [recall, n_queries], lowest recall first. */
  recall_distribution: [number, number][]
  /** Lowest-recall queries, worst first. */
  worst_queries: WorstQuery[]
  /** IVF: mean share of true neighbours in probed lists, the most recall probing allows. */
  probe_coverage: number | null
}

export interface SweepResult {
  param: SweepParam
  k: number
  n_queries: number
  query_origin: 'given' | 'sampled'
  truth_source: 'raw' | 'reconstructed'
  points: SweepPoint[]
  pareto_values: number[]
  repeats: number
  seed: number
  query_sha256: string
  query_seed: number | null
  environment: Record<string, string | number>
}

export interface SweepJob {
  job_id: string
  status: 'running' | 'done' | 'failed' | 'cancelled'
  progress: number
  message: string
  error: string | null
  result: SweepResult | null
}

export type SuggestionKind =
  | 'PROBE_MORE'
  | 'RANKING_LIMIT'
  | 'SEARCH_WIDER'
  | 'RECALL_PLATEAU'
  | 'FAILING_QUERIES'
  | 'LIST_IMBALANCE'
  | 'LATENCY_BUDGET'

/** A suggested next step from a sweep, with the measurements behind it. */
export interface Suggestion {
  kind: SuggestionKind
  title: string
  detail: string
  evidence: { label: string; value: string }[]
  /** Values for a follow-up sweep of the same parameter. */
  sweep_values: number[] | null
  /** A view that shows more ('query' together with `query_id`). */
  view: 'overview' | 'quantization' | 'compare' | 'query' | null
  query_id: UserId | null
  /** The parameter value the evidence was measured at. */
  at_value: number | null
}

export interface SweepAdvice {
  target_recall: number
  confident: boolean
  max_p95_ms: number | null
  /** Most important first; empty when nothing needs changing. */
  suggestions: Suggestion[]
}

/** The Tuner decision a saved run records, or a baseline comparison is judged by. */
export interface RunDecision {
  target: number
  confident: boolean
  max_p95_ms: number | null
}

/** A saved sweep run (faissight.sweep-run); kept as opaque JSON by the UI. */
export type SweepRun = Record<string, unknown> & { format: string; label?: string | null }

export interface PointDelta {
  value: number
  baseline_recall: number
  recall: number
  recall_change: number
  baseline_p95_ms: number
  p95_ms: number
  /** Relative (0.2 = 20% slower). */
  p95_change: number
  recall_regressed: boolean
  latency_regressed: boolean
}

export interface BaselineComparison {
  baseline_label: string | null
  baseline_created_at: string | null
  baseline_index: string | null
  points: PointDelta[]
  /** False when no setting could be judged (see notes): not a pass, nothing ran. */
  comparable: boolean
  regressed: boolean
  recall_comparable: boolean
  latency_comparable: boolean
  /** Why the runs may not be comparable. */
  notes: string[]
  baseline_recommended: number | null
  recommended: number | null
  max_recall_drop: number
  max_p95_increase: number
  min_p95_increase_ms: number
}

export interface SweepRequest {
  param?: SweepParam
  values?: number[]
  k: number
  n_queries: number
  repeats?: number
  seed?: number
}

export interface CompareCandidate {
  index: number
  name: string
  kind: IndexKind
  ntotal: number
  params: Record<string, number | boolean | string>
  /** The speed/recall knob of this index, if any. */
  search_param: SweepParam | null
  /** Largest accepted value (nlist for nprobe; null when unbounded). */
  max_value: number | null
}

export interface CompareRequest {
  candidate: number
  k: number
  n_queries: number
  repeats?: number
  seed?: number
  left_nprobe?: number
  right_nprobe?: number
  left_ef_search?: number
  right_ef_search?: number
}

export interface CompareMeasurement {
  name: string
  kind: IndexKind
  params: Record<string, number>
  /** Size of faiss.serialize_index; not resident memory. */
  serialized_bytes: number
  recall: number
  recall_ci_low: number | null
  recall_ci_high: number | null
  latency_mean_ms: number
  latency_p95_ms: number
}

export interface QueryChange {
  query_no: number
  /** The query's own stored id (sampled queries); null for given queries. */
  id: UserId | null
  left_recall: number
  right_recall: number
  left_only: UserId[]
  right_only: UserId[]
  overlap: number
}

export interface CompareResult {
  metric: Metric
  k: number
  n_queries: number
  query_origin: 'given' | 'sampled'
  query_sha256: string
  query_seed: number | null
  repeats: number
  seed: number
  environment: Record<string, string | number>
  left: CompareMeasurement
  right: CompareMeasurement
  n_changed: number
  n_improved: number
  n_worsened: number
  /** Changed queries, largest recall change first (capped server-side). */
  changes: QueryChange[]
  changes_truncated: boolean
}

export interface CompareJob {
  job_id: string
  status: 'running' | 'done' | 'failed' | 'cancelled'
  progress: number
  message: string
  error: string | null
  result: CompareResult | null
}

export interface HnswLevelStats {
  level: number
  n_nodes: number
  max_links: number
  degree_mean: number
  degree_min: number
  degree_max: number
  degree_hist: number[]
}

export interface HnswStats {
  entry_point: UserId
  max_level: number
  m: number
  ef_search: number
  ef_construction: number
  levels: HnswLevelStats[]
}

export interface HnswGraph {
  level: number
  n_level_nodes: number
  sampled: boolean
  ids: UserId[]
  x: number[]
  y: number[]
  top_levels: number[]
  edges_src: number[]
  edges_dst: number[]
}

export interface HnswVisit {
  node: UserId
  distance: number
  accepted: boolean
}

export interface HnswStep {
  expanded: UserId
  expanded_distance: number
  visits: HnswVisit[]
}

export interface HnswLevelTrace {
  level: number
  entry: UserId
  steps: HnswStep[]
}

export type HnswOutcome = 'FOUND' | 'VISITED_NOT_KEPT' | 'NOT_REACHED'

export interface HnswTrace {
  metric: Metric
  higher_is_closer: boolean
  ef_search: number
  k: number
  entry_point: UserId
  max_level: number
  levels: HnswLevelTrace[]
  results: ResultRow[]
  faiss_ids: UserId[]
  overlap_with_faiss: number
  truth: {
    rank: number
    id: UserId
    distance: number
    outcome: HnswOutcome
    top_level: number
    snippet: Snippet | null
  }[] | null
  recall: number | null
  nodes: { ids: UserId[]; x: number[]; y: number[]; top_levels: number[] }
  query_xy: number[] | null
}

export interface PqError {
  available: boolean
  reason: string | null
  hint: string | null
  kind: IndexKind | null
  metric: Metric | null
  n: number | null
  code_size: number | null
  raw_bytes: number | null
  has_transform: boolean
  mean: number | null
  median: number | null
  p95: number | null
  max: number | null
  relative_mean: number | null
  histogram: { edges: number[]; counts: number[] } | null
  per_list: { list_no: number; size: number; mean_error: number }[] | null
  worst: { id: UserId; error: number; relative: number; list_no: number | null; snippet: Snippet | null }[] | null
  distortion: {
    true: number[]
    approx: number[]
    near: boolean[]
    correlation: number
    near_correlation: number
  } | null
}

export type PqErrorOrStatus = PqError | JobStatus
