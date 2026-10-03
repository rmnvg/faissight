// Public aliases for generated API contracts and UI-only refinements.
import type * as S from './schema.generated'

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

export type ApiErrorBody = S.ErrorResponse
export type Info = S.InfoResponse
export type ListSizes = S.ListSizesResponse
export type ListMembers = S.ListMembersResponse
export type JobStatus = S.JobStatusResponse
export type Projection = S.ProjectionResponse
export type QueryIn = S.QueryIn
export type SearchRequest = S.SearchRequest & Required<Pick<S.SearchRequest, 'k' | 'compare'>>
export type ResultRow = S.ResultRow
export type TruthRow = S.TruthRow
export type SearchResponse = S.SearchResponse
export type ProbeRow = S.ProbeRow
export type IvfTrace = S.IvfTraceResponse
export type MetadataRow = S.MetadataResponse
export type WorstQuery = S.WorstQueryOut
export type SweepPoint = S.SweepPointOut
export type SweepResult = S.SweepResultOut
export type SweepJob = S.SweepJobResponse
export type Suggestion = S.SuggestionOut
export type SweepAdvice = S.SweepAdviceResponse
export type SweepRun = S.SweepRunOut
export type PointDelta = S.PointDeltaOut
export type BaselineComparison = S.BaselineResponse
export type SweepRequest = S.SweepRequest & Required<Pick<S.SweepRequest, 'k' | 'n_queries'>>
export type CompareCandidate = S.CompareCandidateOut
export type CompareRequest = S.CompareRequest & Required<Pick<S.CompareRequest, 'candidate' | 'k' | 'n_queries'>>
export type CompareMeasurement = S.CompareMeasurementOut
export type QueryChange = S.QueryChangeOut
export type CompareResult = S.CompareResultOut
export type CompareJob = S.CompareJobResponse
export type HnswLevelStats = S.HnswLevelStats
export type HnswStats = S.HnswStatsResponse
export type HnswGraph = S.HnswGraphResponse
export type HnswVisit = S.HnswVisitOut
export type HnswStep = S.HnswStepOut
export type HnswLevelTrace = S.HnswLevelTrace
export type HnswTrace = S.HnswTraceResponse
export type PqError = S.PqErrorResponse
export type Snippet = Record<string, string>
export type SweepParam = 'nprobe' | 'efSearch'
export type SuggestionKind = Suggestion['kind']
export type HnswOutcome = 'FOUND' | 'VISITED_NOT_KEPT' | 'NOT_REACHED'
export type ProjectionOrStatus = Projection | JobStatus
export type PqErrorOrStatus = PqError | JobStatus
export type RunDecision = { target: number; confident: boolean; max_p95_ms: number | null }
export type {
  EvaluationJobResponse,
  EvaluationRequest,
  EvaluationResultOut,
  HistoryComparison,
  HistoryRecord,
  HistoryResponse,
  HistorySummary,
} from './schema.generated'
