import type { IvfTrace, SearchRequest, SearchResponse } from '../../api/types'

export type QueryMode = 'text' | 'id' | 'vector' | 'row'

/** One finished search, with the request that produced it. */
export interface RunResult {
  search: SearchResponse
  trace: IvfTrace | null
  request: SearchRequest
}
