import { Card } from '../../components/ui'
import { navigate } from '../../lib/route'
import type { RunResult } from './types'

/** What to look at next when there is no IVF probe chart: the HNSW trace, or why not. */
export function SearchNotes({ result, isIvf, isHnsw }: { result: RunResult; isIvf: boolean; isHnsw: boolean }) {
  return (
    <Card title={isHnsw ? 'How HNSW found these' : isIvf ? 'Probe order' : 'Search parameters'}>
      {isHnsw ? (
        <div className="flex flex-col items-start gap-3 text-sm text-ink-2">
          <p>
            Replay this search step by step: the greedy descent through the upper layers,
            then the candidate expansion on level 0, and which true neighbours were never
            reached. Raising efSearch widens the search.
          </p>
          {result.request.query.id !== undefined ? (
            <button
              onClick={() =>
                navigate('hnsw', { id: result.request.query.id ?? null, ef: result.request.efSearch ?? null })
              }
              className="rounded-lg bg-series-1 px-3 py-1.5 font-medium text-white hover:opacity-90"
            >
              Open the search trace →
            </button>
          ) : (
            <p className="text-xs text-muted">The trace view takes a stored id as the query.</p>
          )}
        </div>
      ) : isIvf ? (
        <p className="text-sm text-ink-2">
          Enable “compare with exact” to see the probe order and which cells held the
          true neighbours.
        </p>
      ) : (
        <p className="text-sm text-ink-2">
          Flat indexes search exhaustively, so every true neighbour is found.
        </p>
      )}
    </Card>
  )
}
