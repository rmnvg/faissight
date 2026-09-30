import type { Info } from '../api/types'
import { Banner, Card, EmptyState, Spinner } from '../components/ui'
import type { ThemeMode } from '../lib/theme'
import { ProbeChart } from './query/ProbeChart'
import { QueryForm } from './query/QueryForm'
import { QueryMap } from './query/QueryMap'
import { SearchNotes } from './query/SearchNotes'
import { ResultsTable, TruthTable } from './query/ResultTables'
import { Headline, ShareBar } from './query/Summary'
import { useQueryRunner } from './query/useQueryRunner'

export default function QueryExplorer({
  info,
  params,
  mode,
}: {
  info: Info
  params: URLSearchParams
  mode: ThemeMode
}) {
  const runner = useQueryRunner(info, params)
  const { isIvf, isHnsw, run } = runner
  const result = run.data

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 p-6">
      <header>
        <h1 className="text-xl font-semibold text-ink">Query explorer</h1>
        <p className="text-sm text-ink-2">
          Search the index, compare with exact ground truth, and see why true neighbours were missed.
        </p>
      </header>

      <QueryForm info={info} runner={runner} />

      {!result && !run.isPending && (
        <Card>
          <EmptyState title="Run a query to begin">
            Try a stored id (every vector in the index can be a query), or pick one from the cluster
            map with “Query →”.
          </EmptyState>
        </Card>
      )}
      {run.isPending && !result && <Spinner label="Searching…" />}

      {result && (
        <div className={`flex flex-col gap-4 ${run.isPending ? 'opacity-60' : ''}`}>
          <ShareBar result={result} />
          <Headline result={result} isIvf={isIvf} />
          {result.search.truth_source === 'reconstructed' && (
            <Banner tone="warning">
              Ground truth computed on reconstructed vectors; PQ/SQ error is not measured.
            </Banner>
          )}
          <div className="grid gap-4 xl:grid-cols-2">
            <QueryMap info={info} result={result} mode={mode} isIvf={isIvf} />
            {isIvf && result.trace ? (
              <ProbeChart trace={result.trace} />
            ) : (
              <SearchNotes result={result} isIvf={isIvf} isHnsw={isHnsw} />
            )}
          </div>
          <ResultsTable result={result.search} />
          {result.search.truth && <TruthTable result={result.search} isIvf={isIvf} />}
        </div>
      )}
    </div>
  )
}
