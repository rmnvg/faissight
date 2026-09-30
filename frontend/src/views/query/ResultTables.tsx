import type { SearchResponse } from '../../api/types'
import { Card, ReasonBadge, SnippetText } from '../../components/ui'
import { fmtDist } from '../../lib/format'
import { navigate } from '../../lib/route'

export function ResultsTable({ result }: { result: SearchResponse }) {
  return (
    <Card title="Approximate results" subtitle={`What the index returned (${result.metric}, ${result.higher_is_closer ? 'higher' : 'lower'} = closer)`}>
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr>
              <th className="py-1 pr-3 font-normal">#</th>
              <th className="py-1 pr-3 font-normal">Id</th>
              <th className="py-1 pr-3 text-right font-normal">Distance</th>
              <th className="py-1 pr-3 text-right font-normal">List</th>
              <th className="py-1 pr-3 font-normal">In ground truth</th>
              <th className="py-1 font-normal">Text</th>
            </tr>
          </thead>
          <tbody>
            {result.results.map((r) => (
              <tr key={r.id} className="border-t border-line align-top">
                <td className="tabular py-1.5 pr-3 text-muted">{r.rank + 1}</td>
                <td className="tabular py-1.5 pr-3">
                  <button onClick={() => navigate('query', { id: r.id })} className="text-ink hover:text-series-1 hover:underline" title="Query with this vector">
                    {r.id}
                  </button>
                </td>
                <td className="tabular py-1.5 pr-3 text-right text-ink">{fmtDist(r.distance)}</td>
                <td className="tabular py-1.5 pr-3 text-right text-ink-2">{r.list_no ?? '—'}</td>
                <td className="py-1.5 pr-3">
                  {r.in_truth === null ? (
                    <span className="text-muted">—</span>
                  ) : r.in_truth ? (
                    <span className="inline-flex items-center gap-1 text-xs text-ink"><span className="h-2 w-2 rounded-full bg-good" />✓ yes</span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-xs text-ink-2"><span className="h-2 w-2 rounded-full bg-warning" />✕ no</span>
                  )}
                </td>
                <td className="py-1.5 text-xs"><SnippetText snippet={r.snippet} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

export function TruthTable({ result, isIvf }: { result: SearchResponse; isIvf: boolean }) {
  return (
    <Card title="Exact nearest neighbours" subtitle="Ground truth, and what happened to each one">
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr>
              <th className="py-1 pr-3 font-normal">#</th>
              <th className="py-1 pr-3 font-normal">Id</th>
              <th className="py-1 pr-3 text-right font-normal">Distance</th>
              {isIvf && <th className="py-1 pr-3 text-right font-normal">List</th>}
              {isIvf && <th className="py-1 pr-3 text-right font-normal">Probe rank</th>}
              <th className="py-1 pr-3 font-normal">Outcome</th>
              <th className="py-1 pr-3 text-right font-normal">Found at</th>
              <th className="py-1 font-normal">Text</th>
            </tr>
          </thead>
          <tbody>
            {result.truth!.map((t) => (
              <tr key={t.id} className="border-t border-line align-top">
                <td className="tabular py-1.5 pr-3 text-muted">{t.rank + 1}</td>
                <td className="tabular py-1.5 pr-3">
                  <button onClick={() => navigate('query', { id: t.id })} className="text-ink hover:text-series-1 hover:underline" title="Query with this vector">
                    {t.id}
                  </button>
                </td>
                <td className="tabular py-1.5 pr-3 text-right text-ink">{fmtDist(t.distance)}</td>
                {isIvf && <td className="tabular py-1.5 pr-3 text-right text-ink-2">{t.list_no ?? '—'}</td>}
                {isIvf && <td className="tabular py-1.5 pr-3 text-right text-ink-2">{t.probe_rank ?? '—'}</td>}
                <td className="py-1.5 pr-3">
                  <ReasonBadge reason={t.reason} />
                  {t.reason === 'QUANTIZATION' && (
                    <button
                      onClick={() => navigate('quantization')}
                      className="ml-1.5 text-xs text-series-1 hover:underline"
                      title="See how much the codes distort distances"
                    >
                      see quantization →
                    </button>
                  )}
                </td>
                <td className="tabular py-1.5 pr-3 text-right text-ink-2">{t.found_rank === null ? '—' : t.found_rank + 1}</td>
                <td className="py-1.5 text-xs"><SnippetText snippet={t.snippet} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
