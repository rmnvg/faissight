import { useState } from 'react'
import type { MissReason } from '../../api/types'
import { ReasonBadge, StatTile } from '../../components/ui'
import { downloadCSV, downloadJSON } from '../../lib/exportData'
import { navigate } from '../../lib/route'
import type { RunResult } from './types'

export function Headline({ result, isIvf }: { result: RunResult; isIvf: boolean }) {
  const s = result.search
  const counts = s.reason_counts
  const misses = counts
    ? (Object.entries(counts) as [MissReason, number][]).filter(([r, n]) => r !== 'FOUND' && n > 0)
    : []
  const used = s.params.nprobe
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <StatTile
        label={`Recall@${s.k}`}
        value={s.recall === null ? '—' : s.recall.toFixed(2)}
        tone={s.recall === null ? 'default' : s.recall >= 1 ? 'good' : s.recall < 0.9 ? 'bad' : 'default'}
        caption={s.recall === null ? 'enable “compare with exact”' : `${Math.round(s.recall * s.k)} of ${s.k} true neighbours found`}
      />
      {isIvf ? (
        <StatTile
          label="Min nprobe for all true neighbours"
          value={result.trace?.min_nprobe ?? s.min_nprobe ?? '—'}
          caption={used !== undefined ? `you searched with nprobe ${used}` : undefined}
        />
      ) : (
        <StatTile label="Parameters" value={Object.entries(s.params).map(([k, v]) => `${k} ${v}`).join(', ') || 'none'} />
      )}
      <StatTile label="Latency" value={`${s.latency_ms.toFixed(2)} ms`} caption="single query, server side" />
      <div className="rounded-xl border border-line bg-surface px-4 py-3">
        <div className="text-xs text-ink-2">Misses by reason</div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {misses.length === 0 ? (
            <span className="text-sm text-ink-2">{s.truth ? 'No misses' : '—'}</span>
          ) : (
            misses.map(([r, n]) => (
              <span key={r} className="inline-flex items-center gap-1 text-sm">
                <ReasonBadge reason={r} /> <span className="tabular text-ink">×{n}</span>
                {r === 'QUANTIZATION' && (
                  <button
                    onClick={() => navigate('quantization')}
                    className="text-xs whitespace-nowrap text-series-1 hover:underline"
                  >
                    why? →
                  </button>
                )}
              </span>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export function ShareBar({ result }: { result: RunResult }) {
  const [copied, setCopied] = useState(false)
  const shareable = result.request.query.vector === undefined
  const stamp = result.request.query.id !== undefined ? `id${result.request.query.id}` : 'query'
  const button =
    'rounded-md border border-line bg-surface px-2.5 py-1 text-xs text-ink-2 hover:text-ink'
  return (
    <div className="flex flex-wrap items-center justify-end gap-2 print:hidden">
      {shareable && (
        <button
          className={button}
          onClick={() =>
            navigator.clipboard?.writeText(window.location.href).then(
              () => {
                setCopied(true)
                window.setTimeout(() => setCopied(false), 1500)
              },
              () => setCopied(false),
            )
          }
        >
          {copied ? 'Link copied ✓' : 'Copy link'}
        </button>
      )}
      <button className={button} onClick={() => downloadCSV(`faissight-${stamp}-results.csv`, result.search.results as never)}>
        Results CSV
      </button>
      {result.search.reranked && (
        <button className={button} onClick={() => downloadCSV(`faissight-${stamp}-reranked.csv`, result.search.reranked as never)}>
          Reranked CSV
        </button>
      )}
      {result.search.truth && (
        <button className={button} onClick={() => downloadCSV(`faissight-${stamp}-truth.csv`, result.search.truth as never)}>
          Ground truth CSV
        </button>
      )}
      <button
        className={button}
        onClick={() =>
          downloadJSON(`faissight-${stamp}.json`, {
            request: result.request,
            search: result.search,
            ivf_trace: result.trace,
          })
        }
      >
        JSON
      </button>
    </div>
  )
}
