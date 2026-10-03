import { useState } from 'react'
import { api } from '../api/client'
import { useJob } from '../api/jobs'
import type { EvaluationJobResponse, EvaluationRequest, EvaluationResultOut, Info } from '../api/types'
import { Field, NumberInput } from '../components/form'
import { JobStatus } from '../components/JobStatus'
import { Banner, Card, EmptyState, Segmented, StatTile } from '../components/ui'
import { downloadCSV, downloadJSON } from '../lib/exportData'
import { navigate } from '../lib/route'

const PAGE = 50
const MAX_BYTES = 5_000_000
const BUTTON =
  'rounded-md border border-line bg-surface px-3 py-1.5 text-sm text-ink hover:bg-accent-wash disabled:opacity-50'
const METRICS = [
  ['recall', 'Recall'],
  ['mrr', 'MRR'],
  ['ndcg', 'nDCG'],
] as const

type Order = 'worst' | 'row' | 'gain'
type Row = EvaluationResultOut['queries'][number]

/** Signed change with three decimals, '0.000' for negligible differences. */
function signed(delta: number): string {
  if (Math.abs(delta) < 5e-4) return '0.000'
  return `${delta > 0 ? '+' : '−'}${Math.abs(delta).toFixed(3)}`
}

function sortRows(rows: Row[], order: Order): Row[] {
  const gain = (r: Row) => (r.reranked_metrics?.ndcg ?? r.metrics.ndcg) - r.metrics.ndcg
  const sorted = [...rows]
  if (order === 'worst') sorted.sort((a, b) => a.metrics.ndcg - b.metrics.ndcg || a.row - b.row)
  if (order === 'gain') sorted.sort((a, b) => gain(b) - gain(a) || a.row - b.row)
  return sorted
}

export default function Evaluation({ info, params }: { info: Info; params: URLSearchParams }) {
  const ivf = info.kind.startsWith('IVF')
  const hnsw = info.kind.startsWith('HNSW')
  const nQueries = info.inputs.queries ?? 0
  const ready = info.inputs.raw_vectors && nQueries > 0
  const [judgements, setJudgements] = useState('')
  const [fileName, setFileName] = useState<string | null>(null)
  const [k, setK] = useState(10)
  const [candidates, setCandidates] = useState(0)
  const [breadth, setBreadth] = useState(Number(ivf ? (info.params.nprobe ?? 1) : (info.params.ef_search ?? 16)))
  const [formError, setFormError] = useState<string | null>(null)

  const { job, poll, start, cancel, missing } = useJob<EvaluationRequest, EvaluationJobResponse>({
    kind: 'evaluation',
    jobId: params.get('job'),
    start: api.evaluate,
    fetch: api.evaluation,
    cancel: api.cancelEvaluation,
    onStarted: (j) => navigate('evaluation', { job: j.job_id }),
  })
  const busy = start.isPending || job?.status === 'running'

  const run = () => {
    if (candidates > 0 && candidates < k) {
      setFormError('Rerank candidates must be at least k (or 0 to skip reranking).')
      return
    }
    setFormError(null)
    start.mutate({
      judgements,
      k,
      candidates: candidates || null,
      ...(ivf ? { nprobe: breadth } : hnsw ? { ef_search: breadth } : {}),
    })
  }

  const readFile = async (file: File | undefined) => {
    setFormError(null)
    if (!file) return
    if (file.size > MAX_BYTES) {
      setFormError('Choose a judgements file smaller than 5 MB.')
      return
    }
    try {
      setJudgements(await file.text())
      setFileName(file.name)
    } catch {
      setFormError('Could not read this file.')
    }
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-4 p-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Relevance</h1>
        <p className="mt-1 text-sm text-ink-2">
          Score retrieval against human judgements: do the returned chunks answer the question? ANN recall (in
          the Tuner) measures agreement with exact search instead; an index can match exact search and still
          return irrelevant chunks.
        </p>
      </header>

      {!ready ? (
        <EmptyState title="Needs held-out queries and raw vectors">
          Restart with <code>--queries queries.npy</code> and <code>--vectors vectors.npy</code>, then supply one
          judgement row per query.
        </EmptyState>
      ) : (
        <Card
          title="Judgements"
          subtitle={`One JSONL row for each of the ${nQueries.toLocaleString()} evaluation queries (rows 0–${nQueries - 1}).`}
        >
          <p className="mb-3 text-xs text-ink-2">
            Format: <code className="font-mono">{'{"row": 0, "relevant": {"42": 2, "77": 1}}'}</code>. Keys are chunk
            ids as decimal strings, values are grades; every query needs at least one positive grade.
          </p>
          <div className="flex flex-col gap-3">
            <Field label={fileName ? `Judgements file (loaded ${fileName})` : 'Judgements file'}>
              <input
                type="file"
                accept=".jsonl,.json,.txt,application/json"
                disabled={busy}
                onChange={(e) => void readFile(e.target.files?.[0])}
                className="text-sm text-ink"
              />
            </Field>
            <Field label="…or paste JSONL">
              <textarea
                value={judgements}
                maxLength={MAX_BYTES}
                disabled={busy}
                spellCheck={false}
                onChange={(e) => {
                  setJudgements(e.target.value)
                  setFileName(null)
                }}
                className="h-32 rounded-md border border-line bg-page p-2 font-mono text-xs text-ink"
              />
            </Field>
            <div className="flex flex-wrap items-end gap-3">
              <Field label="k">
                <NumberInput value={k} onChange={setK} min={1} max={info.demo_limits?.max_sweep_k ?? 1000} />
              </Field>
              <Field label="Rerank candidates (0 = off)">
                <NumberInput
                  value={candidates}
                  onChange={setCandidates}
                  min={0}
                  max={info.demo_limits?.max_k ?? 10_000}
                />
              </Field>
              {(ivf || hnsw) && (
                <Field label={ivf ? 'nprobe' : 'efSearch'}>
                  <NumberInput
                    value={breadth}
                    onChange={setBreadth}
                    min={1}
                    max={ivf ? Number(info.params.nlist) : (info.demo_limits?.max_ef_search ?? 1 << 16)}
                  />
                </Field>
              )}
              <button
                type="button"
                className={BUTTON}
                disabled={busy || !judgements.trim()}
                onClick={run}
              >
                Evaluate
              </button>
              {job?.status === 'running' && (
                <button type="button" className={BUTTON} onClick={() => cancel.mutate(job.job_id)}>
                  Cancel
                </button>
              )}
            </div>
          </div>
        </Card>
      )}

      {formError && <Banner tone="error">{formError}</Banner>}
      <JobStatus
        noun="Evaluation"
        job={job}
        missing={missing}
        errors={[start.error, cancel.error, missing ? null : poll.error]}
        onRunAgain={run}
        // Judgements live only in this page: after a reload there is nothing to rerun.
        running={busy || !judgements.trim()}
      />
      {job?.status === 'done' && job.result && <Results result={job.result} />}
    </div>
  )
}

function Results({ result }: { result: EvaluationResultOut }) {
  const [order, setOrder] = useState<Order>('worst')
  const [page, setPage] = useState(0)
  const reranked = result.reranked_metrics
  const rows = sortRows(result.queries, order)
  const pages = Math.max(1, Math.ceil(rows.length / PAGE))
  const knob = result.params.nprobe !== undefined ? 'nprobe' : result.params.efSearch !== undefined ? 'ef' : null
  const knobValue = result.params.nprobe ?? result.params.efSearch
  const settings = Object.entries(result.params)
    .map(([key, v]) => `${key} ${v}`)
    .join(', ')

  const csvRows = result.queries.map((q) => ({
    row: q.row,
    recall: q.metrics.recall,
    mrr: q.metrics.mrr,
    ndcg: q.metrics.ndcg,
    reranked_recall: q.reranked_metrics?.recall ?? null,
    reranked_mrr: q.reranked_metrics?.mrr ?? null,
    reranked_ndcg: q.reranked_metrics?.ndcg ?? null,
    ids: q.ids.join(' '),
    reranked_ids: q.reranked_ids?.join(' ') ?? null,
  }))

  return (
    <>
      <Card
        title={`Labelled relevance @${result.k}`}
        subtitle={`${result.n_queries.toLocaleString()} queries · ${result.metric}${settings ? ` · ${settings}` : ''}${
          result.candidates !== null ? ` · reranking ${result.candidates} candidates exactly` : ''
        }`}
        actions={
          <div className="flex gap-2">
            <button type="button" className={BUTTON} onClick={() => downloadCSV('faissight-relevance.csv', csvRows)}>
              CSV
            </button>
            <button type="button" className={BUTTON} onClick={() => downloadJSON('faissight-relevance.json', result)}>
              JSON
            </button>
          </div>
        }
      >
        <div className="grid gap-3 sm:grid-cols-3">
          {METRICS.map(([key, label]) => {
            const base = result.metrics[key]
            const after = reranked?.[key]
            return (
              <StatTile
                key={key}
                label={`${label}@${result.k}`}
                value={base.toFixed(3)}
                caption={
                  after !== undefined ? `${after.toFixed(3)} after reranking (${signed(after - base)})` : undefined
                }
                tone={after === undefined || Math.abs(after - base) < 5e-4 ? 'default' : after > base ? 'good' : 'bad'}
              />
            )
          })}
        </div>
        <p className="mt-3 text-xs text-muted">
          Macro averages over queries. nDCG uses linear gains; unjudged chunks count as non-relevant, so incomplete
          judgements understate relevance. Reranking can only reorder what the index returned as candidates.
        </p>
      </Card>

      <Card
        title="Per query"
        subtitle="Explain opens the query in the Query Explorer with these settings"
        actions={
          <Segmented<Order>
            label="Order queries"
            value={order}
            options={[
              { value: 'worst', label: 'Worst first' },
              ...(reranked ? [{ value: 'gain' as const, label: 'Most helped by reranking' }] : []),
              { value: 'row', label: 'By row' },
            ]}
            onChange={(o) => {
              setOrder(o)
              setPage(0)
            }}
          />
        }
      >
        <table className="w-full text-sm">
          <caption className="sr-only">Relevance per evaluation query</caption>
          <thead className="text-left text-xs text-muted">
            <tr>
              <th scope="col" className="py-1 pr-3 font-normal">
                Query row
              </th>
              {METRICS.map(([key, label]) => (
                <th key={key} scope="col" className="py-1 pr-3 text-right font-normal">
                  {label}
                </th>
              ))}
              {reranked && (
                <th scope="col" className="py-1 pr-3 text-right font-normal">
                  nDCG after rerank
                </th>
              )}
              <th scope="col" className="py-1 font-normal">
                <span className="sr-only">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(page * PAGE, (page + 1) * PAGE).map((q) => (
              <tr key={q.row} className="border-t border-line">
                <th scope="row" className="tabular py-1.5 pr-3 text-left font-normal text-ink">
                  #{q.row}
                </th>
                {METRICS.map(([key]) => (
                  <td key={key} className="tabular py-1.5 pr-3 text-right text-ink">
                    {q.metrics[key].toFixed(3)}
                  </td>
                ))}
                {reranked && (
                  <td className="tabular py-1.5 pr-3 text-right text-ink-2">
                    {q.reranked_metrics
                      ? `${q.reranked_metrics.ndcg.toFixed(3)} (${signed(q.reranked_metrics.ndcg - q.metrics.ndcg)})`
                      : '—'}
                  </td>
                )}
                <td className="py-1.5 text-right">
                  <button
                    type="button"
                    aria-label={`Explain query row ${q.row}`}
                    className="rounded-md border border-line px-2 py-0.5 text-xs text-ink-2 hover:text-ink"
                    onClick={() =>
                      navigate('query', {
                        row: q.row,
                        k: result.k,
                        ...(knob && knobValue !== undefined ? { [knob]: knobValue } : {}),
                        ...(result.candidates !== null ? { candidates: result.candidates } : {}),
                      })
                    }
                  >
                    Explain
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {pages > 1 && (
          <nav aria-label="Pages" className="mt-3 flex items-center gap-3 text-sm text-ink-2">
            <button type="button" className={BUTTON} disabled={page === 0} onClick={() => setPage(page - 1)}>
              Previous
            </button>
            <span>
              Page {page + 1} of {pages}
            </span>
            <button
              type="button"
              className={BUTTON}
              disabled={page + 1 >= pages}
              onClick={() => setPage(page + 1)}
            >
              Next
            </button>
          </nav>
        )}
      </Card>
    </>
  )
}
