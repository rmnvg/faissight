import { useEffect, useRef } from 'react'
import type { Info } from '../../api/types'
import { Field, NumberInput } from '../../components/form'
import { Card, Segmented } from '../../components/ui'
import type { QueryRunner } from './useQueryRunner'

/** Query type, query input, k, the index's search parameter, and "compare with exact". */
export function QueryForm({ info, runner }: { info: Info; runner: QueryRunner }) {
  const { isIvf, isHnsw, form: f, submit, run } = runner
  const nlist = Number(info.params.nlist ?? 1)
  const hasEmbedder = info.inputs.embedder !== null
  const embedderReady = info.inputs.embedder_status === 'ready'
  const nEval = info.inputs.queries ?? 0
  const inputRef = useRef<HTMLInputElement & HTMLTextAreaElement>(null)

  // "/" focuses the query box.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <Card>
      <form onSubmit={submit} className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <Segmented
            label="Query type"
            value={f.queryMode}
            onChange={f.setQueryMode}
            options={[
              {
                value: 'text',
                label: 'Text',
                disabled: !hasEmbedder,
                title: hasEmbedder ? undefined : 'Start faissight with --embedder to query by text',
              },
              { value: 'id', label: 'Stored id' },
              {
                value: 'row',
                label: 'Evaluation query',
                disabled: nEval === 0,
                title:
                  nEval > 0
                    ? `A row of the ${nEval} queries given with --queries, as sweeps number them`
                    : 'Start faissight with --queries to explain held-out queries',
              },
              { value: 'vector', label: 'Vector' },
            ]}
          />
          {f.queryMode === 'text' && hasEmbedder && !embedderReady && (
            <span className="text-xs text-muted">
              {info.inputs.embedder_status === 'failed'
                ? 'Embedder failed to load; see the server log.'
                : `Loading ${info.inputs.embedder}… the first query waits for it.`}
            </span>
          )}
          <span className="ml-auto text-xs text-muted">
            Press <kbd className="rounded border border-line px-1">/</kbd> to focus
          </span>
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <label className="flex min-w-64 flex-1 flex-col gap-1 text-xs text-ink-2">
            {f.queryMode === 'text'
              ? 'Query text'
              : f.queryMode === 'id'
                ? 'Stored vector id'
                : f.queryMode === 'row'
                  ? `Evaluation query row (0–${nEval - 1})`
                  : `Vector (${info.d} numbers)`}
            {f.queryMode === 'vector' ? (
              <textarea
                ref={inputRef}
                value={f.vectorInput}
                onChange={(e) => f.setVectorInput(e.target.value)}
                rows={2}
                placeholder="0.12, -0.5, …"
                className="rounded-lg border border-line bg-page px-3 py-2 font-mono text-sm text-ink outline-none focus:border-series-1"
              />
            ) : (
              <input
                ref={inputRef}
                value={f.queryMode === 'text' ? f.text : f.queryMode === 'row' ? f.rowInput : f.idInput}
                onChange={(e) =>
                  f.queryMode === 'text'
                    ? f.setText(e.target.value)
                    : f.queryMode === 'row'
                      ? f.setRowInput(e.target.value)
                      : f.setIdInput(e.target.value)
                }
                inputMode={f.queryMode === 'id' || f.queryMode === 'row' ? 'numeric' : undefined}
                placeholder={f.queryMode === 'text' ? 'what is inverted file indexing?' : f.queryMode === 'row' ? 'e.g. 0' : 'e.g. 42'}
                className="rounded-lg border border-line bg-page px-3 py-2 text-sm text-ink outline-none focus:border-series-1"
              />
            )}
          </label>
          <Field label="k" className="w-24">
            <NumberInput value={f.k} onChange={f.setK} min={1} max={info.demo_limits?.max_k ?? 1000} />
          </Field>
          {isIvf && (
            <label className="flex w-56 flex-col gap-1 text-xs text-ink-2">
              <span>
                nprobe <span className="tabular text-ink">{f.nprobe}</span> / {nlist}
              </span>
              <input
                type="range"
                min={1}
                max={nlist}
                value={f.nprobe}
                onChange={(e) => f.setNprobe(Number(e.target.value))}
                className="accent-series-1"
              />
            </label>
          )}
          {isHnsw && (
            <Field label="efSearch" className="w-24">
              <NumberInput
                value={f.efSearch}
                onChange={f.setEfSearch}
                min={1}
                max={info.demo_limits?.max_ef_search ?? 4096}
              />
            </Field>
          )}
          <label className="flex items-center gap-2 pb-2 text-sm text-ink-2">
            <input
              type="checkbox"
              checked={f.compare}
              onChange={(e) => f.setCompare(e.target.checked)}
              className="accent-series-1"
            />
            Compare with exact
          </label>
          {info.inputs.raw_vectors && (
            <Field label="Rerank candidates (0 = off)" className="w-48">
              <NumberInput value={f.candidates} onChange={f.setCandidates} min={0}
                max={info.demo_limits?.max_k ?? 10000} />
            </Field>
          )}
          <button
            type="submit"
            disabled={run.isPending}
            className="rounded-lg bg-series-1 px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {run.isPending ? 'Searching…' : 'Search'}
          </button>
        </div>
        {f.formError && <p className="text-sm text-critical">{f.formError}</p>}
      </form>
    </Card>
  )
}
