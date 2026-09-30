import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api } from '../api/client'
import { useLatestSignal } from '../api/hooks'
import type { Info, UserId } from '../api/types'
import { Field, NumberInput } from '../components/form'
import { Card, EmptyState, Spinner } from '../components/ui'
import { parseId } from '../lib/ids'
import { intParam, navigate } from '../lib/route'
import type { ThemeMode } from '../lib/theme'
import { StatsPanel, TracePanel } from './hnsw/Panels'
import { Scene } from './hnsw/Scene'

export default function Hnsw({
  info,
  params,
  mode,
}: {
  info: Info
  params: URLSearchParams
  mode: ThemeMode
}) {
  const stats = useQuery({ queryKey: ['hnsw-stats'], queryFn: ({ signal }) => api.hnswStats(signal), staleTime: Infinity })
  const urlId = parseId(params.get('id'))
  const [idInput, setIdInput] = useState(urlId !== null ? String(urlId) : '')
  const [k, setK] = useState(10)
  const [ef, setEf] = useState(intParam(params, 'ef') ?? Number(info.params.ef_search ?? 16))
  const [formError, setFormError] = useState<string | null>(null)

  const nextSignal = useLatestSignal()
  const trace = useMutation({
    mutationFn: (req: { id: UserId; k: number; ef: number }) =>
      api.traceHnsw(
        { query: { id: req.id }, k: req.k, efSearch: req.ef, compare: true },
        nextSignal(),
      ),
  })

  const run = (id: UserId) => trace.mutate({ id, k, ef })
  const autoRan = useRef<UserId | null>(null)
  useEffect(() => {
    if (urlId === null || autoRan.current === urlId) return
    autoRan.current = urlId
    trace.mutate({ id: urlId, k, ef })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlId])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const id = parseId(idInput)
    if (id === null) {
      setFormError('Enter an integer id of a stored vector.')
      return
    }
    setFormError(null)
    navigate('hnsw', { id, ef })
    if (urlId === id) run(id)
  }

  const t = trace.data

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-end gap-3 border-b border-line bg-surface px-6 py-3">
        <div className="mr-2">
          <h1 className="text-base font-semibold text-ink">HNSW graph</h1>
          <p className="text-xs text-ink-2">Layers of the graph and a step-by-step search</p>
        </div>
        <form onSubmit={submit} className="flex flex-wrap items-end gap-3">
          <label className="flex w-40 flex-col gap-1 text-xs text-ink-2">
            Query: stored id
            <input
              value={idInput}
              onChange={(e) => setIdInput(e.target.value)}
              inputMode="numeric"
              placeholder="e.g. 42"
              className="rounded-lg border border-line bg-page px-3 py-1.5 text-sm text-ink outline-none focus:border-series-1"
            />
          </label>
          <Field label="k" className="w-24">
            <NumberInput value={k} onChange={setK} min={1} max={1000} size="sm" />
          </Field>
          <Field label="efSearch" className="w-24">
            <NumberInput value={ef} onChange={setEf} min={1} max={4096} size="sm" />
          </Field>
          <button
            type="submit"
            disabled={trace.isPending}
            className="rounded-lg bg-series-1 px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {trace.isPending ? 'Tracing…' : 'Trace search'}
          </button>
        </form>
        {formError && <span className="text-sm text-critical">{formError}</span>}
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1 bg-surface">
          {stats.data ? (
            <Scene stats={stats.data} trace={t ?? null} mode={mode} />
          ) : stats.isError ? (
            <EmptyState title="Graph unavailable">{stats.error.message}</EmptyState>
          ) : (
            <div className="flex h-full items-center justify-center">
              <Spinner label="Reading the graph…" />
            </div>
          )}
        </div>
        <aside className="w-96 shrink-0 overflow-auto border-l border-line bg-page p-3">
          <div className="flex flex-col gap-3">
            {t && <TracePanel trace={t} />}
            {!t && (
              <Card>
                <EmptyState title="Trace a search">
                  Enter a stored id to replay how HNSW descends the layers and expands its
                  candidates on level 0.
                </EmptyState>
              </Card>
            )}
            {stats.data && <StatsPanel stats={stats.data} />}
          </div>
        </aside>
      </div>
    </div>
  )
}
