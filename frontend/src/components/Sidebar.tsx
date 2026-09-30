import type { Info } from '../api/types'
import type { View } from '../lib/route'
import type { ThemePref } from '../lib/theme'
import { Segmented } from './ui'
import { fmtNum } from '../lib/format'

const PARAM_LABELS: Record<string, string> = {
  nlist: 'nlist',
  nprobe: 'nprobe',
  by_residual: 'by residual',
  pq_m: 'PQ m',
  pq_nbits: 'PQ nbits',
  sq_type: 'SQ type',
  hnsw_m: 'HNSW M',
  ef_search: 'efSearch',
  ef_construction: 'efConstruction',
  max_level: 'max level',
  entry_point: 'entry point',
}

export function Sidebar({
  info,
  view,
  onNavigate,
  themePref,
  onTheme,
  onFocus,
}: {
  info: Info | undefined
  view: View
  onNavigate: (v: View) => void
  themePref: ThemePref
  onTheme: (p: ThemePref) => void
  onFocus: () => void
}) {
  const supported = info?.supported ?? false
  const nav: { view: View; label: string; hint: string; enabled: boolean }[] = [
    { view: 'overview', label: 'Overview', hint: 'Index health', enabled: true },
    { view: 'map', label: 'Cluster map', hint: 'Projected vectors', enabled: supported },
    { view: 'query', label: 'Query explorer', hint: 'Why did it miss?', enabled: supported },
    {
      view: 'hnsw',
      label: 'HNSW graph',
      hint: 'Layers and search trace',
      enabled: supported && (info?.kind === 'HNSW_FLAT' || info?.kind === 'HNSW_OTHER'),
    },
    {
      view: 'quantization',
      label: 'Quantization',
      hint: 'Compression error',
      enabled:
        supported &&
        (['IVF_PQ', 'IVF_SQ', 'HNSW_OTHER'].includes(info?.kind ?? '') ||
          (info?.transforms.some((t) => t.d_out < t.d_in) ?? false)),
    },
    {
      view: 'tuner',
      label: 'Tuner',
      hint: 'Recall vs latency',
      enabled: supported && info?.sweep != null,
    },
    { view: 'compare', label: 'Compare', hint: 'Choose between indexes', enabled: supported },
  ]
  return (
    <aside className="flex w-64 shrink-0 flex-col gap-4 overflow-auto border-r border-line bg-surface p-4 print:hidden">
      <div>
        <div className="text-base font-semibold tracking-tight text-ink">faissight</div>
        <div className="text-xs text-muted">see inside your FAISS index</div>
      </div>

      {info && (
        <div className="rounded-lg border border-line p-3 text-xs">
          <div className="truncate font-medium text-ink" title={info.name}>
            {info.name}
          </div>
          <div className="mt-1 flex flex-wrap gap-1">
            <span className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-ink">
              {info.kind}
            </span>
            {info.demo_limits && (
              <span
                className="rounded bg-accent-wash px-1.5 py-0.5 text-[11px] text-ink"
                title="Public demo: k, sweeps and efSearch are capped; UMAP only if precomputed"
              >
                Read-only demo
              </span>
            )}
          </div>
          <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-ink-2">
            <dt>vectors</dt>
            <dd className="tabular text-right text-ink">{fmtNum(info.ntotal)}</dd>
            <dt>dim</dt>
            <dd className="tabular text-right text-ink">
              {info.d}
              {info.core_d !== info.d && ` → ${info.core_d}`}
            </dd>
            <dt>metric</dt>
            <dd className="text-right text-ink">{info.metric}</dd>
            {Object.entries(info.params)
              .filter(([k]) => k !== 'entry_point' && k !== 'by_residual')
              .map(([k, v]) => (
                <div key={k} className="contents">
                  <dt>{PARAM_LABELS[k] ?? k}</dt>
                  <dd className="tabular text-right text-ink">{String(v)}</dd>
                </div>
              ))}
          </dl>
          {info.class_chain.length > 1 && (
            <div className="mt-2 border-t border-line pt-2 text-[11px] text-muted">
              {info.class_chain.join(' → ')}
            </div>
          )}
        </div>
      )}

      <nav aria-label="Views" className="flex flex-col gap-1">
        {nav.map((n) => (
          <button
            key={n.view}
            disabled={!n.enabled}
            onClick={() => onNavigate(n.view)}
            aria-current={view === n.view ? 'page' : undefined}
            title={n.enabled ? undefined : 'Not available for this index'}
            className={`rounded-lg px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
              view === n.view ? 'bg-accent-wash text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink'
            }`}
          >
            <div className="text-sm font-medium">{n.label}</div>
            <div className="text-xs text-muted">{n.hint}</div>
          </button>
        ))}
      </nav>

      <div className="mt-auto flex flex-col gap-2">
        <Segmented
          label="Theme"
          value={themePref}
          onChange={onTheme}
          options={[
            { value: 'light', label: 'Light' },
            { value: 'dark', label: 'Dark' },
            { value: 'system', label: 'Auto' },
          ]}
        />
        <div className="flex items-center justify-between text-[11px] text-muted">
          {info && <span>faissight {info.version}</span>}
          <button onClick={onFocus} className="hover:text-ink" title="Hide the sidebar (\)">
            Hide sidebar
          </button>
        </div>
      </div>
    </aside>
  )
}
