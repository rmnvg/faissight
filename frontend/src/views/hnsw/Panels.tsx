import { useState } from 'react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { HnswOutcome, HnswStats, HnswTrace } from '../../api/types'
import { Banner, Card, SnippetText, StatTile } from '../../components/ui'
import { fmtDist, fmtNum } from '../../lib/format'
import { LEVEL0_LIMIT } from './constants'

const OUTCOME: Record<HnswOutcome, { label: string; dot: string; help: string }> = {
  FOUND: { label: 'Found', dot: 'bg-good', help: 'Returned by the search.' },
  NOT_REACHED: {
    label: 'Not reached',
    dot: 'bg-critical',
    help: 'Not reached from the visited frontier: no expanded node linked to it before the search stopped. Raise efSearch.',
  },
  VISITED_NOT_KEPT: {
    label: 'Visited, ranked out',
    dot: 'bg-serious',
    help: 'Its distance was computed but approximate (SQ/PQ) distances ranked it out of the top-k.',
  },
}

export function TracePanel({ trace }: { trace: HnswTrace }) {
  const perLevel = trace.levels.map((l) => ({
    level: l.level,
    expansions: l.steps.length,
    distances: l.steps.reduce((a, s) => a + s.visits.length, 0),
  }))
  const totalDist = perLevel.reduce((a, l) => a + l.distances, 0)
  const misses = (trace.truth ?? []).filter((t) => t.outcome !== 'FOUND')
  return (
    <>
      <Banner>
        <strong>Reconstructed trace.</strong> Replayed in Python over the graph; it returns{' '}
        {Math.round(trace.overlap_with_faiss * 100)}% of the ids FAISS itself returned.
      </Banner>
      <div className="grid grid-cols-2 gap-2">
        <StatTile
          label={`Recall@${trace.k}`}
          value={trace.recall === null ? '—' : trace.recall.toFixed(2)}
          tone={trace.recall === null ? 'default' : trace.recall >= 1 ? 'good' : trace.recall < 0.9 ? 'bad' : 'default'}
        />
        <StatTile label="Distance computations" value={fmtNum(totalDist)} caption={`efSearch ${trace.ef_search}`} />
      </div>
      <Card title="Work per level">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr>
              <th className="py-1 font-normal">Level</th>
              <th className="py-1 text-right font-normal">Nodes expanded</th>
              <th className="py-1 text-right font-normal">Distances</th>
            </tr>
          </thead>
          <tbody>
            {perLevel.map((l) => (
              <tr key={l.level} className="border-t border-line">
                <td className="tabular py-1 text-ink">{l.level}</td>
                <td className="tabular py-1 text-right text-ink">{l.expansions}</td>
                <td className="tabular py-1 text-right text-ink-2">{l.distances}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      {trace.truth && (
        <Card title="Exact nearest neighbours" subtitle={misses.length ? `${misses.length} missed` : 'All found'}>
          <ul className="flex flex-col divide-y divide-line text-sm">
            {trace.truth.map((t) => {
              const o = OUTCOME[t.outcome]
              return (
                <li key={t.id} className="flex items-start gap-2 py-1.5" title={o.help}>
                  <span className="tabular w-6 text-muted">{t.rank + 1}</span>
                  <span className="tabular w-14 text-ink">{t.id}</span>
                  <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-line px-2 py-0.5 text-xs text-ink">
                    <span className={`h-2 w-2 rounded-full ${o.dot}`} />
                    {t.outcome === 'FOUND' ? '✓' : '✕'} {o.label}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-xs">
                    <SnippetText snippet={t.snippet} />
                  </span>
                </li>
              )
            })}
          </ul>
          {misses.some((m) => m.outcome === 'NOT_REACHED') && (
            <p className="mt-2 text-xs text-ink-2">
              “Not reached”: no node the search expanded linked to it before the stopping rule
              fired. A larger efSearch widens the frontier.
            </p>
          )}
          <p className="mt-2 text-xs text-muted">
            Distances: {trace.higher_is_closer ? 'inner product (higher = closer)' : 'squared L2'}; nearest{' '}
            {fmtDist(trace.truth[0]?.distance ?? 0)}.
          </p>
        </Card>
      )}
    </>
  )
}

export function StatsPanel({ stats }: { stats: HnswStats }) {
  const [level, setLevel] = useState(0)
  const lv = stats.levels.find((l) => l.level === level) ?? stats.levels[stats.levels.length - 1]
  const hist = lv.degree_hist.map((n, degree) => ({ degree, n }))
  return (
    <Card title="Graph structure" subtitle={`M = ${stats.m} · efConstruction ${stats.ef_construction}`}>
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-muted">
          <tr>
            <th className="py-1 font-normal">Level</th>
            <th className="py-1 text-right font-normal">Nodes</th>
            <th className="py-1 text-right font-normal">Mean links</th>
          </tr>
        </thead>
        <tbody>
          {stats.levels.map((l) => (
            <tr
              key={l.level}
              onClick={() => setLevel(l.level)}
              className={`cursor-pointer border-t border-line hover:bg-surface-2 ${l.level === level ? 'bg-accent-wash' : ''}`}
            >
              <td className="tabular py-1 text-ink">{l.level}</td>
              <td className="tabular py-1 text-right text-ink">{fmtNum(l.n_nodes)}</td>
              <td className="tabular py-1 text-right text-ink-2">
                {l.degree_mean.toFixed(1)} / {l.max_links}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 text-xs text-ink-2">Links per node on level {lv.level} (click a row)</div>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={hist} margin={{ top: 6, right: 4, bottom: 0, left: 0 }} barCategoryGap={2}>
            <XAxis dataKey="degree" tick={{ fill: 'var(--muted)', fontSize: 10 }} tickLine={false} axisLine={{ stroke: 'var(--axis)' }} interval="preserveStartEnd" />
            <YAxis tick={{ fill: 'var(--muted)', fontSize: 10 }} tickLine={false} axisLine={false} width={34} allowDecimals={false} />
            <Tooltip
              cursor={{ fill: 'var(--accent-wash)' }}
              content={({ active, payload }) =>
                active && payload?.[0] ? (
                  <div className="rounded-lg border border-line bg-surface px-2 py-1 text-xs shadow">
                    {payload[0].payload.degree} links: {fmtNum(payload[0].payload.n)} nodes
                  </div>
                ) : null
              }
            />
            <Bar dataKey="n" fill="var(--series-1)" radius={[3, 3, 0, 0]} maxBarSize={24} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-xs text-muted">
        Entry point id {stats.entry_point} on level {stats.max_level}. Level 0 shows a sample of{' '}
        {fmtNum(LEVEL0_LIMIT)} nodes around where the search entered it.
      </p>
    </Card>
  )
}
