import type { ReactNode } from 'react'
import type { MissReason } from '../api/types'

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={`rounded-xl border border-line bg-surface p-4 shadow-[0_1px_2px_rgba(0,0,0,0.04)] ${className}`}
    >
      {(title || actions) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-2">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}

/** Label, value, optional caption. Proportional figures for the value (not tabular). */
export function StatTile({
  label,
  value,
  caption,
  tone = 'default',
}: {
  label: string
  value: ReactNode
  caption?: ReactNode
  tone?: 'default' | 'good' | 'bad'
}) {
  const valueColor = tone === 'good' ? 'text-good-text' : tone === 'bad' ? 'text-critical' : 'text-ink'
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3">
      <div className="text-xs text-ink-2">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${valueColor}`}>{value}</div>
      {caption && <div className="mt-0.5 text-xs text-muted">{caption}</div>}
    </div>
  )
}

export function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T
  options: { value: T; label: string; disabled?: boolean; title?: string }[]
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === value}
          disabled={o.disabled}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
            o.value === value
              ? 'bg-surface text-ink shadow-sm'
              : 'text-ink-2 hover:text-ink'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Banner({
  tone = 'info',
  children,
}: {
  tone?: 'info' | 'warning' | 'error'
  children: ReactNode
}) {
  const styles = {
    info: 'border-series-1/30 bg-accent-wash',
    warning: 'border-warning/50 bg-warning/10',
    error: 'border-critical/40 bg-critical/10',
  }[tone]
  const icon = { info: 'ℹ', warning: '⚠', error: '✕' }[tone]
  return (
    <div role="status" className={`flex gap-2 rounded-lg border px-3 py-2 text-sm text-ink ${styles}`}>
      <span aria-hidden className="text-ink-2">
        {icon}
      </span>
      <div>{children}</div>
    </div>
  )
}

export function Progress({ value, message }: { value: number; message: string }) {
  return (
    <div className="w-72 max-w-full" role="progressbar" aria-valuenow={Math.round(value * 100)}>
      <div className="mb-1.5 flex justify-between text-xs text-ink-2">
        <span>{message || 'Working…'}</span>
        <span className="tabular">{Math.round(value * 100)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
        <div
          className="h-full rounded-full bg-series-1 transition-[width] duration-300"
          style={{ width: `${Math.max(4, value * 100)}%` }}
        />
      </div>
    </div>
  )
}

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-ink-2" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-grid border-t-series-1" />
      {label}
    </div>
  )
}

/** Restarts failed background work (projections, analyses). */
export function RetryButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="mt-2 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm font-medium text-ink hover:bg-accent-wash"
    >
      Try again
    </button>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex h-full min-h-40 flex-col items-center justify-center gap-1 p-6 text-center">
      <div className="text-sm font-medium text-ink">{title}</div>
      {children && <div className="max-w-md text-sm text-ink-2">{children}</div>}
    </div>
  )
}

const REASONS: Record<MissReason, { label: string; icon: string; dot: string; help: string }> = {
  FOUND: {
    label: 'Found',
    icon: '✓',
    dot: 'bg-good',
    help: 'Returned by the approximate search.',
  },
  CELL_NOT_PROBED: {
    label: 'Cell not probed',
    icon: '✕',
    dot: 'bg-critical',
    help: 'Its inverted list ranked beyond nprobe, so it was never scanned.',
  },
  QUANTIZATION: {
    label: 'Quantization',
    icon: '≈',
    dot: 'bg-serious',
    help: 'Its list was scanned, but PQ/SQ approximate distances pushed it out of the top-k.',
  },
  TRANSFORM: {
    label: 'Transform',
    icon: '↘',
    dot: 'bg-serious',
    help: 'Its list was scanned, but the dimension-reducing PreTransform changed the ranking.',
  },
  RANKED_OUT: {
    label: 'Ranked out',
    icon: '=',
    dot: 'bg-warning',
    help: 'Its list was scanned with exact codes but it still missed the top-k (ties).',
  },
}

/** Status dot + icon + label, never colour alone. */
export function ReasonBadge({ reason }: { reason: MissReason | null }) {
  if (!reason) return <span className="text-muted">—</span>
  const r = REASONS[reason]
  return (
    <span
      title={r.help}
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-line px-2 py-0.5 text-xs text-ink"
    >
      <span className={`h-2 w-2 rounded-full ${r.dot}`} aria-hidden />
      <span aria-hidden className="text-ink-2">
        {r.icon}
      </span>
      {r.label}
    </span>
  )
}

export function SnippetText({ snippet }: { snippet: { title?: string; text?: string } | null }) {
  if (!snippet || (!snippet.title && !snippet.text)) return <span className="text-muted">—</span>
  return (
    <span className="line-clamp-2 text-ink-2">
      {snippet.title && <span className="font-medium text-ink">{snippet.title} · </span>}
      {snippet.text}
    </span>
  )
}
