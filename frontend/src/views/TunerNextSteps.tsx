import { useSweepAdvice } from '../api/hooks'
import type { SweepParam } from '../api/types'
import { Card, Spinner } from '../components/ui'
import { navigate } from '../lib/route'
import { suggestionLink } from '../lib/tuner'

const BUTTON =
  'rounded-md border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink hover:bg-accent-wash disabled:opacity-50'

/** What to change next, each step with the sweep measurements that support it. */
export function NextSteps({
  jobId,
  param,
  k,
  target,
  confident,
  targetMet,
  onSweep,
  sweeping,
}: {
  jobId: string
  param: SweepParam
  k: number
  target: number
  confident: boolean
  targetMet: boolean
  /** Start a follow-up sweep of the same parameter with these values. */
  onSweep: (values: number[]) => void
  sweeping: boolean
}) {
  const advice = useSweepAdvice(jobId, target, confident)
  const suggestions = advice.data?.suggestions
  return (
    <Card
      title="Suggested next steps"
      subtitle={`From this sweep's measurements, for a target recall@${k} of ${target.toFixed(2)}`}
    >
      {advice.isPending ? (
        <Spinner label="Reading the measurements" />
      ) : advice.isError ? (
        <p className="text-sm text-critical">{advice.error.message}</p>
      ) : suggestions && suggestions.length > 0 ? (
        <ol className={`flex flex-col gap-4 ${advice.isPlaceholderData ? 'opacity-60' : ''}`}>
          {suggestions.map((s, i) => {
            const link = suggestionLink(s, param, k)
            return (
              <li key={s.kind} className="flex gap-3">
                <span className="tabular mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-wash text-xs font-semibold text-ink">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-medium text-ink">{s.title}</h3>
                  <p className="mt-1 text-sm text-ink-2">{s.detail}</p>
                  {s.evidence.length > 0 && (
                    <dl className="mt-2 flex flex-wrap gap-2">
                      {s.evidence.map((e) => (
                        <div key={e.label} className="rounded-md bg-surface-2 px-2 py-1 text-xs">
                          <dt className="inline text-muted">{e.label}: </dt>
                          <dd className="tabular inline font-medium text-ink">{e.value}</dd>
                        </div>
                      ))}
                    </dl>
                  )}
                  {(s.sweep_values || link) && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {s.sweep_values && (
                        <button
                          type="button"
                          className={BUTTON}
                          disabled={sweeping}
                          onClick={() => onSweep(s.sweep_values as number[])}
                        >
                          Sweep {param} {s.sweep_values.join(', ')}
                        </button>
                      )}
                      {link && (
                        <button type="button" className={BUTTON} onClick={() => navigate(link.view, link.params)}>
                          {link.label}
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </li>
            )
          })}
        </ol>
      ) : targetMet ? (
        <p className="text-sm text-ink-2">
          Nothing to change: the target is met without a tail of failing queries
          {param === 'nprobe' ? ' or uneven lists' : ''}.
        </p>
      ) : (
        <p className="text-sm text-ink-2">
          These measurements don't point to a single cause. Open the worst queries below to see
          why their neighbours were missed.
        </p>
      )}
    </Card>
  )
}
