import type { JobSnapshot } from '../api/jobs'
import { Banner, Card, Progress } from './ui'

/** Progress, failure, cancellation and "no longer on the server" states of a job. */
export function JobStatus({
  noun,
  job,
  missing,
  errors,
  onRunAgain,
  running,
}: {
  /** "Sweep", "Comparison", … */
  noun: string
  job: JobSnapshot | undefined
  missing: boolean
  /** Request errors to show (start, cancel, polling other than "missing"). */
  errors: (Error | null)[]
  /** Restart with the settings in the URL/form; shown for failed, cancelled and missing jobs. */
  onRunAgain: () => void
  /** True while a start request is in flight. */
  running: boolean
}) {
  const again = (
    <button
      type="button"
      disabled={running}
      onClick={onRunAgain}
      className="ml-2 rounded-md border border-line bg-surface px-2 py-0.5 text-xs font-medium text-ink hover:bg-accent-wash disabled:opacity-50"
    >
      Run again
    </button>
  )
  return (
    <>
      {job?.status === 'running' && (
        <Card>
          <div className="flex justify-center py-6">
            <Progress value={job.progress} message={job.message} />
          </div>
        </Card>
      )}
      {job?.status === 'failed' && (
        <Banner tone="error">
          {noun} failed: {job.error}
          {again}
        </Banner>
      )}
      {job?.status === 'cancelled' && (
        <Banner>
          {noun} cancelled.
          {again}
        </Banner>
      )}
      {missing && (
        <Banner tone="warning">
          This {noun.toLowerCase()} is no longer on the server: finished results are kept for an hour, and a
          restarted server starts empty.
          {again}
        </Banner>
      )}
      {errors.map((e, i) => e && <Banner key={i} tone="error">{e.message}</Banner>)}
    </>
  )
}
