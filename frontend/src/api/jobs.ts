import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError } from './client'

/** The status fields every background job response shares (sweeps, comparisons). */
export interface JobSnapshot {
  job_id: string
  status: 'running' | 'done' | 'failed' | 'cancelled'
  progress: number
  message: string
  error: string | null
}

/**
 * Start, poll and cancel one kind of background job, with the job id kept in the URL.
 *
 * Job ids are derived from the settings, so restarting a failed or cancelled job with the
 * same settings reuses its id (and URL). Start and cancel responses are therefore written
 * straight into the query cache: otherwise the cached terminal status would stay on screen
 * and polling would never resume.
 */
export function useJob<Req, J extends JobSnapshot>(opts: {
  kind: string
  jobId: string | null
  start: (req: Req) => Promise<J>
  fetch: (jobId: string, signal?: AbortSignal) => Promise<J>
  cancel: (jobId: string) => Promise<J>
  /** Called after a start succeeds, e.g. to put the job id and settings in the URL. */
  onStarted: (job: J, req: Req) => void
}) {
  const client = useQueryClient()
  const { kind, jobId } = opts
  const poll = useQuery({
    queryKey: [kind, jobId],
    queryFn: ({ signal }) => opts.fetch(jobId as string, signal),
    enabled: jobId !== null,
    staleTime: Infinity,
    retry: false,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 400 : false),
  })
  const start = useMutation({
    mutationFn: opts.start,
    onSuccess: (job, req) => {
      client.setQueryData([kind, job.job_id], job)
      opts.onStarted(job, req)
    },
  })
  const cancel = useMutation({
    mutationFn: opts.cancel,
    onSuccess: (job) => client.setQueryData([kind, job.job_id], job),
  })
  // A job the server no longer has: it expired, was evicted, or the server restarted.
  // React Query keeps the last data after an error: don't show a stale "running" job.
  const missing = poll.error instanceof ApiError && poll.error.status === 404
  return { job: missing ? undefined : poll.data, poll, start, cancel, missing }
}
