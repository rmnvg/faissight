import type { UserId } from './types'
import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useRef } from 'react'
import { api } from './client'
import type { Dims, Projection, ProjectionMethod } from './types'

/**
 * Signals for one-at-a-time requests (mutations): each call aborts the previous request,
 * and unmounting aborts the last, so a slow stale search can't overwrite a newer one.
 */
export function useLatestSignal(): () => AbortSignal {
  const current = useRef<AbortController | null>(null)
  useEffect(() => () => current.current?.abort(), [])
  return useCallback(() => {
    current.current?.abort()
    current.current = new AbortController()
    return current.current.signal
  }, [])
}

export function useInfo() {
  return useQuery({
    queryKey: ['info'],
    queryFn: ({ signal }) => api.info(signal),
    staleTime: Infinity,
    // Poll while the embedder model is still loading so the text tab unlocks by itself.
    refetchInterval: (q) => (q.state.data?.inputs.embedder_status === 'loading' ? 2000 : false),
  })
}

export function useIvfLists(enabled: boolean) {
  return useQuery({ queryKey: ['ivf-lists'], queryFn: ({ signal }) => api.ivfLists(signal), staleTime: Infinity, enabled })
}

export function useListMembers(listNo: number | null, offset: number, limit: number) {
  return useQuery({
    queryKey: ['list-members', listNo, offset, limit],
    queryFn: ({ signal }) => api.listMembers(listNo as number, offset, limit, signal),
    enabled: listNo !== null,
    staleTime: Infinity,
    placeholderData: (prev) => prev,
  })
}

export function useProjection(
  kind: 'points' | 'centroids',
  method: ProjectionMethod,
  dims: Dims,
  enabled = true,
) {
  return useQuery({
    queryKey: ['projection', kind, method, dims],
    queryFn: ({ signal }) => api.projection(kind, method, dims, signal),
    enabled,
    staleTime: Infinity,
    retry: false,
    refetchInterval: (q) => (q.state.data && q.state.data.status === 'running' ? 500 : false),
  })
}

export function isReady(p: { status: string } | undefined): p is Projection {
  return p?.status === 'done'
}

export function useMetadata(id: UserId | null, enabled: boolean) {
  return useQuery({
    queryKey: ['metadata', id],
    queryFn: ({ signal }) => api.metadata(id as number, signal),
    enabled: enabled && id !== null,
    staleTime: Infinity,
    retry: false,
  })
}

/** Poll a sweep job until it finishes. */
export function useSweep(jobId: string | null) {
  return useQuery({
    queryKey: ['sweep', jobId],
    queryFn: ({ signal }) => api.sweep(jobId as string, signal),
    enabled: jobId !== null,
    staleTime: Infinity,
    retry: false,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 400 : false),
  })
}
