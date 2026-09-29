import { useQuery } from '@tanstack/react-query'
import { api } from './client'
import type { Dims, Projection, ProjectionMethod } from './types'

export function useInfo() {
  return useQuery({
    queryKey: ['info'],
    queryFn: api.info,
    staleTime: Infinity,
    // Poll while the embedder model is still loading so the text tab unlocks by itself.
    refetchInterval: (q) => (q.state.data?.inputs.embedder_status === 'loading' ? 2000 : false),
  })
}

export function useIvfLists(enabled: boolean) {
  return useQuery({ queryKey: ['ivf-lists'], queryFn: api.ivfLists, staleTime: Infinity, enabled })
}

export function useListMembers(listNo: number | null, offset: number, limit: number) {
  return useQuery({
    queryKey: ['list-members', listNo, offset, limit],
    queryFn: () => api.listMembers(listNo as number, offset, limit),
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
    queryFn: () => api.projection(kind, method, dims),
    enabled,
    staleTime: Infinity,
    retry: false,
    refetchInterval: (q) => (q.state.data && q.state.data.status === 'running' ? 500 : false),
  })
}

export function isReady(p: { status: string } | undefined): p is Projection {
  return p?.status === 'done'
}

export function useMetadata(id: number | null, enabled: boolean) {
  return useQuery({
    queryKey: ['metadata', id],
    queryFn: () => api.metadata(id as number),
    enabled: enabled && id !== null,
    staleTime: Infinity,
    retry: false,
  })
}
