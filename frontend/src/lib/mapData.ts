import type { UserId } from '../api/types'
import { useMemo } from 'react'
import { isReady, useProjection } from '../api/hooks'
import type { Dims, JobStatus, Projection, ProjectionMethod } from '../api/types'
import { packPositions } from './points'

export interface MapData {
  points: Projection
  centroids: Projection | null
  positions: Float32Array
  centroidPositions: Float32Array | null
  /** Projection row of each point id. */
  indexOf: Map<UserId, number>
}

/** Load a projection (points + IVF centroids); `status` reports progress while computing. */
export function useMapData(method: ProjectionMethod, dims: Dims, isIvf: boolean) {
  const pts = useProjection('points', method, dims)
  const cents = useProjection('centroids', method, dims, isIvf && isReady(pts.data))

  const data = useMemo<MapData | null>(() => {
    if (!isReady(pts.data)) return null
    if (isIvf && !isReady(cents.data)) return null
    const points = pts.data
    const centroids = isIvf && isReady(cents.data) ? cents.data : null
    const indexOf = new Map<UserId, number>()
    points.ids.forEach((id, i) => indexOf.set(id, i))
    return {
      points,
      centroids,
      positions: packPositions(points.x, points.y, points.z),
      centroidPositions: centroids ? packPositions(centroids.x, centroids.y, centroids.z) : null,
      indexOf,
    }
  }, [pts.data, cents.data, isIvf])

  const status: JobStatus | null =
    pts.data && pts.data.status !== 'done' ? (pts.data as JobStatus) : null
  return { data, status, error: pts.error ?? cents.error, loading: !data && !pts.error }
}
