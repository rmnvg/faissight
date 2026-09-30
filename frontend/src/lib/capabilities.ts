import type { Info } from '../api/types'

/** True when the index compresses vectors (PQ/SQ codes or a dimension-reducing transform),
 * which is what the Quantization view analyses. */
export function hasQuantization(info: Pick<Info, 'kind' | 'transforms'>): boolean {
  return (
    ['IVF_PQ', 'IVF_SQ', 'HNSW_OTHER'].includes(info.kind) || info.transforms.some((t) => t.d_out < t.d_in)
  )
}
