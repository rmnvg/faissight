import { describe, expect, it } from 'vitest'
import { hasQuantization } from './capabilities'

describe('hasQuantization', () => {
  it('covers PQ/SQ codes and dimension-reducing transforms only', () => {
    expect(hasQuantization({ kind: 'IVF_PQ', transforms: [] })).toBe(true)
    expect(hasQuantization({ kind: 'IVF_FLAT', transforms: [] })).toBe(false)
    expect(hasQuantization({ kind: 'IVF_FLAT', transforms: [{ name: 'PCA', d_in: 64, d_out: 16 }] })).toBe(true)
    expect(hasQuantization({ kind: 'HNSW_FLAT', transforms: [{ name: 'OPQ', d_in: 64, d_out: 64 }] })).toBe(false)
  })
})
