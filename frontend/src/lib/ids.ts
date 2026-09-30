import type { UserId } from '../api/types'

/**
 * URL params that open a sweep/comparison query in the Query Explorer: its stored id for
 * sampled queries, else its row in the --queries set (held-out queries have no id).
 */
export function queryRefParams(q: { id: UserId | null; query_no: number }): Record<string, string | number> {
  return q.id !== null ? { id: String(q.id) } : { row: q.query_no }
}

/** Parse an unsigned decimal int64 id without ever rounding it. */
export function parseId(value: string | null): UserId | null {
  const text = value?.trim() ?? ''
  if (!/^\d+$/.test(text)) return null
  const id = BigInt(text)
  if (id > 9223372036854775807n) return null
  return id <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(id) : id.toString()
}
