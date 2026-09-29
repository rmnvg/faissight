import type { UserId } from '../api/types'

/** Parse an unsigned decimal int64 id without ever rounding it. */
export function parseId(value: string | null): UserId | null {
  const text = value?.trim() ?? ''
  if (!/^\d+$/.test(text)) return null
  const id = BigInt(text)
  if (id > 9223372036854775807n) return null
  return id <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(id) : id.toString()
}
