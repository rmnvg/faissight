import { describe, expect, it } from 'vitest'
import { parseId } from './ids'
import { toQueryString } from './route'
import { toCSV } from './exportData'

describe('lossless ids', () => {
  it('preserves adjacent int64 ids through input, links and exports', () => {
    const ids = ['9007199254740992', '9007199254740993', '9223372036854775807']
    for (const id of ids) {
      expect(parseId(id)).toBe(id)
      expect(parseId(new URLSearchParams(toQueryString({ id })).get('id'))).toBe(id)
      expect(JSON.parse(JSON.stringify({ id })).id).toBe(id)
      expect(toCSV([{ id }])).toContain(id)
    }
    expect(new Set(ids.map(parseId)).size).toBe(3)
  })
  it('normalizes small ids and rejects invalid or out-of-range ids', () => {
    expect(parseId(' 00042 ')).toBe(42)
    expect(parseId('9007199254740991')).toBe(Number.MAX_SAFE_INTEGER)
    for (const text of ['', ' ', '-1', '1.5', '1e3', '0xff', '9223372036854775808', null]) {
      expect(parseId(text)).toBeNull()
    }
  })
})
