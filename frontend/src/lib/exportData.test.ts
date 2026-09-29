import { describe, expect, it } from 'vitest'
import { flatten, toCSV } from './exportData'

describe('toCSV', () => {
  it('writes a header and rows', () => {
    expect(toCSV([{ id: 1, d: 0.5 }, { id: 2, d: 0.25 }])).toBe('id,d\r\n1,0.5\r\n2,0.25\r\n')
  })
  it('quotes separators, quotes and newlines', () => {
    const csv = toCSV([{ text: 'a, "b"\nc' }])
    expect(csv).toBe('text\r\n"a, ""b""\nc"\r\n')
  })
  it('handles missing keys, nulls and explicit columns', () => {
    expect(toCSV([{ a: 1 }, { b: null }])).toBe('a,b\r\n1,\r\n,\r\n')
    expect(toCSV([{ a: 1, b: 2 }], ['b'])).toBe('b\r\n2\r\n')
  })
})

describe('flatten', () => {
  it('flattens nested objects but keeps arrays', () => {
    expect(flatten({ id: 1, snippet: { title: 'T', text: 'x' }, tags: [1, 2] })).toEqual({
      id: 1,
      'snippet.title': 'T',
      'snippet.text': 'x',
      tags: [1, 2],
    })
  })
})
