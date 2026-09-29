import { describe, expect, it } from 'vitest'
import { intParam } from './route'

describe('intParam', () => {
  it('reads integers and rejects everything else', () => {
    const p = new URLSearchParams('id=42&neg=-3&f=1.5&s=abc&e=')
    expect(intParam(p, 'id')).toBe(42)
    expect(intParam(p, 'neg')).toBe(-3)
    expect(intParam(p, 'f')).toBeNull()
    expect(intParam(p, 's')).toBeNull()
    expect(intParam(p, 'e')).toBeNull()
    expect(intParam(p, 'missing')).toBeNull()
  })
})
