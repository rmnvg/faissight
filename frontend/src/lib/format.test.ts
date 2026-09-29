import { describe, expect, it } from 'vitest'
import { fmtDist, fmtNum } from './format'
import { hex, rampColor } from './theme'

describe('format', () => {
  it('formats counts and distances', () => {
    expect(fmtNum(20000)).toBe('20,000')
    expect(fmtDist(63.7991)).toBe('63.799')
    expect(fmtDist(0.12345)).toBe('0.1235')
    expect(fmtDist(1234.5)).toBe('1234.5')
    expect(fmtDist(0.00001)).toBe('1.00e-5')
  })
})

describe('theme', () => {
  it('parses hex colours and clamps the ramp', () => {
    expect(hex('#2a78d6')).toEqual([42, 120, 214, 255])
    expect(rampColor('light', -1)).toEqual(hex('#cde2fb', 220))
    expect(rampColor('light', 2)).toEqual(hex('#0d366b', 220))
  })
})
