import { describe, expect, it } from 'vitest'
import { isChunkLoadError } from './errors'

describe('isChunkLoadError', () => {
  it('recognises failed lazy imports across browsers', () => {
    for (const message of [
      'Failed to fetch dynamically imported module: http://x/assets/Tuner-abc.js',
      'Importing a module script failed.',
      'error loading dynamically imported module: http://x/assets/Tuner-abc.js',
    ])
      expect(isChunkLoadError(new TypeError(message))).toBe(true)
  })

  it('leaves ordinary render errors alone', () => {
    expect(isChunkLoadError(new TypeError("Cannot read properties of undefined (reading 'x')"))).toBe(false)
    expect(isChunkLoadError('boom')).toBe(false)
  })
})
