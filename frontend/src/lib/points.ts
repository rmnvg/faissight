import type { RGBA } from './theme'

/** Pack a projection's columns into an xyz Float32Array. */
export function packPositions(x: number[], y: number[], z: number[] | null): Float32Array {
  const out = new Float32Array(x.length * 3)
  for (let i = 0; i < x.length; i++) {
    out[i * 3] = x[i]
    out[i * 3 + 1] = y[i]
    out[i * 3 + 2] = z ? z[i] : 0
  }
  return out
}

export function fillColors(n: number, color: (i: number) => RGBA): Uint8Array {
  const out = new Uint8Array(n * 4)
  for (let i = 0; i < n; i++) out.set(color(i), i * 4)
  return out
}
