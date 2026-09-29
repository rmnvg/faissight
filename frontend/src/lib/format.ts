/** Number formatting shared by views. */

export function fmtNum(n: number): string {
  return n.toLocaleString('en-US')
}

export function fmtDist(n: number): string {
  const a = Math.abs(n)
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return n.toExponential(2)
  return n.toFixed(a < 1 ? 4 : a < 100 ? 3 : 1)
}
