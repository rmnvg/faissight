/** Client-side export of tables as CSV/JSON files. */

export type Row = Record<string, unknown>

function cell(v: unknown): string {
  if (v === null || v === undefined) return ''
  const s = typeof v === 'object' ? JSON.stringify(v) : String(v)
  // RFC 4180: quote fields containing separators, quotes or newlines; double inner quotes.
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/** CSV with a header row. Columns default to the union of keys, in first-seen order. */
export function toCSV(rows: Row[], columns?: string[]): string {
  const cols = columns ?? [...new Set(rows.flatMap((r) => Object.keys(r)))]
  const lines = [cols.map(cell).join(','), ...rows.map((r) => cols.map((c) => cell(r[c])).join(','))]
  return lines.join('\r\n') + '\r\n'
}

/** Flatten `{snippet: {title, text}}` style nesting into `snippet.title`, `snippet.text`. */
export function flatten(row: Row, prefix = ''): Row {
  const out: Row = {}
  for (const [k, v] of Object.entries(row)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === 'object' && !Array.isArray(v)) Object.assign(out, flatten(v as Row, key))
    else out[key] = v
  }
  return out
}

export function download(filename: string, content: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function downloadCSV(filename: string, rows: Row[]): void {
  download(filename, toCSV(rows.map((r) => flatten(r))), 'text/csv;charset=utf-8')
}

export function downloadJSON(filename: string, data: unknown): void {
  download(filename, JSON.stringify(data, null, 2) + '\n', 'application/json')
}
