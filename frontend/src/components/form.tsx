import type { ReactNode } from 'react'

/** A labelled form control (label above). */
export function Field({
  label,
  className = '',
  children,
}: {
  label: string
  className?: string
  children: ReactNode
}) {
  return (
    <label className={`flex flex-col gap-1 text-xs text-ink-2 ${className}`}>
      {label}
      {children}
    </label>
  )
}

/** Integer input clamped to [min, max]; empty or invalid input becomes `min`. */
export function NumberInput({
  value,
  onChange,
  min,
  max,
}: {
  value: number
  onChange: (v: number) => void
  min: number
  max: number
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))}
      className="tabular rounded-lg border border-line bg-page px-3 py-2 text-sm text-ink outline-none focus:border-series-1"
    />
  )
}
