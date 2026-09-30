import type { CheckItem } from '../lib/evaluation'
import { Card } from './ui'

const MARK = {
  ok: { icon: '✓', cls: 'text-good-text', label: 'in place' },
  fix: { icon: '!', cls: 'text-critical', label: 'worth fixing' },
  info: { icon: 'i', cls: 'text-ink-2', label: 'note' },
}

/** "How far to trust this": the inputs and measurement behind a tuning or comparison result. */
export function EvaluationChecklist({ items }: { items: CheckItem[] }) {
  return (
    <Card title="How far to trust this" subtitle="What the numbers below are measured against">
      <ul className="grid gap-x-6 gap-y-2 md:grid-cols-2">
        {items.map((item) => {
          const m = item.ok === true ? MARK.ok : item.ok === false ? MARK.fix : MARK.info
          return (
            <li key={item.key} className="flex gap-2 text-sm">
              <span
                aria-label={m.label}
                className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-current text-[10px] font-semibold ${m.cls}`}
              >
                {m.icon}
              </span>
              <div>
                <div className="font-medium text-ink">{item.label}</div>
                <div className="text-xs text-ink-2">{item.detail}</div>
              </div>
            </li>
          )
        })}
      </ul>
    </Card>
  )
}
