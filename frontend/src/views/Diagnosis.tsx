import type { Info } from '../api/types'
import { Card } from '../components/ui'

/** A capability-aware path through the existing diagnostic views. */
export function Diagnosis({ info }: { info: Info }) {
  const hasQueries = (info.inputs.queries ?? 0) > 0
  const steps = [
    {
      title: 'Check your inputs',
      detail: info.inputs.raw_vectors
        ? 'Raw vectors are available for exact comparisons. Check metadata coverage in Inputs above.'
        : 'Restart with --vectors to measure against the original embeddings; reconstructed vectors can hide compression loss.',
    },
    {
      title: 'Inspect a representative query',
      detail: hasQueries
        ? 'Start with the first held-out query, then inspect its missed neighbours and their explanations.'
        : 'Try a query your users would ask. Supply --queries for repeatable evaluation; stored-vector queries can be optimistic.',
      href: hasQueries ? '#/query?row=0' : '#/query',
      action: 'Inspect a query',
    },
    ...(info.sweep ? [{
      title: 'Tune recall and latency',
      detail: 'Choose a recall target and latency budget. Inspect the worst queries and follow the measured next steps.',
      href: '#/tuner', action: 'Open tuner',
    }] : []),
    {
      title: 'Verify the improvement',
      detail: info.compare.length > 0
        ? 'Compare your indexes on the same queries, then inspect which neighbours changed.'
        : info.sweep
          ? 'Save a run in the Tuner, make your change, then compare with that baseline using the same held-out queries.'
          : 'Start with --compare OTHER.index to compare a candidate index on identical queries.',
      href: info.compare.length > 0 ? '#/compare' : info.sweep ? '#/tuner' : undefined,
      action: info.compare.length > 0 ? 'Compare indexes' : 'Save a baseline',
    },
  ]
  return (
    <Card title="Diagnose my index" subtitle="Follow a query from its inputs to a measured improvement">
      <ol className="grid gap-4 md:grid-cols-2">
        {steps.map((step, i) => (
          <li key={step.title}>
            <h3 className="text-sm font-medium">{i + 1}. {step.title}</h3>
            <p className="mt-1 text-sm text-ink-2">{step.detail}</p>
            {step.href && <a href={step.href} className="mt-2 inline-block text-sm text-series-1 underline">{step.action} →</a>}
          </li>
        ))}
      </ol>
      <p className="mt-4 text-xs text-muted">ANN recall measures agreement with exact vector search. Use labelled evaluation to measure whether those chunks are relevant to the question.</p>
    </Card>
  )
}
