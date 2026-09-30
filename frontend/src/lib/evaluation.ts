import type { Info } from '../api/types'

export interface CheckItem {
  key: 'queries' | 'truth' | 'precision' | 'relevance'
  label: string
  /** true = in place, false = worth fixing, null = informational or not measured yet. */
  ok: boolean | null
  detail: string
}

/** Half-width of a recall interval, or null without one. */
export function halfWidth(low: number | null, high: number | null): number | null {
  return low === null || high === null ? null : (high - low) / 2
}

/** How far to trust a tuning or comparison result, and what would make it more trustworthy.
 *
 * `measured` describes a finished run (its query origin and the recall interval's half-width
 * at the setting that matters); without it, the checklist describes the inputs only. */
export function evaluationChecklist(
  info: Pick<Info, 'ground_truth_source' | 'inputs'>,
  measured?: { queryOrigin: 'given' | 'sampled'; nQueries: number; halfWidth: number | null },
): CheckItem[] {
  const given = measured ? measured.queryOrigin === 'given' : info.inputs.queries !== null
  const items: CheckItem[] = [
    given
      ? {
          key: 'queries',
          label: 'Representative queries',
          ok: true,
          detail: `Using ${measured ? measured.nQueries : 'the'} given queries. Make sure they are held out (not the vectors themselves) and look like production traffic.`,
        }
      : {
          key: 'queries',
          label: 'Representative queries',
          ok: false,
          detail:
            'Queries are sampled stored vectors (each excluded from its own results). Real queries often come from a different distribution (questions vs passages), so recall can differ in production. Pass held-out queries with --queries.',
        },
    info.ground_truth_source === 'raw'
      ? { key: 'truth', label: 'Exact ground truth', ok: true, detail: 'Computed on the raw vectors.' }
      : {
          key: 'truth',
          label: 'Exact ground truth',
          ok: false,
          detail: 'Computed on vectors decoded from the index, so PQ/SQ error is invisible. Pass --vectors.',
        },
  ]
  const hw = measured?.halfWidth ?? null
  items.push(
    hw === null
      ? {
          key: 'precision',
          label: 'Enough queries',
          ok: null,
          detail: 'Run to see how precisely recall is measured (its 95% interval).',
        }
      : hw <= 0.01
        ? { key: 'precision', label: 'Enough queries', ok: true, detail: `Recall is measured to ±${hw.toFixed(3)}.` }
        : {
            key: 'precision',
            label: 'Enough queries',
            ok: false,
            detail: `Recall is only measured to ±${hw.toFixed(3)}; about ${Math.ceil(
              (measured!.nQueries * (hw / 0.01) ** 2) / 100,
            ) * 100} queries would bring it to ±0.01.`,
          },
  )
  items.push({
    key: 'relevance',
    label: 'Recall is not relevance',
    ok: null,
    detail:
      'Recall here is agreement with exact search on the same embeddings. If exact neighbours are themselves poor answers, the problem is the embedding model or chunking, not the index; judge that with labelled queries.',
  })
  return items
}
