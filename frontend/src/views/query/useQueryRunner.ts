import { useMutation } from '@tanstack/react-query'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api } from '../../api/client'
import { useLatestSignal } from '../../api/hooks'
import type { Info, SearchRequest } from '../../api/types'
import { parseId } from '../../lib/ids'
import { intParam, navigate, toQueryString } from '../../lib/route'
import type { QueryMode, RunResult } from './types'

export const IVF_KINDS = new Set(['IVF_FLAT', 'IVF_PQ', 'IVF_SQ'])

/**
 * Query Explorer state: the form, the URL it mirrors, and the search request.
 *
 * The URL is the source of truth for stored-id and text queries, so results are shareable
 * and survive reloads; the query the URL describes runs on arrival. Pasted vectors are too
 * long for a URL and run directly.
 */
export function useQueryRunner(info: Info, params: URLSearchParams) {
  const isIvf = IVF_KINDS.has(info.kind)
  const isHnsw = info.kind.startsWith('HNSW')
  const hasEmbedder = info.inputs.embedder !== null

  const urlId = parseId(params.get('id'))
  const rowParam = intParam(params, 'row')
  const urlRow = rowParam !== null && rowParam >= 0 ? rowParam : null
  const [queryMode, setQueryMode] = useState<QueryMode>(
    urlId !== null ? 'id' : urlRow !== null ? 'row' : hasEmbedder ? 'text' : 'id',
  )
  const [text, setText] = useState(params.get('text') ?? '')
  const [idInput, setIdInput] = useState(urlId !== null ? String(urlId) : '')
  const [rowInput, setRowInput] = useState(urlRow !== null ? String(urlRow) : '')
  const [vectorInput, setVectorInput] = useState('')
  // Stable fallbacks for a URL that omits these: not the current form state, which drifts
  // as the user runs searches and would make the same shared link reproduce differently.
  const K_DEFAULT = 10
  const nprobeDefault = Number(info.params.nprobe ?? 1)
  const efSearchDefault = Number(info.params.ef_search ?? 16)
  const [k, setK] = useState(intParam(params, 'k') ?? K_DEFAULT)
  const [nprobe, setNprobe] = useState(intParam(params, 'nprobe') ?? nprobeDefault)
  const [efSearch, setEfSearch] = useState(intParam(params, 'ef') ?? efSearchDefault)
  const [compare, setCompare] = useState(true)
  const [candidates, setCandidates] = useState(intParam(params, 'candidates') ?? 0)
  const [formError, setFormError] = useState<string | null>(null)

  const nextSignal = useLatestSignal()
  const run = useMutation({
    mutationFn: async (req: SearchRequest): Promise<RunResult> => {
      // One round trip: the probe trace rides on the search, and only when comparing,
      // since explaining misses needs exact ground truth.
      const search = await api.search({ ...req, trace: isIvf && req.compare }, nextSignal())
      return { search, trace: search.ivf_trace, request: req }
    },
  })

  function fail(msg: string): null {
    setFormError(msg)
    return null
  }

  const buildRequest = (): SearchRequest | null => {
    let query: SearchRequest['query']
    if (queryMode === 'text') {
      if (!text.trim()) return fail('Type some query text.')
      query = { text: text.trim() }
    } else if (queryMode === 'id') {
      const id = parseId(idInput)
      if (id === null) return fail('Enter an integer id.')
      query = { id }
    } else if (queryMode === 'row') {
      const row = Number(rowInput.trim())
      const n = info.inputs.queries ?? 0
      if (rowInput.trim() === '' || !Number.isInteger(row) || row < 0 || row >= n)
        return fail(`Enter a row from 0 to ${n - 1} of the evaluation queries.`)
      query = { row }
    } else {
      const nums = vectorInput
        .replace(/[[\]\s]+/g, ' ')
        .split(/[ ,]+/)
        .filter(Boolean)
        .map(Number)
      if (nums.length !== info.d || nums.some((n) => !Number.isFinite(n)))
        return fail(`Paste ${info.d} numbers separated by commas or spaces (got ${nums.length}).`)
      query = { vector: nums }
    }
    if (candidates !== 0 && (!Number.isInteger(candidates) || candidates < k || candidates > (info.demo_limits?.max_k ?? 10000)))
      return fail('Rerank candidates must be at least k and within the displayed limit.')
    setFormError(null)
    return {
      query,
      k,
      compare,
      ...(candidates > 0 ? { candidates } : {}),
      ...(isIvf ? { nprobe } : {}),
      ...(isHnsw ? { efSearch } : {}),
      projection: { method: 'pca', dims: 2 },
    }
  }

  const urlParams = (req: SearchRequest): Record<string, string | number | null> => ({
    id: req.query.id ?? null,
    row: req.query.row ?? null,
    text: req.query.text ?? null,
    k: req.k,
    nprobe: req.nprobe ?? null,
    ef: req.efSearch ?? null,
    compare: req.compare ? null : 0,
    candidates: req.candidates ?? null,
  })

  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    const req = buildRequest()
    if (!req) return
    if (req.query.vector) {
      run.mutate(req)
      return
    }
    const next = urlParams(req)
    if (toQueryString(next) === params.toString()) run.mutate(req) // same URL: re-run
    else navigate('query', next)
  }

  // Run the query the URL describes (links from other views, shared links, reloads).
  const urlKey = params.toString()
  const urlText = params.get('text')
  const autoRan = useRef<string | null>(null)
  useEffect(() => {
    if ((urlId === null && urlRow === null && !urlText) || autoRan.current === urlKey) return
    autoRan.current = urlKey
    const urlK = intParam(params, 'k') ?? K_DEFAULT
    const urlNprobe = intParam(params, 'nprobe') ?? nprobeDefault
    const urlEf = intParam(params, 'ef') ?? efSearchDefault
    const urlCompare = params.get('compare') !== '0'
    setQueryMode(urlId !== null ? 'id' : urlRow !== null ? 'row' : 'text')
    if (urlId !== null) setIdInput(String(urlId))
    else if (urlRow !== null) setRowInput(String(urlRow))
    else setText(urlText ?? '')
    setK(urlK)
    setNprobe(urlNprobe)
    setEfSearch(urlEf)
    setCompare(urlCompare)
    const urlCandidates = intParam(params, 'candidates') ?? 0
    setCandidates(urlCandidates)
    run.mutate({
      query: urlId !== null ? { id: urlId } : urlRow !== null ? { row: urlRow } : { text: urlText ?? '' },
      k: urlK,
      compare: urlCompare,
      ...(urlCandidates > 0 ? { candidates: urlCandidates } : {}),
      ...(isIvf ? { nprobe: urlNprobe } : {}),
      ...(isHnsw ? { efSearch: urlEf } : {}),
      projection: { method: 'pca', dims: 2 },
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlKey])

  return {
    isIvf,
    isHnsw,
    form: {
      queryMode,
      setQueryMode,
      text,
      setText,
      idInput,
      setIdInput,
      rowInput,
      setRowInput,
      vectorInput,
      setVectorInput,
      k,
      setK,
      nprobe,
      setNprobe,
      efSearch,
      setEfSearch,
      compare,
      setCompare,
      candidates,
      setCandidates,
      formError,
    },
    submit,
    run,
  }
}

export type QueryRunner = ReturnType<typeof useQueryRunner>
