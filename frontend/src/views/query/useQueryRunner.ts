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
  const [queryMode, setQueryMode] = useState<QueryMode>(
    urlId !== null ? 'id' : hasEmbedder ? 'text' : 'id',
  )
  const [text, setText] = useState(params.get('text') ?? '')
  const [idInput, setIdInput] = useState(urlId !== null ? String(urlId) : '')
  const [vectorInput, setVectorInput] = useState('')
  const [k, setK] = useState(intParam(params, 'k') ?? 10)
  const [nprobe, setNprobe] = useState(intParam(params, 'nprobe') ?? Number(info.params.nprobe ?? 1))
  const [efSearch, setEfSearch] = useState(intParam(params, 'ef') ?? Number(info.params.ef_search ?? 16))
  const [compare, setCompare] = useState(true)
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
    setFormError(null)
    return {
      query,
      k,
      compare,
      ...(isIvf ? { nprobe } : {}),
      ...(isHnsw ? { efSearch } : {}),
      projection: { method: 'pca', dims: 2 },
    }
  }

  const urlParams = (req: SearchRequest): Record<string, string | number | null> => ({
    id: req.query.id ?? null,
    text: req.query.text ?? null,
    k: req.k,
    nprobe: req.nprobe ?? null,
    ef: req.efSearch ?? null,
    compare: req.compare ? null : 0,
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
    if ((urlId === null && !urlText) || autoRan.current === urlKey) return
    autoRan.current = urlKey
    const urlK = intParam(params, 'k') ?? k
    const urlNprobe = intParam(params, 'nprobe') ?? nprobe
    const urlEf = intParam(params, 'ef') ?? efSearch
    const urlCompare = params.get('compare') !== '0'
    setQueryMode(urlId !== null ? 'id' : 'text')
    if (urlId !== null) setIdInput(String(urlId))
    else setText(urlText ?? '')
    setK(urlK)
    setNprobe(urlNprobe)
    setEfSearch(urlEf)
    setCompare(urlCompare)
    run.mutate({
      query: urlId !== null ? { id: urlId } : { text: urlText ?? '' },
      k: urlK,
      compare: urlCompare,
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
      formError,
    },
    submit,
    run,
  }
}

export type QueryRunner = ReturnType<typeof useQueryRunner>
