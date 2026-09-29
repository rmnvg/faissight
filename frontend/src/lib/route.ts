import { useEffect, useState } from 'react'

export type View = 'overview' | 'map' | 'query' | 'tuner' | 'hnsw' | 'quantization'
export interface Route {
  view: View
  params: URLSearchParams
}

const VIEWS: View[] = ['overview', 'map', 'query', 'tuner', 'hnsw', 'quantization']

function parse(hash: string): Route {
  const [path, qs = ''] = hash.replace(/^#\/?/, '').split('?')
  const view = (VIEWS as string[]).includes(path) ? (path as View) : 'overview'
  return { view, params: new URLSearchParams(qs) }
}

/** Hash-based route: `#/map?list=3`. Keeps views linkable without a router dependency. */
export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parse(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export function navigate(view: View, params: Record<string, string | number | null> = {}) {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== '') qs.set(k, String(v))
  const s = qs.toString()
  window.location.hash = `/${view}${s ? `?${s}` : ''}`
}

/** Update query params of the current view without adding history entries. */
export function replaceParams(view: View, params: Record<string, string | number | null>) {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== '') qs.set(k, String(v))
  const s = qs.toString()
  const url = `${window.location.pathname}${window.location.search}#/${view}${s ? `?${s}` : ''}`
  window.history.replaceState(null, '', url)
}

export function intParam(params: URLSearchParams, key: string): number | null {
  const v = params.get(key)
  if (v === null || v.trim() === '') return null
  const n = Number(v)
  return Number.isInteger(n) ? n : null
}
