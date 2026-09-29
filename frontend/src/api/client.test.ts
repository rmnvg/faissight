import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, isAbort, REQUEST_TIMEOUT_MS } from './client'

/** A fetch that never answers until its signal aborts, like a hung server. */
function hangingFetch() {
  return vi.fn(
    (_url: string, init?: RequestInit) =>
      new Promise<Response>((_, reject) => {
        const abort = () => reject(new DOMException('aborted', 'AbortError'))
        if (init?.signal?.aborted) abort()  // real fetch rejects at once, too
        init?.signal?.addEventListener('abort', abort)
      }),
  )
}

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('api client', () => {
  it('times out a hung request with a structured error', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', hangingFetch())
    const pending = api.info().catch((e: unknown) => e)
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS)
    const err = await pending
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).code).toBe('TIMEOUT')
    expect(isAbort(err)).toBe(false)
  })

  it('reports a caller cancellation as an abort, not a failure', async () => {
    vi.stubGlobal('fetch', hangingFetch())
    const ctrl = new AbortController()
    const pending = api.search({ query: { id: 1 }, k: 10, compare: true }, ctrl.signal).catch(
      (e: unknown) => e,
    )
    ctrl.abort()
    const err = await pending
    expect(isAbort(err)).toBe(true)
    expect(err).not.toBeInstanceOf(ApiError)
  })

  it('rejects at once when the signal is already aborted', async () => {
    const fetch = hangingFetch()
    vi.stubGlobal('fetch', fetch)
    const ctrl = new AbortController()
    ctrl.abort()
    const err = await api.info(ctrl.signal).catch((e: unknown) => e)
    expect(isAbort(err)).toBe(true)
    expect((fetch.mock.calls[0][1] as RequestInit).signal?.aborted).toBe(true)
  })

  it('maps a dropped connection to NETWORK_ERROR', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    const err = await api.info().catch((e: unknown) => e)
    expect((err as ApiError).code).toBe('NETWORK_ERROR')
  })

  it('passes structured server errors through', async () => {
    const body = { error_code: 'NOT_IVF', message: 'Not an IVF index.', hint: 'Use /search.' }
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(JSON.stringify(body), { status: 400 }))))
    const err = (await api.info().catch((e: unknown) => e)) as ApiError
    expect([err.status, err.code, err.hint]).toEqual([400, 'NOT_IVF', 'Use /search.'])
  })

  it.each([
    ['an HTML page', '<!doctype html><title>Sign in</title>', 'text/html'],
    ['an empty body', '', 'application/json'],
    ['a bare JSON value', 'null', 'application/json'],
    ['a JSON array', '[1, 2]', 'application/json'],
  ])('rejects %s sent with HTTP 200', async (_, text, type) => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(text, { status: 200, headers: { 'content-type': type } }))))
    const err = (await api.info().catch((e: unknown) => e)) as ApiError
    expect(err).toBeInstanceOf(ApiError)
    expect(err.code).toBe('INVALID_RESPONSE')
    expect(err.message).toContain('info')
  })

  it('still reports a non-JSON error page by its HTTP status', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response('Bad gateway', { status: 502 }))))
    const err = (await api.info().catch((e: unknown) => e)) as ApiError
    expect([err.code, err.message]).toEqual(['HTTP_502', 'Bad gateway'])
  })
})
