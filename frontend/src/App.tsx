import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { useInfo } from './api/hooks'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Sidebar } from './components/Sidebar'
import { Banner, Spinner } from './components/ui'
import { navigate, useRoute } from './lib/route'
import { useTheme } from './lib/theme'
import { Overview } from './views/Overview'

// deck.gl is heavy; only load it when a map view is opened.
const Evaluation = lazy(() => import('./views/Evaluation'))
const History = lazy(() => import('./views/History'))
const ClusterMap = lazy(() => import('./views/ClusterMap'))
const QueryExplorer = lazy(() => import('./views/QueryExplorer'))
const Tuner = lazy(() => import('./views/Tuner'))
const Compare = lazy(() => import('./views/Compare'))
const Hnsw = lazy(() => import('./views/Hnsw'))
const Quantization = lazy(() => import('./views/Quantization'))

export default function App() {
  const info = useInfo()
  const route = useRoute()
  const { mode, pref, setPref } = useTheme()

  // Focus mode hides the sidebar (screenshots, small screens). Toggle with the backslash key.
  // Narrow frames (phones, notebook iframes) start with it hidden and open it as an overlay.
  const narrow = useNarrow()
  const [focus, setFocus] = useState(narrow)
  const [wasNarrow, setWasNarrow] = useState(narrow)
  const menu = useRef<HTMLDivElement>(null)
  const menuToggle = useRef<HTMLButtonElement>(null)
  // On narrow frames the open menu is a modal dialog: focus moves into it, Tab cycles
  // inside it, Escape closes it, and focus returns to the Menu button.
  useEffect(() => {
    if (!narrow || focus || !menu.current) return
    const container = menu.current
    const toggle = menuToggle
    const controls = () =>
      [...container.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], input, select, [tabindex="0"]')]
    controls()[0]?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        setFocus(true)
        return
      }
      if (e.key !== 'Tab') return
      const items = controls()
      const first = items[0]
      const last = items[items.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last?.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first?.focus()
      }
    }
    container.addEventListener('keydown', onKey)
    return () => {
      container.removeEventListener('keydown', onKey)
      // The Menu button only renders again once the menu has closed.
      requestAnimationFrame(() => toggle.current?.focus())
    }
  }, [narrow, focus])

  if (narrow !== wasNarrow) {
    setWasNarrow(narrow)
    setFocus(narrow)
  }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (e.key === '\\' && tag !== 'INPUT' && tag !== 'TEXTAREA') setFocus((f) => !f)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const supported = info.data?.supported ?? false
  const view = supported ? route.view : 'overview'

  return (
    <div className="flex h-full bg-page text-ink">
      {!focus && narrow && (
        <button
          aria-label="Close the menu"
          onClick={() => setFocus(true)}
          className="fixed inset-0 z-30 bg-black/30 print:hidden"
        />
      )}
      {!focus && (
        <div
          ref={menu}
          role={narrow ? 'dialog' : undefined}
          aria-modal={narrow ? true : undefined}
          aria-label={narrow ? 'Navigation menu' : undefined}
          className={narrow ? 'fixed inset-y-0 left-0 z-40 flex shadow-xl' : 'flex'}
        >
          <Sidebar
            info={info.data}
            view={view}
            onNavigate={(v) => {
              navigate(v)
              if (narrow) setFocus(true)
            }}
            themePref={pref}
            onTheme={setPref}
            onFocus={() => setFocus(true)}
          />
        </div>
      )}
      <main className="relative min-w-0 flex-1 overflow-auto">
        {focus && (
          // On narrow frames the button sits in its own bar so it never covers a page heading.
          <div className={narrow ? 'sticky top-0 z-20 border-b border-line bg-page/95 px-2 py-1.5 print:hidden' : ''}>
            <button
              ref={menuToggle}
              onClick={() => setFocus(false)}
              className={`${narrow ? '' : 'absolute top-2 left-2 z-20'} rounded-md border border-line bg-surface/90 px-2 py-1 text-xs text-ink-2 hover:text-ink print:hidden`}
              title="Show the sidebar (\)"
            >
              ☰ Menu
            </button>
          </div>
        )}
        {info.isPending ? (
          <div className="p-8">
            <Spinner label="Connecting to faissight…" />
          </div>
        ) : info.isError ? (
          <div className="p-8">
            <Banner tone="error">
              Could not load the index: {info.error.message}. Is <code>faissight serve</code>{' '}
              running?
            </Banner>
          </div>
        ) : (
          <ErrorBoundary key={view}>
            <Suspense
              fallback={
                <div className="p-8">
                  <Spinner />
                </div>
              }
            >
              {view === 'evaluation' && <Evaluation info={info.data} params={route.params} />}
              {view === 'history' && <History />}
              {view === 'overview' && <Overview info={info.data} />}
              {view === 'map' && <ClusterMap info={info.data} params={route.params} mode={mode} />}
              {view === 'tuner' && <Tuner info={info.data} params={route.params} />}
              {view === 'compare' && <Compare info={info.data} params={route.params} />}
              {view === 'hnsw' && <Hnsw info={info.data} params={route.params} mode={mode} />}
              {view === 'quantization' && <Quantization info={info.data} />}
              {view === 'query' && (
                <QueryExplorer info={info.data} params={route.params} mode={mode} />
              )}
            </Suspense>
          </ErrorBoundary>
        )}
      </main>
    </div>
  )
}

const NARROW_QUERY = '(max-width: 767px)'

/** True while the frame is narrower than Tailwind's `md` breakpoint. */
function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => window.matchMedia?.(NARROW_QUERY).matches ?? false)
  useEffect(() => {
    const mq = window.matchMedia?.(NARROW_QUERY)
    if (!mq) return
    const onChange = () => setNarrow(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  return narrow
}
