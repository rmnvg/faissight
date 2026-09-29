import { lazy, Suspense, useEffect, useState } from 'react'
import { useInfo } from './api/hooks'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Sidebar } from './components/Sidebar'
import { Banner, Spinner } from './components/ui'
import { navigate, useRoute } from './lib/route'
import { useTheme } from './lib/theme'
import { Overview } from './views/Overview'

// deck.gl is heavy; only load it when a map view is opened.
const ClusterMap = lazy(() => import('./views/ClusterMap'))
const QueryExplorer = lazy(() => import('./views/QueryExplorer'))
const Tuner = lazy(() => import('./views/Tuner'))
const Hnsw = lazy(() => import('./views/Hnsw'))
const Quantization = lazy(() => import('./views/Quantization'))

export default function App() {
  const info = useInfo()
  const route = useRoute()
  const { mode, pref, setPref } = useTheme()

  // Focus mode hides the sidebar (screenshots, small screens). Toggle with the backslash key.
  const [focus, setFocus] = useState(false)
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
      {!focus && (
        <Sidebar
          info={info.data}
          view={view}
          onNavigate={(v) => navigate(v)}
          themePref={pref}
          onTheme={setPref}
          onFocus={() => setFocus(true)}
        />
      )}
      <main className="relative min-w-0 flex-1 overflow-auto">
        {focus && (
          <button
            onClick={() => setFocus(false)}
            className="absolute top-2 left-2 z-20 rounded-md border border-line bg-surface/90 px-2 py-1 text-xs text-ink-2 hover:text-ink print:hidden"
            title="Show the sidebar (\)"
          >
            ☰ Menu
          </button>
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
              {view === 'overview' && <Overview info={info.data} />}
              {view === 'map' && <ClusterMap info={info.data} params={route.params} mode={mode} />}
              {view === 'tuner' && <Tuner info={info.data} params={route.params} />}
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
