import { lazy, Suspense } from 'react'
import { useInfo } from './api/hooks'
import { Sidebar } from './components/Sidebar'
import { Banner, Spinner } from './components/ui'
import { navigate, useRoute } from './lib/route'
import { useTheme } from './lib/theme'
import { Overview } from './views/Overview'

// deck.gl is heavy; only load it when a map view is opened.
const ClusterMap = lazy(() => import('./views/ClusterMap'))
const QueryExplorer = lazy(() => import('./views/QueryExplorer'))
const Tuner = lazy(() => import('./views/Tuner'))

export default function App() {
  const info = useInfo()
  const route = useRoute()
  const { mode, pref, setPref } = useTheme()

  const supported = info.data?.supported ?? false
  const view = supported ? route.view : 'overview'

  return (
    <div className="flex h-full bg-page text-ink">
      <Sidebar
        info={info.data}
        view={view}
        onNavigate={(v) => navigate(v)}
        themePref={pref}
        onTheme={setPref}
      />
      <main className="min-w-0 flex-1 overflow-auto">
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
            {view === 'query' && (
              <QueryExplorer info={info.data} params={route.params} mode={mode} />
            )}
          </Suspense>
        )}
      </main>
    </div>
  )
}
