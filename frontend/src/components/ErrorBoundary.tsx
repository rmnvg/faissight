import { Component, type ErrorInfo, type ReactNode } from 'react'
import { isChunkLoadError } from '../lib/errors'
import { Banner } from './ui'

interface State {
  error: unknown
}

/**
 * Keeps one broken view from blanking the whole app: shows the error with a way out.
 * Key it by view so navigating elsewhere starts clean.
 */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: unknown): State {
    return { error }
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error('faissight view crashed:', error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (error === null) return this.props.children
    // React.lazy caches a failed import, so only a reload can fetch the chunk again.
    const chunk = isChunkLoadError(error)
    const button =
      'rounded-lg border border-line bg-surface px-3 py-1.5 text-sm font-medium text-ink hover:bg-accent-wash'
    return (
      <div className="flex flex-col items-start gap-3 p-8">
        <Banner tone="error">
          {chunk
            ? 'This view could not be loaded from the faissight server. Is `faissight serve` still running?'
            : `This view hit an error: ${error instanceof Error ? error.message : String(error)}`}
        </Banner>
        <div className="flex gap-2">
          {!chunk && (
            <button className={button} onClick={() => this.setState({ error: null })}>
              Try again
            </button>
          )}
          <button className={button} onClick={() => window.location.reload()}>
            Reload page
          </button>
        </div>
      </div>
    )
  }
}
