import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { ApiError } from '../api/client'

interface Toast {
  id: number
  title: string
  message: string
  hint: string | null
}

const ToastContext = createContext<(err: unknown) => void>(() => {})

let nextId = 1

export function ToastProvider({
  children,
  register,
}: {
  children: ReactNode
  /** Receives the push function so non-React code (query caches) can raise toasts. */
  register?: (push: (err: unknown) => void) => void
}) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), [])

  const push = useCallback(
    (err: unknown) => {
      const toast: Toast =
        err instanceof ApiError
          ? { id: nextId++, title: err.code, message: err.message, hint: err.hint }
          : { id: nextId++, title: 'Error', message: String(err), hint: null }
      setToasts((t) => [...t.filter((x) => x.message !== toast.message), toast].slice(-4))
      window.setTimeout(() => dismiss(toast.id), 8000)
    },
    [dismiss],
  )

  useEffect(() => register?.(push), [register, push])

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            role="alert"
            className="pointer-events-auto rounded-lg border border-critical/40 bg-surface p-3 text-sm shadow-lg"
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="flex items-center gap-1.5 font-medium text-ink">
                  <span className="h-2 w-2 rounded-full bg-critical" aria-hidden />
                  {t.title}
                </div>
                <div className="mt-0.5 text-ink-2">{t.message}</div>
                {t.hint && <div className="mt-1 text-xs text-muted">Hint: {t.hint}</div>}
              </div>
              <button
                onClick={() => dismiss(t.id)}
                className="text-muted hover:text-ink"
                aria-label="Dismiss"
              >
                ✕
              </button>
            </div>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useToast() {
  return useContext(ToastContext)
}
