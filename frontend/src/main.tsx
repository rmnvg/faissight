import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { isAbort } from './api/client'
import App from './App.tsx'
import { ToastProvider } from './components/toasts'
import './index.css'

// Query/mutation failures surface as toasts; views still render their own inline states.
let pushToast: (err: unknown) => void = () => {}
const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: (err) => pushToast(err) }),
  // Superseded searches are cancelled on purpose; that isn't worth a toast.
  mutationCache: new MutationCache({ onError: (err) => !isAbort(err) && pushToast(err) }),
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ToastProvider register={(push) => (pushToast = push)}>
        <App />
      </ToastProvider>
    </QueryClientProvider>
  </StrictMode>,
)
