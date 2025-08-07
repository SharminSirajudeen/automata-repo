import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import errorMonitoring, { ErrorBoundary } from './utils/error-monitoring'

// Initialize error monitoring before app starts
errorMonitoring.initialize()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
