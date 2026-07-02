import { useState } from 'react'
import Predictor from './components/Predictor.jsx'
import Dashboard from './components/Dashboard.jsx'

const GOLD = '#C9A02C'
const GOLD_L = '#E8C96A'

const s = {
  app: { minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#000', color: '#E8E8E8',
         fontFamily: "'Montserrat', 'Segoe UI', sans-serif" },
  header: {
    background: '#000',
    borderBottom: `1px solid rgba(201,160,44,0.35)`,
    padding: '22px 40px',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  },
  logoWrap: { display: 'flex', alignItems: 'center', gap: 16 },
  logoIcon: { fontSize: 26, color: GOLD },
  logoLine1: { color: GOLD, fontSize: 18, fontWeight: '600', letterSpacing: '0.22em', textTransform: 'uppercase' },
  logoLine2: { color: '#666', fontSize: 10, letterSpacing: '0.25em', textTransform: 'uppercase', marginTop: 3 },
  badge: {
    background: 'rgba(201,160,44,0.10)', border: `1px solid rgba(201,160,44,0.35)`,
    color: GOLD, padding: '5px 14px', borderRadius: 20, fontSize: 10, letterSpacing: '0.12em',
    textTransform: 'uppercase',
  },
  tabs: {
    display: 'flex', background: '#000', borderBottom: `1px solid rgba(201,160,44,0.18)`,
    padding: '0 40px',
  },
  tab: (active) => ({
    padding: '16px 28px', fontSize: 11, fontWeight: '600', letterSpacing: '0.18em',
    textTransform: 'uppercase', border: 'none', background: 'none', cursor: 'pointer',
    color: active ? GOLD : '#555',
    borderBottom: active ? `2px solid ${GOLD}` : '2px solid transparent',
    transition: 'all 0.2s',
  }),
  main: { flex: 1, padding: '40px', maxWidth: 1100, margin: '0 auto', width: '100%' },
  footer: {
    borderTop: `1px solid rgba(201,160,44,0.12)`, padding: '16px 40px',
    textAlign: 'center', color: '#444', fontSize: 10, letterSpacing: '0.12em', background: '#000',
  },
}

export default function App() {
  const [tab, setTab] = useState('predictor')

  return (
    <div style={s.app}>
      <header style={s.header}>
        <div style={s.logoWrap}>
          <span style={s.logoIcon}>◆</span>
          <div>
            <div style={s.logoLine1}>Lartisien Collection</div>
            <div style={s.logoLine2}>Rate Intelligence · ML Predictor</div>
          </div>
        </div>
        <span style={s.badge}>MADA 2025</span>
      </header>

      <nav style={s.tabs}>
        <button style={s.tab(tab === 'predictor')} onClick={() => setTab('predictor')}>
          ◈ Rate Predictor
        </button>
        <button style={s.tab(tab === 'dashboard')} onClick={() => setTab('dashboard')}>
          ◉ Model Dashboard
        </button>
      </nav>

      <main style={s.main}>
        {tab === 'predictor' ? <Predictor /> : <Dashboard />}
      </main>

      <footer style={s.footer}>
        GradientBoosting · scikit-learn · XGBoost &nbsp;·&nbsp; Classification F1=0.805 &nbsp;·&nbsp; Regression R²=0.736
        &nbsp;|&nbsp; MADA 2025 Final Project
      </footer>
    </div>
  )
}
