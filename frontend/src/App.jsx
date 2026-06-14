import { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import { Toast, TopBar, ThemeToggle } from './components.jsx'
import Dashboard from './views/Dashboard.jsx'
import Models from './views/Models.jsx'
import Launch from './views/Launch.jsx'
import Servers from './views/Servers.jsx'
import Settings from './views/Settings.jsx'

const TABS = [
  { id: 'dashboard', label: 'Dashboard', icon: '■' },
  { id: 'models', label: 'Models', icon: '▶', badge: true },
  { id: 'launch', label: 'Launch', icon: '⚡' },
  { id: 'servers', label: 'Servers', icon: '⚙' },
  { id: 'settings', label: 'Settings', icon: '✎' },
]

const BREADCRUMBS = {
  dashboard: ['Overview / ', <span key="s">System status</span>],
  models: ['Library / ', <span key="s">Installed & search</span>],
  launch: ['Launch / ', <span key="s">Configure & start</span>],
  servers: ['Servers / ', <span key="s">Running instances</span>],
  settings: ['Settings / ', <span key="s">Configuration</span>],
}

export default function App() {
  const [tab, setTab] = useState('dashboard')
  const [hardware, setHardware] = useState(null)
  const [servers, setServers] = useState([])
  const [toast, setToast] = useState(null)
  const [launchModel, setLaunchModel] = useState(null)
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light')

  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next = prev === 'light' ? 'dark' : 'light'
      localStorage.setItem('theme', next)
      document.documentElement.setAttribute('data-theme', next)
      return next
    })
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const notify = useCallback((message, error = false) => setToast({ message, error, at: Date.now() }), [])

  const refreshServers = useCallback(async () => {
    try {
      const r = await api.servers()
      setServers(r?.servers ?? [])
    } catch {
      setServers([])
    }
  }, [])

  useEffect(() => {
    api.hardware().then(setHardware).catch(() => setHardware(null))
    refreshServers()
    const t = setInterval(refreshServers, 3000)
    return () => clearInterval(t)
  }, [refreshServers])

  const goLaunch = useCallback((model) => {
    setLaunchModel(model)
    setTab('launch')
  }, [])

  const running = servers.filter((s) => s.running).length
  const modelCount = hardware ? null : null
  const vramUsed = hardware?.total_vram_mb
    ? (hardware.total_vram_mb / 1024).toFixed(1)
    : null
  const vramTotal = hardware?.total_vram_mb
    ? Math.round(hardware.total_vram_mb / 1024)
    : null

  return (
    <div className="frame">
      <aside className="sidebar">
        <div className="sb-brand">
          <div className="sb-brand-row">
            <div className="sb-logo">LL</div>
            <div className="sb-brand-text">
              <div className="sb-brand-name">Local LLM</div>
              <div className="sb-brand-sub">Launcher v0.4.0</div>
            </div>
          </div>
        </div>

        <div className="sb-section">
          <div className="sb-section-label">Navigation</div>
        </div>
        <nav className="sb-nav" aria-label="Main">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`sb-btn ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              <span className="sb-icon">{t.icon}</span>
              {t.label}
              {t.id === 'servers' && running > 0 && <span className="sb-badge">{running}</span>}
            </button>
          ))}
        </nav>

        <div className="sb-spacer" />

        <div className="sb-status">
          <div className="sb-status-row">
            <span className="sb-status-dot" />
            <span className="sb-status-text">Active</span>
            <span className="sb-status-val">{running} server{running !== 1 ? 's' : ''}</span>
          </div>
          {vramUsed && (
            <div className="sb-status-row">
              <span className="sb-status-dot" style={{ background: 'var(--accent)' }} />
              <span className="sb-status-text">VRAM</span>
              <span className="sb-status-val">{vramUsed} / {vramTotal} GB</span>
            </div>
          )}
        </div>
      </aside>

      <main className="main">
        <TopBar title={TABS.find(t => t.id === tab)?.label} breadcrumb={BREADCRUMBS[tab]}>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
          <button className="topbar-btn topbar-btn-ghost" onClick={refreshServers}>Refresh</button>
          <button className="topbar-btn topbar-btn-primary" onClick={() => goLaunch(null)}>Launch a model</button>
        </TopBar>

        <div className="content">
          {tab === 'dashboard' && (
            <Dashboard hardware={hardware} servers={servers} goLaunch={goLaunch} setTab={setTab} notify={notify} />
          )}
          {tab === 'models' && <Models hardware={hardware} goLaunch={goLaunch} notify={notify} />}
          {tab === 'launch' && (
            <Launch hardware={hardware} initialModel={launchModel} notify={notify}
              onLaunched={() => { refreshServers(); setTab('servers') }} />
          )}
          {tab === 'servers' && <Servers servers={servers} refresh={refreshServers} notify={notify} />}
          {tab === 'settings' && <Settings hardware={hardware} notify={notify} />}
        </div>
      </main>

      <Toast toast={toast} onDone={() => setToast(null)} />
    </div>
  )
}
