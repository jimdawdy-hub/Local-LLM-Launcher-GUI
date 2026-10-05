import { useCallback, useEffect, useRef, useState } from 'react'
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
  dashboard: 'Overview / System status',
  models: 'Library / Installed & search',
  launch: 'Launch / Configure & start',
  servers: 'Servers / Running instances',
  settings: 'Settings / Configuration',
}

export default function App() {
  const [tab, setTab] = useState('dashboard')
  const [hardware, setHardware] = useState(null)
  const [servers, setServers] = useState([])
  const [toast, setToast] = useState(null)
  const [launchModel, setLaunchModel] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [version, setVersion] = useState(null)
  const serversRequest = useRef(null)
  const hardwareRequest = useRef(null)
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

  useEffect(() => {
    api.about().then(r => setVersion(r?.version ?? null)).catch(() => setVersion(null))
  }, [])

  const notify = useCallback((message, error = false) => setToast({ message, error, at: Date.now() }), [])

  const refreshServers = useCallback(() => {
    if (serversRequest.current) return serversRequest.current
    serversRequest.current = (async () => {
      try {
        const r = await api.servers()
        setServers(r?.servers ?? [])
        return true
      } catch {
        return false
      } finally {
        serversRequest.current = null
      }
    })()
    return serversRequest.current
  }, [])

  const refreshHardware = useCallback(() => {
    if (hardwareRequest.current) return hardwareRequest.current
    hardwareRequest.current = (async () => {
      try {
        setHardware(await api.hardware())
        return true
      } catch {
        return false
      } finally {
        hardwareRequest.current = null
      }
    })()
    return hardwareRequest.current
  }, [])

  const refresh = async () => {
    setRefreshing(true)
    try {
      const results = await Promise.all([refreshHardware(), refreshServers()])
      const ok = results.every(Boolean)
      notify(ok ? 'Hardware and server status refreshed.' : 'Refresh failed. Check the launcher connection.', !ok)
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    refreshHardware()
    refreshServers()
    const t = setInterval(refreshServers, 3000)
    const hw = setInterval(refreshHardware, 6000)
    return () => { clearInterval(t); clearInterval(hw) }
  }, [refreshServers, refreshHardware])

  const goLaunch = useCallback((model) => {
    setLaunchModel(model)
    setTab('launch')
  }, [])

  const running = servers.filter((s) => s.running).length
  const vramUsed = hardware?.total_vram_mb
    ? ((hardware.gpus ?? []).reduce((sum, gpu) => sum + Math.max(0, gpu.vram_total_mb - gpu.vram_free_mb), 0) / 1024).toFixed(1)
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
              <div className="sb-brand-sub">Launcher{version && ` v${version}`}</div>
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
          <button className="topbar-btn topbar-btn-ghost" onClick={refresh} disabled={refreshing}
            aria-busy={refreshing}>{refreshing ? 'Refreshing…' : 'Refresh'}</button>
          <button className="topbar-btn topbar-btn-primary" onClick={() => goLaunch(null)}>Launch a model</button>
        </TopBar>

        <div className="content">
          {tab === 'dashboard' && (
            <Dashboard hardware={hardware} servers={servers} goLaunch={goLaunch} setTab={setTab} notify={notify} />
          )}
          {tab === 'models' && <Models goLaunch={goLaunch} notify={notify} />}
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
