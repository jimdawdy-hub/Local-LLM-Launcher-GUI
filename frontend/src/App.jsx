import { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import { Led, Toast } from './components.jsx'
import Dashboard from './views/Dashboard.jsx'
import Models from './views/Models.jsx'
import Launch from './views/Launch.jsx'
import Servers from './views/Servers.jsx'
import Settings from './views/Settings.jsx'

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'models', label: 'Models' },
  { id: 'launch', label: 'Launch' },
  { id: 'servers', label: 'Servers' },
  { id: 'settings', label: 'Settings' },
]

export default function App() {
  const [tab, setTab] = useState('dashboard')
  const [hardware, setHardware] = useState(null)
  const [servers, setServers] = useState([])
  const [toast, setToast] = useState(null)
  // Set when a view wants to pre-select a model on the Launch tab.
  const [launchModel, setLaunchModel] = useState(null)

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

  return (
    <div className="frame">
      <aside className="rail">
        <div className="brand">
          <h1>Local-LLM<br />Launcher</h1>
          <span className="sub">launch control</span>
        </div>
        <nav aria-label="Main">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`navbtn ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.id === 'servers'
                ? <Led level={running > 0 ? 'green' : 'off'} pulse={running > 0} title={`${running} running`} />
                : <span style={{ width: 17 }} />}
              {t.label}
              {t.id === 'servers' && running > 0 ? ` (${running})` : ''}
            </button>
          ))}
        </nav>
        <div className="spacer" />
        <div className="railfoot">
          {hardware && <span className="mono">{hardware.gpus.length > 0
            ? `${hardware.gpus.length} GPU · ${Math.round(hardware.total_vram_mb / 1024)} GB VRAM`
            : hardware.apple_silicon ? hardware.apple_silicon.chip : 'CPU only'}</span>}
          <span>Based on <a href="https://github.com/Chen-zexi/vllm-cli" target="_blank" rel="noreferrer">vllm-cli</a> by Chen-zexi</span>
        </div>
      </aside>

      <main className="main">
        <div className="main-inner">
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
