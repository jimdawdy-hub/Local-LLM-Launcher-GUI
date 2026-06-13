import { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'
import { Led } from '../components.jsx'

function OpenWebUIPanel({ notify }) {
  const [status, setStatus] = useState(null)
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)

  const refresh = useCallback(() => {
    api.openwebui().then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 4000)
    return () => clearInterval(t)
  }, [refresh])

  const launch = async () => {
    setBusy(true)
    try {
      const s = await api.launchOpenwebui()
      setStatus(s)
      const n = s.connected_models || 0
      notify(
        n > 0
          ? `Open WebUI is starting and will connect to your ${n} running model${n > 1 ? 's' : ''}. Your browser opens automatically when it's ready (first run can take a minute).`
          : "Open WebUI is starting; your browser opens when it's ready. Start a model on the Launch tab to chat with it.",
      )
    } catch (e) {
      notify(e.message, true)
    } finally {
      setBusy(false)
    }
  }

  const stop = async () => {
    setBusy(true)
    try {
      setStatus(await api.stopOpenwebui())
      notify('Open WebUI stopped.')
    } catch (e) {
      notify(e.message, true)
    } finally {
      setBusy(false)
    }
  }

  const copyCmd = () => {
    navigator.clipboard?.writeText(status?.install_command || 'pip install open-webui')
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const installed = !!status?.installed
  const running = !!status?.running

  return (
    <div className="panel">
      <div className="row between">
        <div className="row">
          <Led level={running ? 'green' : 'off'} pulse={running} title={running ? 'running' : 'stopped'} />
          <h2>Open WebUI</h2>
        </div>
        {running ? (
          <div className="row">
            <a className="btn primary sm" href={status.url} target="_blank" rel="noreferrer">Open Open WebUI ↗</a>
            <button className="btn danger sm" onClick={stop} disabled={busy}>Stop</button>
          </div>
        ) : (
          <button className="btn primary" onClick={launch} disabled={!installed || busy}
            title={installed ? 'Start Open WebUI' : 'Open WebUI is not installed'}>
            {busy ? 'Launching…' : 'Launch Open WebUI'}
          </button>
        )}
      </div>

      <p className="small muted" style={{ marginTop: 8 }}>
        A polished chat interface for your local models. Launch it here, then point it at a
        running model's endpoint (shown on the Servers tab).
      </p>

      {status && !installed && (
        <div className="stack" style={{ marginTop: 10 }}>
          <p className="small" style={{ color: 'var(--caution)' }}>
            Open WebUI isn't installed yet. Run this command in your terminal, then it'll
            light up here:
          </p>
          <div className="row">
            <input className="mono" readOnly value={status.install_command}
              onFocus={(e) => e.target.select()}
              style={{ flex: 1, fontSize: 12.5 }}
              aria-label="Open WebUI install command" />
            <button className="btn sm" onClick={copyCmd}>{copied ? 'Copied' : 'Copy'}</button>
          </div>
        </div>
      )}

      {running && (
        <p className="small faint mono" style={{ marginTop: 8 }}>Serving at {status.url}</p>
      )}
    </div>
  )
}

function GpuBar({ gpu }) {
  const usedMb = gpu.vram_total_mb - gpu.vram_free_mb
  const pct = gpu.vram_total_mb > 0 ? Math.round((usedMb / gpu.vram_total_mb) * 100) : 0
  return (
    <div className="gpubar">
      <div className="row between small">
        <span>GPU {gpu.index} — {gpu.name}</span>
        <span className="mono muted">{(usedMb / 1024).toFixed(1)} / {(gpu.vram_total_mb / 1024).toFixed(0)} GB held (apps + driver)</span>
      </div>
      <div className="track"><div className="fill" style={{ width: `${pct}%` }} /></div>
    </div>
  )
}

function EngineRow({ ok, name, note }) {
  return (
    <div className="row" style={{ padding: '6px 0' }}>
      <Led level={ok ? 'green' : 'off'} title={ok ? 'available' : 'not found'} />
      <span style={{ fontWeight: 600 }}>{name}</span>
      <span className="small muted">{note}</span>
    </div>
  )
}

export default function Dashboard({ hardware, servers, goLaunch, setTab, notify }) {
  const running = servers.filter((s) => s.running)
  return (
    <>
      <div className="row between">
        <h1>Your machine</h1>
        <button className="btn primary" onClick={() => goLaunch(null)}>Launch a model</button>
      </div>

      <div className="panel">
        {!hardware && <p className="muted">Reading your hardware…</p>}
        {hardware && (
          <div className="stack">
            <p>{hardware.summary}</p>
            {hardware.gpus.map((g) => <GpuBar key={g.index} gpu={g} />)}
            {hardware.notes.map((n, i) => (
              <p key={i} className="small" style={{ color: 'var(--caution)' }}>{n}</p>
            ))}
            <div className="row small muted" style={{ gap: 18 }}>
              <span>{hardware.cpu_cores} CPU cores</span>
              <span>{hardware.ram_gb} GB RAM</span>
              <span>{hardware.disk_free_gb} GB disk free</span>
            </div>
          </div>
        )}
      </div>

      <div className="grid2">
        <div className="panel">
          <h2 style={{ marginBottom: 8 }}>Engines</h2>
          {hardware ? (
            <>
              <EngineRow ok={hardware.engines.vllm_native} name="vLLM"
                note={hardware.engines.vllm_native ? 'installed' : 'not installed'} />
              <EngineRow ok={hardware.engines.vllm_docker} name="vLLM (Docker)"
                note={hardware.engines.vllm_docker ? 'image found' : 'no Docker image'} />
              <EngineRow ok={!!hardware.engines.llamacpp_path} name="llama.cpp"
                note={hardware.engines.llamacpp_path || 'not found — see Settings'} />
            </>
          ) : <p className="muted">…</p>}
        </div>

        <div className="panel">
          <h2 style={{ marginBottom: 8 }}>Running now</h2>
          {running.length === 0 && (
            <p className="muted small">Nothing running. Pick a model on the Launch tab to start one.</p>
          )}
          {running.map((s) => (
            <div key={s.id} className="row" style={{ padding: '6px 0' }}>
              <Led level="green" pulse title="running" />
              <span className="mono small" style={{ flex: 1, wordBreak: 'break-all' }}>{s.model}</span>
              <button className="btn sm" onClick={() => setTab('servers')}>Manage</button>
            </div>
          ))}
        </div>
      </div>

      <OpenWebUIPanel notify={notify} />

      <div className="panel inset small muted">
        <strong>How this works:</strong> pick or download a model on the <em>Models</em> tab, review the
        traffic-light settings on the <em>Launch</em> tab, and press launch. Every setting shows
        green when it's safe for your hardware, yellow when it deserves a look, and red when it
        would fail. You never have to guess a flag again.
      </div>
    </>
  )
}
