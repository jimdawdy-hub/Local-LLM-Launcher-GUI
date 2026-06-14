import { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'
import { MetricCard, RingGauge, StatusBadge } from '../components.jsx'

function OpenWebUIPanel({ notify }) {
  const [status, setStatus] = useState(null)
  const [busy, setBusy] = useState(false)

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

  const installed = !!status?.installed
  const running = !!status?.running

  return (
    <div className="webui-panel">
      <div className="webui-ring">
        <RingGauge percent={running ? 100 : 0} color={running ? 'var(--go)' : 'var(--ink-4)'}
          label={running ? '✓' : '—'} />
      </div>
      <div className="webui-info">
        <div className="webui-header">
          <span className="webui-title">Open WebUI</span>
          <StatusBadge level={running ? 'green' : 'neutral'}>{running ? 'Running' : 'Stopped'}</StatusBadge>
        </div>
        <p className="webui-desc">Polished chat interface connected to your running models.</p>
        <div style={{ display: 'flex', gap: 8 }}>
          {running ? (
            <>
              <a className="btn btn-primary" href={status.url} target="_blank" rel="noreferrer">Open Open WebUI</a>
              <button className="btn btn-secondary" onClick={stop} disabled={busy}>Stop</button>
            </>
          ) : (
            <button className="btn btn-primary" onClick={launch} disabled={!installed || busy}>
              {busy ? 'Launching…' : 'Launch Open WebUI'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function EngineRow({ ok, name, note }) {
  return (
    <div className="engine-row">
      <span className={`engine-dot ${ok ? 'on' : 'off'}`} />
      <span className="engine-name">{name}</span>
      <span className="engine-note">{note}</span>
    </div>
  )
}

export default function Dashboard({ hardware, servers, goLaunch, setTab, notify }) {
  const running = servers.filter((s) => s.running)

  const gpu0 = hardware?.gpus?.[0]
  const gpu1 = hardware?.gpus?.[1]

  const gpu0Used = gpu0 ? ((gpu0.vram_total_mb - gpu0.vram_free_mb) / 1024).toFixed(1) : '—'
  const gpu0Total = gpu0 ? Math.round(gpu0.vram_total_mb / 1024) : 0
  const gpu0Pct = gpu0 ? Math.round(((gpu0.vram_total_mb - gpu0.vram_free_mb) / gpu0.vram_total_mb) * 100) : 0

  const gpu1Used = gpu1 ? ((gpu1.vram_total_mb - gpu1.vram_free_mb) / 1024).toFixed(1) : '—'
  const gpu1Total = gpu1 ? Math.round(gpu1.vram_total_mb / 1024) : 0
  const gpu1Pct = gpu1 ? Math.round(((gpu1.vram_total_mb - gpu1.vram_free_mb) / gpu1.vram_total_mb) * 100) : 0

  const totalVramUsed = hardware ? (hardware.gpus.reduce((s, g) => s + (g.vram_total_mb - g.vram_free_mb), 0) / 1024).toFixed(1) : '0'
  const totalVram = hardware ? Math.round(hardware.total_vram_mb / 1024) : 0
  const totalVramPct = hardware && totalVram > 0 ? Math.round((parseFloat(totalVramUsed) / totalVram) * 100) : 0

  return (
    <>
      {/* METRIC CARDS */}
      <div className="metrics">
        {gpu0 && (
          <MetricCard colorClass="accent" label={`GPU ${gpu0.index}`}
            icon="■" value={gpu0Used} unit="GB"
            sub={`${gpu0.name} · ${gpu0Total} GB total`}
            change={`${gpu0Pct}%`} changeDir="up" />
        )}
        {gpu1 && (
          <MetricCard colorClass="green" label={`GPU ${gpu1.index}`}
            icon="■" value={gpu1Used} unit="GB"
            sub={`${gpu1.name} · ${gpu1Total} GB total`}
            change={`${gpu1Pct}%`} changeDir="up" />
        )}
        {hardware && (
          <MetricCard label="System" icon="■"
            value={hardware.ram_gb} unit="GB"
            sub={`${hardware.cpu_cores} CPU cores`} />
        )}
        {hardware && (
          <MetricCard colorClass="amber" label="Disk" icon="■"
            value={hardware.disk_free_gb} unit="GB"
            sub="Free space available" />
        )}
      </div>

      {/* RUNNING SERVERS TABLE */}
      <div className="section">
        <div className="section-head">
          <div>
            <div className="section-title">Running servers</div>
            <div className="section-subtitle">Active model instances and their endpoints</div>
          </div>
          <div className="section-actions">
            <button className="section-action" onClick={() => setTab('servers')}>View all</button>
          </div>
        </div>
        {running.length === 0 ? (
          <div style={{ padding: '20px 16px', textAlign: 'center', color: 'var(--ink-3)', fontSize: 13 }}>
            Nothing running. Pick a model on the Launch tab to start one.
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Engine</th>
                <th>Port</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {running.map((s) => (
                <tr key={s.id}>
                  <td>
                    <div className="model-name">{s.model}</div>
                  </td>
                  <td><span className="mono" style={{ fontSize: 12 }}>{s.engine}</span></td>
                  <td><span className="mono" style={{ fontSize: 12 }}>:{s.port}</span></td>
                  <td><StatusBadge level={s.healthy ? 'green' : 'amber'}>{s.healthy ? 'Running' : 'Starting'}</StatusBadge></td>
                  <td><button className="section-action" onClick={() => setTab('servers')}>Manage</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* ENGINES + VRAM BUDGET side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 18 }}>
        <div className="section" style={{ marginBottom: 0 }}>
          <div className="section-head">
            <div className="section-title">Engines</div>
            {hardware && (
              <StatusBadge level="green">
                {[hardware.engines.vllm_native, hardware.engines.vllm_docker, !!hardware.engines.llamacpp_path, hardware.engines.sglang].filter(Boolean).length} available
              </StatusBadge>
            )}
          </div>
          {hardware ? (
            <>
              <EngineRow ok={hardware.engines.vllm_native} name="vLLM"
                note={hardware.engines.vllm_native ? 'installed' : 'not installed'} />
              <EngineRow ok={hardware.engines.vllm_docker} name="vLLM (Docker)"
                note={hardware.engines.vllm_docker ? 'image found' : 'no Docker image'} />
              <EngineRow ok={!!hardware.engines.llamacpp_path} name="llama.cpp"
                note={hardware.engines.llamacpp_path || 'not found'} />
              <EngineRow ok={hardware.engines.sglang} name="SGLang"
                note={hardware.engines.sglang ? 'installed' : 'not installed'} />
            </>
          ) : <p style={{ padding: 16, color: 'var(--ink-3)' }}>…</p>}
        </div>

        <div className="section" style={{ marginBottom: 0 }}>
          <div className="section-head">
            <div className="section-title">VRAM Budget</div>
          </div>
          <div style={{ padding: '20px 16px', display: 'flex', alignItems: 'center', gap: 20 }}>
            <RingGauge percent={totalVramPct} color={totalVramPct > 80 ? 'var(--nogo)' : totalVramPct > 50 ? 'var(--caution)' : 'var(--go)'} />
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: '-0.02em' }}>
                {totalVramUsed} / {totalVram} GB
              </div>
              <div style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 2 }}>Used across both GPUs</div>
              <div style={{
                fontSize: 12, fontWeight: 600, marginTop: 4,
                color: totalVramPct > 80 ? 'var(--nogo)' : totalVramPct > 50 ? 'var(--caution)' : 'var(--go)',
              }}>
                {totalVramPct > 80 ? 'Running tight' : totalVramPct > 50 ? 'Moderate usage' : 'Comfortable headroom'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* OPEN WEBUI */}
      <OpenWebUIPanel notify={notify} />

      {/* TIP */}
      <div className="tip">
        <strong>How this works:</strong> download a model on Models, review the traffic-light settings on Launch, and press launch. Every setting is checked against your actual hardware — no guessing, no cryptic errors.
      </div>
    </>
  )
}
