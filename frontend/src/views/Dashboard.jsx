import { Led } from '../components.jsx'

function GpuBar({ gpu }) {
  const usedMb = gpu.vram_total_mb - gpu.vram_free_mb
  const pct = Math.round((usedMb / gpu.vram_total_mb) * 100)
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

export default function Dashboard({ hardware, servers, goLaunch, setTab }) {
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

      <div className="panel inset small muted">
        <strong>How this works:</strong> pick or download a model on the <em>Models</em> tab, review the
        traffic-light settings on the <em>Launch</em> tab, and press launch. Every setting shows
        green when it's safe for your hardware, yellow when it deserves a look, and red when it
        would fail. You never have to guess a flag again.
      </div>
    </>
  )
}
