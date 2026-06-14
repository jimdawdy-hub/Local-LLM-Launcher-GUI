import { useCallback, useEffect, useRef, useState } from 'react'
import { api, gb } from '../api.js'
import { StatusBadge } from '../components.jsx'

const FIT_TEXT = { green: 'Fits', yellow: 'Tight', red: "Won't fit" }
const FIT_LEVEL = { green: 'green', yellow: 'amber', red: 'red' }

function InstalledCard({ model, goLaunch }) {
  return (
    <div className="modelcard">
      <StatusBadge level={FIT_LEVEL[model.fit] || 'neutral'}>{FIT_TEXT[model.fit] || 'Unknown'}</StatusBadge>
      <div style={{ flex: 1 }}>
        <div className="title">{model.repo_id}</div>
        <div className="meta small muted">
          <StatusBadge>{model.format}</StatusBadge>
          {model.quant && <StatusBadge>{model.quant}</StatusBadge>}
          <span className="mono">{model.size_gb} GB</span>
        </div>
      </div>
      <button className="btn btn-primary sm" onClick={() => goLaunch(model)}>Launch…</button>
    </div>
  )
}

function DownloadRow({ d }) {
  const level = d.status === 'error' ? 'red' : d.status === 'done' ? 'green' : 'amber'
  return (
    <div className="row" style={{ padding: '7px 0' }}>
      <StatusBadge level={level}>{d.status}</StatusBadge>
      <div style={{ flex: 1 }}>
        <span className="mono small">{d.repo_id}{d.filename ? ` · ${d.filename}` : ''}</span>
        {d.status === 'running' && (
          <div className="gpubar" style={{ marginTop: 4 }}>
            <div className="track"><div className="fill" style={{ width: `${d.percent ?? 0}%` }} /></div>
          </div>
        )}
        {d.error && <p className="small" style={{ color: 'var(--nogo)' }}>{d.error}</p>}
      </div>
      <span className="mono small muted">
        {d.status === 'running' ? `${gb(d.bytes_done)} / ${gb(d.bytes_total)} GB` : d.status}
      </span>
    </div>
  )
}

function SearchResult({ r, onPick, busy }) {
  return (
    <div className="modelcard">
      <div style={{ flex: 1 }}>
        <div className="title">{r.repo_id}</div>
        <div className="meta small muted">
          {r.is_gguf && <StatusBadge>GGUF</StatusBadge>}
          {r.gated && <StatusBadge level="amber">license required</StatusBadge>}
          {r.downloads != null && <span>{r.downloads.toLocaleString()} downloads</span>}
        </div>
      </div>
      <button className="btn btn-ghost sm" disabled={busy} onClick={() => onPick(r)}>Download…</button>
    </div>
  )
}

function PickFileModal({ repo, onClose, onStart }) {
  const modalRef = useRef(null)

  useEffect(() => {
    if (!repo) return
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKey)
    const el = modalRef.current
    if (el) {
      const first = el.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')
      if (first) first.focus()
    }
    return () => document.removeEventListener('keydown', handleKey)
  }, [repo, onClose])

  if (!repo) return null
  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" ref={modalRef} role="dialog" aria-modal="true"
        aria-label={`Download options for ${repo.repo_id}`}
        onClick={(e) => e.stopPropagation()}>
        <h2>{repo.repo_id}</h2>
        {repo.is_gguf ? (
          <>
            <p className="small muted" style={{ margin: '8px 0 12px' }}>
              GGUF models come in several compression levels (quantizations). Pick one file —
              Q4_K_M is the usual sweet spot between quality and size.
            </p>
            <table className="plain">
              <tbody>
                {repo.gguf_files.map((f) => (
                  <tr key={f.filename}>
                    <td className="mono small">{f.filename}</td>
                    <td className="mono small muted">{f.size_gb} GB</td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn btn-primary sm" onClick={() => onStart(repo.repo_id, f.filename)}>
                        Download
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <>
            <p className="small muted" style={{ margin: '8px 0 12px' }}>
              This downloads the full model ({gb(repo.safetensors_total_bytes)} GB) for use with vLLM.
            </p>
            <button className="btn btn-primary" onClick={() => onStart(repo.repo_id, null)}>
              Download {gb(repo.safetensors_total_bytes)} GB
            </button>
          </>
        )}
        <div style={{ marginTop: 14 }}>
          <button className="btn btn-ghost sm" onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  )
}

export default function Models({ goLaunch, notify }) {
  const [models, setModels] = useState(null)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [searching, setSearching] = useState(false)
  const [pickRepo, setPickRepo] = useState(null)
  const [downloads, setDownloads] = useState([])

  const refreshModels = useCallback(() => {
    api.models().then((r) => setModels(r?.models ?? [])).catch(() => setModels([]))
  }, [])

  useEffect(() => {
    refreshModels()
    const t = setInterval(async () => {
      try {
        const r = await api.downloads()
        const list = r?.downloads ?? []
        setDownloads(list)
        if (list.some((d) => d.status === 'done')) refreshModels()
      } catch { /* ignore poll errors */ }
    }, 2000)
    return () => clearInterval(t)
  }, [refreshModels])

  const doSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setSearching(true)
    try {
      const r = await api.search(query.trim())
      setResults(r?.results ?? [])
    } catch (err) {
      notify(err.message, true)
    } finally {
      setSearching(false)
    }
  }

  const pick = async (r) => {
    try {
      setPickRepo(await api.repoDetail(r.repo_id))
    } catch (err) {
      notify(err.message, true)
    }
  }

  const startDownload = async (repoId, filename) => {
    setPickRepo(null)
    try {
      await api.startDownload(repoId, filename)
      notify(`Download started: ${repoId}`)
    } catch (err) {
      notify(err.message, true)
    }
  }

  const active = downloads.filter((d) => d.status !== 'done')

  return (
    <>
      {/* SEARCH */}
      <div className="section">
        <div className="section-head">
          <div>
            <div className="section-title">Get a new model</div>
            <div className="section-subtitle">Search huggingface.co for models to download</div>
          </div>
        </div>
        <div style={{ padding: '14px 20px' }}>
          <form className="row" onSubmit={doSearch}>
            <input style={{ flex: 1 }} value={query} onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Qwen3 8B GGUF" aria-label="Search models" />
            <button className="btn btn-primary" disabled={searching}>
              {searching ? 'Searching…' : 'Search'}
            </button>
          </form>
          {results && (
            <div className="stack" style={{ marginTop: 12 }}>
              {results.length === 0 && <p className="muted small">No matches — try different words.</p>}
              {results.map((r) => <SearchResult key={r.repo_id} r={r} onPick={pick} />)}
            </div>
          )}
        </div>
      </div>

      {/* DOWNLOADS */}
      {active.length > 0 && (
        <div className="section">
          <div className="section-head">
            <div className="section-title">Downloads</div>
          </div>
          <div style={{ padding: '10px 16px' }}>
            {active.map((d) => <DownloadRow key={d.id} d={d} />)}
          </div>
        </div>
      )}

      {/* INSTALLED */}
      <div className="section">
        <div className="section-head">
          <div>
            <div className="section-title">Installed on this computer</div>
          </div>
        </div>
        <div style={{ padding: '14px 20px' }}>
          {models === null && <p className="muted">Scanning…</p>}
          {models?.length === 0 && (
            <div className="empty">
              No models yet. Search above to download your first one — a small GGUF model
              (e.g. "Qwen3 4B GGUF") is a good start.
            </div>
          )}
          <div className="stack">
            {models?.map((m) => <InstalledCard key={m.repo_id + m.path} model={m} goLaunch={goLaunch} />)}
          </div>
        </div>
      </div>

      <PickFileModal repo={pickRepo} onClose={() => setPickRepo(null)} onStart={startDownload} />
    </>
  )
}
