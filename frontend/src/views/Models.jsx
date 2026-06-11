import { useCallback, useEffect, useState } from 'react'
import { api, gb } from '../api.js'
import { Badge, Led } from '../components.jsx'

const FIT_TEXT = { green: 'Fits', yellow: 'Tight', red: "Won't fit" }

function InstalledCard({ model, goLaunch }) {
  return (
    <div className="modelcard">
      <Led level={model.fit} title={model.fit_headline} />
      <div style={{ flex: 1 }}>
        <div className="title">{model.repo_id}</div>
        <div className="meta small muted">
          <Badge>{model.format}</Badge>
          {model.quant && <Badge>{model.quant}</Badge>}
          <span className="mono">{model.size_gb} GB</span>
          <Badge level={model.fit}>{FIT_TEXT[model.fit]}</Badge>
        </div>
      </div>
      <button className="btn primary sm" onClick={() => goLaunch(model)}>Launch…</button>
    </div>
  )
}

function DownloadRow({ d }) {
  const level = d.status === 'error' ? 'red' : d.status === 'done' ? 'green' : 'yellow'
  return (
    <div className="row" style={{ padding: '7px 0' }}>
      <Led level={level} />
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
          {r.is_gguf && <Badge>GGUF</Badge>}
          {r.gated && <Badge level="yellow">license required</Badge>}
          {r.downloads != null && <span>{r.downloads.toLocaleString()} downloads</span>}
        </div>
      </div>
      <button className="btn sm" disabled={busy} onClick={() => onPick(r)}>Download…</button>
    </div>
  )
}

// Modal to choose a single GGUF quant file (or confirm a full safetensors repo).
function PickFileModal({ repo, onClose, onStart }) {
  if (!repo) return null
  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
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
                      <button className="btn sm primary" onClick={() => onStart(repo.repo_id, f.filename)}>
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
            <button className="btn primary" onClick={() => onStart(repo.repo_id, null)}>
              Download {gb(repo.safetensors_total_bytes)} GB
            </button>
          </>
        )}
        <div style={{ marginTop: 14 }}>
          <button className="btn ghost sm" onClick={onClose}>Cancel</button>
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

  const active = downloads.filter((d) => d.status !== 'done' || Date.now() === 0)

  return (
    <>
      <h1>Models</h1>

      <div className="panel">
        <h2 style={{ marginBottom: 6 }}>Get a new model</h2>
        <p className="small muted" style={{ marginBottom: 10 }}>
          Search huggingface.co — the public library where models are published. Tip: add
          “GGUF” to your search for llama.cpp models, or look for “AWQ” / “4bit” versions for
          vLLM on smaller GPUs.
        </p>
        <form className="row" onSubmit={doSearch}>
          <input
            style={{ flex: 1 }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. Qwen3 8B GGUF"
            aria-label="Search models"
          />
          <button className="btn primary" disabled={searching}>
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

      {active.length > 0 && (
        <div className="panel">
          <h2>Downloads</h2>
          {active.map((d) => <DownloadRow key={d.id} d={d} />)}
        </div>
      )}

      <div className="panel">
        <h2 style={{ marginBottom: 10 }}>Installed on this computer</h2>
        {models === null && <p className="muted">Scanning…</p>}
        {models?.length === 0 && (
          <div className="empty">
            No models yet. Search above to download your first one — a small GGUF model
            (e.g. “Qwen3 4B GGUF”) is a good start.
          </div>
        )}
        <div className="stack">
          {models?.map((m) => <InstalledCard key={m.repo_id + m.path} model={m} goLaunch={goLaunch} />)}
        </div>
      </div>

      <PickFileModal repo={pickRepo} onClose={() => setPickRepo(null)} onStart={startDownload} />
    </>
  )
}
