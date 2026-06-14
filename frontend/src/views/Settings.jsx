import { useEffect, useState } from 'react'
import { api } from '../api.js'

const LLAMA_INSTALL = {
  linux: `# Easiest: prebuilt binary via your package manager, or build with CUDA:
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
# binary lands at llama.cpp/build/bin/llama-server`,
  mac: `# Homebrew (Apple Silicon — Metal acceleration included):
brew install llama.cpp`,
}

export default function Settings({ hardware, notify }) {
  const [settings, setSettings] = useState(null)
  const [token, setToken] = useState('')
  const [folders, setFolders] = useState('')
  const [llamaPath, setLlamaPath] = useState('')
  const [about, setAbout] = useState(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.settings().then((s) => {
      setSettings(s)
      setToken(s.hf_token ?? '')
      setFolders((s.gguf_folders ?? []).join('\n'))
      setLlamaPath(s.llamacpp_path ?? '')
    }).catch(() => setSettings({}))
    api.about().then(setAbout).catch(() => setAbout(null))
  }, [])

  const save = async (e) => {
    e.preventDefault()
    setSaving(true)
    try {
      const updated = await api.saveSettings({
        hf_token: token || null,
        gguf_folders: folders.split('\n').map((f) => f.trim()).filter(Boolean),
        llamacpp_path: llamaPath || null,
      })
      setSettings(updated)
      setToken(updated.hf_token ?? '')
      notify('Settings saved.')
    } catch (err) {
      notify(err.message, true)
    } finally {
      setSaving(false)
    }
  }

  const isMac = !!hardware?.apple_silicon
  const llamaFound = !!hardware?.engines?.llamacpp_path

  return (
    <>
      <form className="section" onSubmit={save}>
        <div className="section-head">
          <div>
            <div className="section-title">Settings</div>
            <div className="section-subtitle">Configure tokens, paths, and preferences</div>
          </div>
          <button className="btn btn-primary" disabled={saving}>{saving ? 'Saving…' : 'Save settings'}</button>
        </div>

        <div style={{ padding: '14px 20px' }} className="stack">
          <div>
            <h3>Hugging Face access token</h3>
            <p className="small muted" style={{ margin: '4px 0 8px' }}>
              Only needed for "gated" models (like Meta's Llama) where you must accept a license
              first. Create one at{' '}
              <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                huggingface.co/settings/tokens
              </a>{' '}
              — a "read" token is enough. It is stored only on this computer.
            </p>
            <input type="password" value={token} onChange={(e) => setToken(e.target.value)}
              placeholder={settings?.hf_token_set ? 'Saved (hidden)' : 'hf_…'}
              style={{ width: '100%', maxWidth: 420 }} />
          </div>

          <div>
            <h3>Extra GGUF folders</h3>
            <p className="small muted" style={{ margin: '4px 0 8px' }}>
              If you keep .gguf model files outside the standard download cache, list those
              folders here (one per line) and they'll show up on the Models tab.
            </p>
            <textarea rows={3} value={folders} onChange={(e) => setFolders(e.target.value)}
              placeholder={'/home/you/models\n/mnt/storage/gguf'}
              style={{ width: '100%', maxWidth: 560, fontFamily: 'var(--font-mono)', fontSize: 12.5 }} />
          </div>

          <div>
            <h3>llama.cpp location</h3>
            <p className="small muted" style={{ margin: '4px 0 8px' }}>
              {llamaFound
                ? <>Found at <code className="mono">{hardware.engines.llamacpp_path}</code>. Set a path here only to use a different copy.</>
                : 'llama-server was not found automatically. If you installed it somewhere unusual, give the full path here.'}
            </p>
            <input value={llamaPath} onChange={(e) => setLlamaPath(e.target.value)}
              placeholder="/path/to/llama-server"
              style={{ width: '100%', maxWidth: 560, fontFamily: 'var(--font-mono)', fontSize: 12.5 }} />
          </div>
        </div>
      </form>

      {!llamaFound && (
        <div className="section">
          <div className="section-head">
            <div className="section-title">Installing llama.cpp</div>
          </div>
          <div style={{ padding: '14px 20px' }}>
            <p className="small muted" style={{ marginBottom: 10 }}>
              llama.cpp is the engine for GGUF models — the most beginner-friendly format.
              Install it with the commands below, then come back here.
            </p>
            <div className="logbox" style={{ height: 'auto', maxHeight: 200 }}>
              {isMac ? LLAMA_INSTALL.mac : LLAMA_INSTALL.linux}
            </div>
          </div>
        </div>
      )}

      <div className="section">
        <div className="section-head">
          <div className="section-title">About</div>
        </div>
        <div style={{ padding: '14px 20px' }}>
          <p className="small muted">
            Local-LLM-Launcher-GUI {about?.version && `v${about.version}`} — a friendly way to run
            large language models on your own computer.
          </p>
          {about?.credits?.map((c) => (
            <p key={c.url} className="small" style={{ padding: '3px 0' }}>
              <a href={c.url} target="_blank" rel="noreferrer">{c.name}</a>
              {c.note && <span className="muted"> — {c.note}</span>}
            </p>
          ))}
        </div>
      </div>
    </>
  )
}
