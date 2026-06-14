import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { StatusBadge } from '../components.jsx'

function uptime(startedAt) {
  if (!startedAt) return ''
  const secs = Math.max(0, (Date.now() - new Date(startedAt).getTime()) / 1000)
  const h = Math.floor(secs / 3600)
  const m = Math.floor((secs % 3600) / 60)
  return h > 0 ? `${h}h ${m}m` : `${m}m`
}

function LogViewer({ serverId }) {
  const [lines, setLines] = useState([])
  const boxRef = useRef(null)
  const pinned = useRef(true)

  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const r = await api.serverLogs(serverId, 1000)
        if (alive) setLines(r?.lines ?? [])
      } catch { /* server may be gone */ }
    }
    poll()
    const t = setInterval(poll, 2000)
    return () => { alive = false; clearInterval(t) }
  }, [serverId])

  useEffect(() => {
    const box = boxRef.current
    if (box && pinned.current) box.scrollTop = box.scrollHeight
  }, [lines])

  return (
    <div className="logbox" ref={boxRef}
      onScroll={(e) => {
        const b = e.target
        pinned.current = b.scrollHeight - b.scrollTop - b.clientHeight < 40
      }}>
      {lines.length === 0 ? 'No log output yet…' : lines.join('\n')}
    </div>
  )
}

function ChatTest({ server, notify }) {
  const [input, setInput] = useState('')
  const [history, setHistory] = useState([])
  const [busy, setBusy] = useState(false)

  const send = async (e) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || busy) return
    const messages = [...history, { role: 'user', content: text }]
    setHistory(messages)
    setInput('')
    setBusy(true)
    try {
      const r = await api.chat(server.id, messages)
      setHistory([...messages, { role: 'assistant', content: r.content }])
    } catch (err) {
      notify(err.message, true)
      setHistory(messages.slice(0, -1))
      setInput(text)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="stack">
      {history.length > 0 && (
        <div className="panel inset stack" style={{ maxHeight: 260, overflowY: 'auto' }}>
          {history.map((m, i) => (
            <p key={i} className="small">
              <strong className={m.role === 'user' ? '' : 'muted'}>
                {m.role === 'user' ? 'You' : 'Model'}:
              </strong>{' '}
              {m.content}
            </p>
          ))}
          {busy && <p className="small muted">Model is thinking…</p>}
        </div>
      )}
      <form className="row" onSubmit={send}>
        <input style={{ flex: 1 }} value={input} disabled={busy}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Say something to test the model…" aria-label="Chat message" />
        <button className="btn btn-primary sm" disabled={busy || !input.trim()}>Send</button>
      </form>
    </div>
  )
}

function ServerCard({ server, refresh, notify }) {
  const [openLogs, setOpenLogs] = useState(!server.running)
  const [openChat, setOpenChat] = useState(false)
  const [stopping, setStopping] = useState(false)

  const level = server.running ? (server.healthy ? 'green' : 'amber') : 'red'
  const stateText = server.running
    ? (server.healthy ? 'Running' : 'Starting up…')
    : `Stopped${server.exit_code != null ? ` (exit ${server.exit_code})` : ''}`

  const stop = async () => {
    setStopping(true)
    try {
      await api.stopServer(server.id)
      notify('Server stopped.')
      refresh()
    } catch (err) {
      notify(err.message, true)
    } finally {
      setStopping(false)
    }
  }

  const remove = async () => {
    try {
      await api.removeServer(server.id)
      refresh()
    } catch (err) {
      notify(err.message, true)
    }
  }

  const copyEndpoint = () => {
    navigator.clipboard?.writeText(`${server.endpoint}`)
    notify('Endpoint address copied.')
  }

  return (
    <div className="section">
      <div className="section-head">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <StatusBadge level={level}>{stateText}</StatusBadge>
          <div>
            <div className="section-title" style={{ border: 'none', padding: 0, margin: 0, textTransform: 'none', letterSpacing: 0, fontSize: 14 }}>
              {server.model}
            </div>
            <div className="small muted" style={{ display: 'flex', gap: 8, marginTop: 2 }}>
              <span className="mono">{server.engine}</span>
              <span className="mono">port {server.port}</span>
              {server.running && <span>up {uptime(server.started_at)}</span>}
            </div>
          </div>
        </div>
        <div className="section-actions">
          {server.running ? (
            <button className="btn btn-danger sm" onClick={stop} disabled={stopping}>
              {stopping ? 'Stopping…' : 'Stop'}
            </button>
          ) : (
            <button className="btn btn-ghost sm" onClick={remove}>Remove</button>
          )}
        </div>
      </div>

      <div style={{ padding: '14px 20px' }} className="stack">
        {server.failure_explanation && (
          <div className="why red" style={{ borderLeft: '3px solid var(--nogo)', background: 'var(--nogo-bg)', padding: '8px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12.5 }}>
            <strong>What went wrong:</strong> {server.failure_explanation}
          </div>
        )}

        {server.running && server.healthy && (
          <div className="row small" style={{ flexWrap: 'wrap' }}>
            <span className="muted">Connect apps to:</span>
            <code className="mono">{server.endpoint}</code>
            <button className="btn btn-ghost sm" onClick={copyEndpoint}>Copy</button>
          </div>
        )}

        <div className="row">
          <button className="btn btn-ghost sm" onClick={() => setOpenLogs(!openLogs)}>
            {openLogs ? 'Hide log' : 'Show log'}
          </button>
          {server.running && server.healthy && (
            <button className="btn btn-ghost sm" onClick={() => setOpenChat(!openChat)}>
              {openChat ? 'Hide test chat' : 'Test chat'}
            </button>
          )}
        </div>

        {openLogs && (
          <>
            <LogViewer serverId={server.id} />
            <p className="small faint mono">Full log file: {server.log_path}</p>
          </>
        )}
        {openChat && server.running && <ChatTest server={server} notify={notify} />}
      </div>
    </div>
  )
}

export default function Servers({ servers, refresh, notify }) {
  return (
    <>
      {servers.length === 0 && (
        <div className="empty">
          Nothing here yet. Launch a model and it will appear with its log,
          status light, and a test chat.
        </div>
      )}
      {servers.map((s) => (
        <ServerCard key={s.id} server={s} refresh={refresh} notify={notify} />
      ))}
    </>
  )
}
