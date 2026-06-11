import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { Badge, Led } from '../components.jsx'

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
        const r = await api.serverLogs(serverId, 300)
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
        <div className="stack panel inset" style={{ maxHeight: 260, overflowY: 'auto' }}>
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
        <button className="btn primary sm" disabled={busy || !input.trim()}>Send</button>
      </form>
    </div>
  )
}

function ServerCard({ server, refresh, notify }) {
  const [openLogs, setOpenLogs] = useState(!server.running)
  const [openChat, setOpenChat] = useState(false)
  const [stopping, setStopping] = useState(false)

  const level = server.running ? (server.healthy ? 'green' : 'yellow') : 'red'
  const stateText = server.running
    ? (server.healthy ? 'Running — ready for requests' : 'Starting up… (big models take 10-15 min)')
    : `Stopped${server.exit_code != null ? ` (exit code ${server.exit_code})` : ''}`

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
    notify('Endpoint address copied. Paste it into any app that supports the OpenAI API.')
  }

  return (
    <div className="panel stack">
      <div className="row between" style={{ alignItems: 'flex-start' }}>
        <div className="row" style={{ alignItems: 'flex-start' }}>
          <Led level={level} pulse={server.running && server.healthy} title={stateText} />
          <div>
            <h2 style={{ wordBreak: 'break-all' }}>{server.model}</h2>
            <div className="row small muted" style={{ marginTop: 4, flexWrap: 'wrap' }}>
              <Badge>{server.engine}</Badge>
              <span className="mono">port {server.port}</span>
              {server.running && <span>up {uptime(server.started_at)}</span>}
              <span>{stateText}</span>
            </div>
          </div>
        </div>
        <div className="row">
          {server.running ? (
            <button className="btn danger sm" onClick={stop} disabled={stopping}>
              {stopping ? 'Stopping…' : 'Stop'}
            </button>
          ) : (
            <button className="btn ghost sm" onClick={remove}>Remove from list</button>
          )}
        </div>
      </div>

      {server.failure_explanation && (
        <div className="flagrow why red" style={{ display: 'block' }}>
          <strong>What went wrong:</strong> {server.failure_explanation}
        </div>
      )}

      {server.running && server.healthy && (
        <div className="row small" style={{ flexWrap: 'wrap' }}>
          <span className="muted">Connect apps to:</span>
          <code>{server.endpoint}</code>
          <button className="btn sm" onClick={copyEndpoint}>Copy</button>
        </div>
      )}

      <div className="row">
        <button className="btn sm" onClick={() => setOpenLogs(!openLogs)}>
          {openLogs ? 'Hide log' : 'Show log'}
        </button>
        {server.running && server.healthy && (
          <button className="btn sm" onClick={() => setOpenChat(!openChat)}>
            {openChat ? 'Hide test chat' : 'Test chat'}
          </button>
        )}
      </div>

      {openLogs && <LogViewer serverId={server.id} />}
      {openChat && server.running && <ChatTest server={server} notify={notify} />}
    </div>
  )
}

export default function Servers({ servers, refresh, notify }) {
  return (
    <>
      <h1>Servers</h1>
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
