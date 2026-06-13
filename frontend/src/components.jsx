// Small shared pieces: status LEDs (shape + color for color-blind users),
// badges, toasts, and the VRAM gauge.
import { useEffect } from 'react'

const LED_GLYPH = { green: '✓', yellow: '!', red: '✕', off: '' }

export function Led({ level = 'off', pulse = false, title }) {
  return (
    <span
      className={`led ${level} ${pulse ? 'pulse' : ''}`}
      title={title || level}
      aria-label={title || `status: ${level}`}
      role="img"
    >
      {LED_GLYPH[level]}
    </span>
  )
}

export function Badge({ level, children }) {
  return <span className={`badge ${level || ''}`}>{children}</span>
}

export function Toast({ toast, onDone }) {
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(onDone, toast.error ? 9000 : 4000)
    return () => clearTimeout(t)
  }, [toast, onDone])
  if (!toast) return null
  return (
    <div className={`toast ${toast.error ? 'error' : ''}`} role="status" aria-live="polite">
      {toast.message}
      <button className="toast-dismiss" onClick={onDone} aria-label="Dismiss notification"
        style={{ marginLeft: 12, background: 'none', border: 'none', color: 'inherit',
                 cursor: 'pointer', fontSize: 16, opacity: 0.7 }}>
        ✕
      </button>
    </div>
  )
}

// The signature element: a segmented memory "fuel gauge".
// Shows weights / conversation memory / working space vs. what's available.
export function VramGauge({ budget }) {
  if (!budget) return null
  const avail = budget.available_gb || 0
  const segs = [
    { key: 'weights', label: 'Model weights', value: budget.weights_gb },
    { key: 'kv', label: 'Conversation memory', value: budget.kv_cache_gb },
    { key: 'buffer', label: 'Working space', value: budget.working_buffer_gb },
  ]
  const needed = budget.needed_gb || 0
  const over = needed > avail && avail > 0
  const scale = avail > 0 ? Math.min(100 / Math.max(avail, needed), 100 / avail) : 0

  return (
    <div className="gauge">
      <div className="row between">
        <span className="small muted">Memory needed vs. available ({budget.basis})</span>
        <span className="mono small">
          {needed.toFixed(1)} / {avail.toFixed(1)} GB
        </span>
      </div>
      <div className="bar" role="img"
        aria-label={`Needs ${needed.toFixed(1)} of ${avail.toFixed(1)} gigabytes available`}>
        {segs.map((s) => (
          <div
            key={s.key}
            className={`seg ${s.key} ${over ? 'overcap' : ''}`}
            style={{ width: `${Math.max((s.value || 0) * scale, 0)}%` }}
            title={`${s.label}: ${s.value} GB`}
          />
        ))}
      </div>
      <div className="legend">
        {segs.map((s) => (
          <span key={s.key}>
            <span className={`chip seg ${s.key}`} style={over ? { background: 'var(--nogo)' } : {}} />
            {s.label} <span className="mono">{s.value} GB</span>
          </span>
        ))}
      </div>
    </div>
  )
}

export function FitVerdict({ overall }) {
  if (!overall) return null
  const word = { green: 'GO', yellow: 'CAUTION', red: 'NO-GO' }[overall.level]
  return (
    <div className="row" style={{ alignItems: 'flex-start', gap: 12 }}>
      <Led level={overall.level} />
      <div>
        <span className="mono" style={{ fontWeight: 700, letterSpacing: '0.1em' }}>{word}</span>
        <p className="small muted" style={{ marginTop: 3 }}>{overall.headline}</p>
        {(overall.details ?? []).map((d, i) => (
          <p key={i} className="small" style={{ marginTop: 5, color: 'var(--caution)' }}>{d}</p>
        ))}
      </div>
    </div>
  )
}
