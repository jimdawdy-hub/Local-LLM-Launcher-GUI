import { useEffect } from 'react'

export function StatusBadge({ level = 'neutral', children }) {
  return (
    <span className={`badge ${level}`}>
      <span className="badge-dot" />
      {children}
    </span>
  )
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

export function RingGauge({ percent = 0, color = 'var(--go)', label, size = 48 }) {
  const r = (size - 8) / 2
  const circ = 2 * Math.PI * r
  const offset = circ - (Math.min(percent, 100) / 100) * circ
  return (
    <div className="ring-wrap" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle className="ring-bg" cx={size/2} cy={size/2} r={r} />
        <circle className="ring-fill" cx={size/2} cy={size/2} r={r}
          stroke={color} strokeDasharray={circ} strokeDashoffset={offset} />
      </svg>
      <div className="ring-label" style={{ color }}>{label ?? `${Math.round(percent)}%`}</div>
    </div>
  )
}

export function MetricCard({ colorClass, label, icon, value, unit, sub, change, changeDir }) {
  return (
    <div className={`metric-card ${colorClass || ''}`}>
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        {icon && <span className={`metric-icon ${colorClass === 'green' ? 'green' : colorClass === 'amber' ? 'amber' : 'blue'}`}>{icon}</span>}
      </div>
      <div className="metric-value">
        {value} {unit && <span style={{ fontSize: 14, fontWeight: 500, color: 'var(--ink-3)' }}>{unit}</span>}
      </div>
      {sub && <div className="metric-sub">{sub}</div>}
      {change && (
        <div className="metric-sub">
          <span className={`metric-change ${changeDir || 'up'}`}>{change}</span>
        </div>
      )}
    </div>
  )
}

export function FitVerdict({ overall }) {
  if (!overall) return null
  const word = { green: 'GO', yellow: 'CAUTION', red: 'NO-GO' }[overall.level]
  return (
    <div className="row" style={{ alignItems: 'flex-start', gap: 12 }}>
      <StatusBadge level={overall.level === 'yellow' ? 'amber' : overall.level}>{word}</StatusBadge>
      <div>
        <p className="small muted">{overall.headline}</p>
        {(overall.details ?? []).map((d, i) => (
          <p key={i} className="small" style={{ marginTop: 5, color: 'var(--caution)' }}>{d}</p>
        ))}
      </div>
    </div>
  )
}

export function TopBar({ title, breadcrumb, children }) {
  return (
    <div className="topbar">
      <div className="topbar-left">
        <div>
          <div className="topbar-title">{title}</div>
          {breadcrumb && <div className="topbar-breadcrumb">{breadcrumb}</div>}
        </div>
      </div>
      {children && <div className="topbar-right">{children}</div>}
    </div>
  )
}

export function ThemeToggle({ theme, onToggle }) {
  return (
    <button className="theme-toggle" onClick={onToggle} aria-label="Toggle theme" title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}>
      {theme === 'light' ? '🌙' : '☀️'}
    </button>
  )
}
