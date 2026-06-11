import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api.js'
import { Badge, FitVerdict, Led, VramGauge } from '../components.jsx'

const ENGINE_LABELS = {
  'vllm-native': 'vLLM',
  'vllm-docker': 'vLLM (Docker)',
  llamacpp: 'llama.cpp',
}
const CATEGORY_TITLES = {
  essential: 'Essential',
  performance: 'Performance & memory',
  api: 'Connection',
}

function adviceEngine(mode) {
  return mode === 'llamacpp' ? 'llamacpp' : 'vllm'
}

function defaultEngineMode(model, hardware) {
  if (!model) return null
  if (model.format === 'gguf') return 'llamacpp'
  if (!hardware) return 'vllm-native'
  if (hardware.apple_silicon || hardware.gpus.length === 0) return 'llamacpp'
  if (hardware.engines.vllm_native) return 'vllm-native'
  if (hardware.engines.vllm_docker) return 'vllm-docker'
  return 'vllm-native'
}

function engineAvailable(mode, hardware) {
  if (!hardware) return true
  if (mode === 'vllm-native') return hardware.engines.vllm_native
  if (mode === 'vllm-docker') return hardware.engines.vllm_docker
  return !!hardware.engines.llamacpp_path
}

function FlagControl({ spec, value, onChange }) {
  const v = value ?? spec.default ?? ''
  if (spec.type === 'bool') {
    return (
      <label className="row small" style={{ cursor: 'pointer' }}>
        <input type="checkbox" checked={!!v}
          onChange={(e) => onChange(e.target.checked)} />
        {v ? 'On' : 'Off'}
      </label>
    )
  }
  if (spec.type === 'choice') {
    return (
      <select value={v ?? ''} onChange={(e) => onChange(e.target.value === '' ? null : e.target.value)}>
        {spec.choices.map((c) => (
          <option key={String(c)} value={c ?? ''}>{c === null ? '(automatic)' : String(c)}</option>
        ))}
      </select>
    )
  }
  if (spec.type === 'float') {
    return (
      <span className="row">
        <input type="range" min={spec.min} max={spec.max} step={spec.step || 0.01}
          value={v || spec.min}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          aria-label={spec.label} />
        <span className="mono small" style={{ width: 44 }}>{(v || spec.min).toFixed(2)}</span>
      </span>
    )
  }
  if (spec.type === 'int') {
    return (
      <input type="number" min={spec.min} max={spec.max} step={spec.step || 1}
        value={v === '' ? '' : v}
        placeholder="(automatic)"
        onChange={(e) => onChange(e.target.value === '' ? null : parseInt(e.target.value, 10))} />
    )
  }
  return (
    <input type={spec.secret ? 'password' : 'text'} value={v ?? ''}
      placeholder="(not set)"
      onChange={(e) => onChange(e.target.value === '' ? null : e.target.value)} />
  )
}

function FlagRow({ spec, value, rating, onChange }) {
  const level = rating?.level ?? 'green'
  return (
    <div className="flagrow">
      <Led level={level} title={rating?.message || 'Looks fine for your hardware'} />
      <div>
        <div className="name">{spec.label}</div>
        {spec.flag && <div className="flagname">{spec.flag}</div>}
      </div>
      <div className="control">
        <FlagControl spec={spec} value={value} onChange={onChange} />
        <p className="help">{spec.help}</p>
        {rating?.message && (
          <div className={`why ${level}`}>{rating.message}</div>
        )}
      </div>
    </div>
  )
}

export default function Launch({ hardware, initialModel, notify, onLaunched }) {
  const [models, setModels] = useState([])
  const [repoId, setRepoId] = useState(initialModel?.repo_id ?? '')
  const [engineMode, setEngineMode] = useState(null)
  const [catalog, setCatalog] = useState(null)
  const [config, setConfig] = useState({})
  const [presets, setPresets] = useState([])
  const [activePreset, setActivePreset] = useState(null)
  const [advice, setAdvice] = useState(null)
  const [launching, setLaunching] = useState(false)
  const debounce = useRef(null)

  const model = useMemo(() => models.find((m) => m.repo_id === repoId), [models, repoId])

  useEffect(() => {
    api.models().then((r) => {
      const list = r?.models ?? []
      setModels(list)
      if (!repoId && list.length > 0) setRepoId(initialModel?.repo_id ?? list[0].repo_id)
    }).catch(() => setModels([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // When the model changes, pick the right engine and reset config from the Safe preset.
  useEffect(() => {
    if (!model) return
    const mode = defaultEngineMode(model, hardware)
    setEngineMode(mode)
  }, [model, hardware])

  useEffect(() => {
    if (!model || !engineMode) return
    const eng = adviceEngine(engineMode)
    api.catalog(eng).then(setCatalog).catch(() => setCatalog(null))
    api.presets(eng, model.repo_id).then((r) => {
      const ps = r?.presets ?? []
      setPresets(ps)
      if (ps.length > 0) {
        setConfig({ ...ps[0].config })
        setActivePreset(ps[0].name)
      }
    }).catch(() => setPresets([]))
  }, [model, engineMode])

  // Debounced live advice as the config changes, then re-checked every 8s so
  // the free-GPU-memory numbers track what your desktop is actually using.
  useEffect(() => {
    if (!model || !engineMode) return
    const fetchAdvice = () => {
      api.advise(adviceEngine(engineMode), model.repo_id, config)
        .then(setAdvice)
        .catch(() => setAdvice(null))
    }
    clearTimeout(debounce.current)
    debounce.current = setTimeout(fetchAdvice, 250)
    const t = setInterval(fetchAdvice, 8000)
    return () => { clearTimeout(debounce.current); clearInterval(t) }
  }, [model, engineMode, config])

  const setFlag = useCallback((key, value) => {
    setActivePreset(null)
    setConfig((c) => {
      const next = { ...c }
      if (value === null || value === undefined) delete next[key]
      else next[key] = value
      return next
    })
  }, [])

  const applyPreset = (p) => {
    setConfig({ ...p.config })
    setActivePreset(p.name)
  }

  const launch = async () => {
    setLaunching(true)
    try {
      await api.launch(engineMode, model.repo_id, config)
      notify(`Launching ${model.repo_id}. Big models can take 10-15 minutes to load — silence in the log is normal.`)
      onLaunched()
    } catch (err) {
      notify(err.message, true)
    } finally {
      setLaunching(false)
    }
  }

  if (models.length === 0) {
    return (
      <>
        <h1>Launch</h1>
        <div className="empty">No models installed yet — grab one on the Models tab first.</div>
      </>
    )
  }

  const flags = catalog?.flags ?? []
  const grouped = ['essential', 'performance', 'api'].map((cat) => ({
    cat, items: flags.filter((f) => f.category === cat && !f.advanced),
  }))
  const advanced = flags.filter((f) => f.advanced)
  const engineMissing = engineMode && !engineAvailable(engineMode, hardware)
  const level = engineMissing ? 'red' : advice?.overall?.level
  const ggufChoices = model?.gguf_files ?? []

  return (
    <>
      <h1>Launch</h1>

      <div className="panel stack">
        <div className="grid2">
          <label className="stack" style={{ gap: 4 }}>
            <span className="small muted">Model</span>
            <select value={repoId} onChange={(e) => setRepoId(e.target.value)}>
              {models.map((m) => (
                <option key={m.repo_id + m.path} value={m.repo_id}>
                  {m.repo_id} ({m.size_gb} GB{m.quant ? `, ${m.quant}` : ''})
                </option>
              ))}
            </select>
          </label>
          <label className="stack" style={{ gap: 4 }}>
            <span className="small muted">Engine</span>
            <select value={engineMode ?? ''} onChange={(e) => setEngineMode(e.target.value)}>
              {Object.entries(ENGINE_LABELS).map(([mode, label]) => (
                <option key={mode} value={mode} disabled={!engineAvailable(mode, hardware)}>
                  {label}{!engineAvailable(mode, hardware) ? ' — not installed' : ''}
                </option>
              ))}
            </select>
          </label>
        </div>

        {engineMode === 'llamacpp' && ggufChoices.length > 1 && (
          <label className="stack" style={{ gap: 4 }}>
            <span className="small muted">Which file (compression level)</span>
            <select value={config.gguf_file ?? ggufChoices[0].filename}
              onChange={(e) => setFlag('gguf_file', e.target.value)}>
              {ggufChoices.map((f) => (
                <option key={f.filename} value={f.filename}>
                  {f.filename} ({f.size_gb} GB)
                </option>
              ))}
            </select>
          </label>
        )}

        {presets.length > 0 && (
          <div className="row" style={{ flexWrap: 'wrap' }}>
            <span className="small muted">Presets:</span>
            {presets.map((p) => (
              <button key={p.name}
                className={`btn sm ${activePreset === p.name ? 'primary' : ''}`}
                title={p.description}
                onClick={() => applyPreset(p)}>
                {p.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {engineMissing && (
        <div className="panel stack" aria-live="polite">
          <FitVerdict overall={{
            level: 'red',
            headline: `${ENGINE_LABELS[engineMode]} is not installed on this computer. ` +
              (engineMode === 'llamacpp'
                ? 'See the Settings tab for install instructions.'
                : 'Install vLLM (pip install vllm) or pull the vllm/vllm-openai Docker image.'),
          }} />
        </div>
      )}
      {!engineMissing && advice && (
        <div className="panel stack" aria-live="polite">
          <FitVerdict overall={advice.overall} />
          {advice.budget?.available_gb != null && <VramGauge budget={advice.budget} />}
        </div>
      )}

      <div className="panel">
        {grouped.map(({ cat, items }) => items.length > 0 && (
          <div key={cat} style={{ marginBottom: 14 }}>
            <div className="section-title">{CATEGORY_TITLES[cat]}</div>
            {items.map((spec) => (
              <FlagRow key={spec.key} spec={spec}
                value={config[spec.key]}
                rating={advice?.flags?.[spec.key]}
                onChange={(v) => setFlag(spec.key, v)} />
            ))}
          </div>
        ))}
        {advanced.length > 0 && (
          <details className="advanced">
            <summary>Advanced settings — most people never need these</summary>
            {advanced.map((spec) => (
              <FlagRow key={spec.key} spec={spec}
                value={config[spec.key]}
                rating={advice?.flags?.[spec.key]}
                onChange={(v) => setFlag(spec.key, v)} />
            ))}
          </details>
        )}
      </div>

      <div className="panel row between">
        <div className="small muted">
          {level === 'red'
            ? 'Fix the red items above before launching.'
            : level === 'yellow'
              ? 'You can launch, but read the yellow notes first.'
              : 'All clear for your hardware.'}
        </div>
        <button
          className={`launchbtn ${level === 'yellow' ? 'caution' : level === 'red' ? 'nogo' : ''}`}
          disabled={launching || level === 'red'}
          onClick={launch}>
          {launching ? 'Launching…' : level === 'yellow' ? 'Launch anyway' : 'Launch'}
        </button>
      </div>
    </>
  )
}
