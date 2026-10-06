import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api.js'
import { StatusBadge, FitVerdict, RingGauge } from '../components.jsx'

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
        <input type="checkbox" aria-label={spec.label} checked={!!v}
          onChange={(e) => onChange(e.target.checked)} />
        {v ? 'On' : 'Off'}
      </label>
    )
  }
  if (spec.type === 'choice') {
    return (
      <select aria-label={spec.label} value={v ?? ''} onChange={(e) => onChange(e.target.value === '' ? null : e.target.value)}>
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
      <input aria-label={spec.label} type="number" min={spec.min} max={spec.max} step={spec.step || 1}
        value={v === '' ? '' : v}
        placeholder="(automatic)"
        onChange={(e) => onChange(e.target.value === '' ? null : parseInt(e.target.value, 10))} />
    )
  }
  return (
    <input aria-label={spec.label} type={spec.secret ? 'password' : 'text'} value={v ?? ''}
      placeholder="(not set)"
      onChange={(e) => onChange(e.target.value === '' ? null : e.target.value)} />
  )
}

function FlagRow({ spec, value, rating, onChange }) {
  const level = rating?.level ?? 'green'
  return (
    <div className="flagrow">
      <StatusBadge level={level === 'yellow' ? 'amber' : level}>{level === 'green' ? '✓' : level === 'yellow' ? '!' : '✕'}</StatusBadge>
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
  const [adviceError, setAdviceError] = useState(null)
  const [launching, setLaunching] = useState(false)
  const debounce = useRef(null)
  const initializedModel = useRef(null)

  const model = useMemo(() => models.find((m) => m.repo_id === repoId), [models, repoId])

  useEffect(() => {
    api.models().then((r) => {
      const list = r?.models ?? []
      setModels(list)
      if (!repoId && list.length > 0) setRepoId(initialModel?.repo_id ?? list[0].repo_id)
    }).catch(() => setModels([]))
  }, [])

  useEffect(() => {
    if (!model || !hardware || initializedModel.current === model.repo_id) return
    initializedModel.current = model.repo_id
    const mode = defaultEngineMode(model, hardware)
    setEngineMode(mode)
  }, [model, hardware])

  useEffect(() => {
    if (!model || !engineMode) return
    let active = true
    const eng = adviceEngine(engineMode)
    setCatalog(null)
    setPresets([])
    setConfig({})
    api.catalog(eng).then((result) => { if (active) setCatalog(result) }).catch(() => { if (active) setCatalog(null) })
    api.presets(eng, model.repo_id).then((r) => {
      if (!active) return
      const ps = r?.presets ?? []
      setPresets(ps)
      if (ps.length > 0) {
        setConfig({ ...ps[0].config })
        setActivePreset(ps[0].name)
      }
    }).catch(() => { if (active) setPresets([]) })
    return () => { active = false }
  }, [model, engineMode])

  useEffect(() => {
    if (!model || !engineMode) return
    let active = true
    let sequence = 0
    setAdvice(null)
    setAdviceError(null)
    const fetchAdvice = () => {
      const request = ++sequence
      api.advise(adviceEngine(engineMode), model.repo_id, config, engineMode)
        .then((result) => { if (active && request === sequence) { setAdvice(result); setAdviceError(null) } })
        .catch((error) => { if (active && request === sequence) { setAdvice(null); setAdviceError(error.message) } })
    }
    clearTimeout(debounce.current)
    debounce.current = setTimeout(fetchAdvice, 250)
    const t = setInterval(fetchAdvice, 8000)
    return () => { active = false; clearTimeout(debounce.current); clearInterval(t) }
  }, [model, engineMode, config])

  const setFlag = useCallback((key, value) => {
    setActivePreset(null)
    setConfig((c) => {
      const next = { ...c }
      if (key === "cpu_moe" && value) delete next.n_cpu_moe
      if (key === "n_cpu_moe" && value > 0) delete next.cpu_moe
      if (key === "split_mode" && value === "none") delete next.tensor_split
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
      <div className="empty">No models installed yet — grab one on the Models tab first.</div>
    )
  }

  const flags = (catalog?.flags ?? []).filter((f) =>
    !(engineMode === "vllm-docker" && f.key === "numactl_interleave") &&
    !(config.split_mode === "none" && f.key === "tensor_split") &&
    !(!config.use_mtp && f.key === "mtp_draft_max"))
  const grouped = ['essential', 'performance', 'api'].map((cat) => ({
    cat, items: flags.filter((f) => f.category === cat && !f.advanced),
  }))
  const advanced = flags.filter((f) => f.advanced)
  const engineMissing = engineMode && !engineAvailable(engineMode, hardware)
  const level = engineMissing || adviceError ? 'red' : advice?.overall?.level
  // Red from the memory estimate alone is advice, not a hard stop.
  const warned = level === 'red' && !engineMissing && !adviceError && advice?.overall?.override === true
  const ggufChoices = model?.gguf_files ?? []

  const budget = advice?.budget
  const budgetPct = budget?.available_gb > 0
    ? Math.round(((budget.needed_gb || 0) / budget.available_gb) * 100)
    : 0

  return (
    <>
      {/* MODEL & ENGINE PICKER */}
      <div className="section">
        <div className="section-head">
          <div>
            <div className="section-title">Configuration</div>
            <div className="section-subtitle">Select model, engine, and tune parameters</div>
          </div>
        </div>
        <div style={{ padding: '14px 20px' }} className="stack">
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
                  className={`btn ${activePreset === p.name ? 'btn-primary' : 'btn-ghost'} sm`}
                  title={p.description}
                  onClick={() => applyPreset(p)}>
                  {p.name}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="section" style={{ padding: '14px 20px' }}>
        <div className="small">NUMA nodes (groups of CPU cores with their own memory, found on multi-socket machines): {hardware?.numa?.node_count ?? 'unknown'}.
          {' '}numactl tool {hardware?.numa?.numactl_path ? 'installed' : 'not installed'}.
          {hardware?.numa?.node_count === 1 && ' With a single node, the NUMA settings have nothing to spread across and won\'t help.'}
          {' '}These settings are optional; machines with several nodes can benefit, but compare speeds before keeping one.</div>
        {engineMode === 'llamacpp' && <div className="small" style={{ marginTop: 8 }}>
          llama.cpp devices: {(hardware?.llama_devices ?? []).length
            ? hardware.llama_devices.map((d) => d.name).join(', ')
            : 'not detected; leave device selection unset or enter names from llama-server --list-devices.'}
          {(hardware?.llama_devices ?? []).length > 0 && <div className="row" style={{ flexWrap: 'wrap' }}>
            {hardware.llama_devices.map((d) => <label key={d.name} title={d.description} className="row">
              <input type="checkbox" checked={(config.device || '').split(',').includes(d.name)}
                onChange={(event) => {
                  const selected = (config.device || '').split(',').filter(Boolean)
                  setFlag('device', (event.target.checked ? [...selected, d.name] : selected.filter((name) => name !== d.name)).join(',') || null)
                }} />{d.name}
            </label>)}
          </div>}
        </div>}
      </div>
      {adviceError && <div className="section" role="alert" style={{ padding: '14px 20px', color: 'var(--nogo)' }}>{adviceError}</div>}
      {/* FIT VERDICT + VRAM */}
      {engineMissing && (
        <div className="section">
          <div style={{ padding: '14px 20px' }}>
            <FitVerdict overall={{
              level: 'red',
              headline: `${ENGINE_LABELS[engineMode]} is not installed on this computer. ` +
                (engineMode === 'llamacpp'
                  ? 'See the Settings tab for install instructions.'
                  : 'Install vLLM (pip install vllm) or pull the vllm/vllm-openai Docker image.'),
            }} />
          </div>
        </div>
      )}
      {!engineMissing && advice && (
        <div className="section">
          <div style={{ padding: '14px 20px' }} className="stack">
            <FitVerdict overall={advice.overall} />
            {budget?.available_gb != null && !budget.fit_unknown && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <RingGauge percent={budgetPct} color={budgetPct > 90 ? 'var(--nogo)' : budgetPct > 70 ? 'var(--caution)' : 'var(--go)'} />
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700 }}>{budget.needed_gb?.toFixed(1)} / {budget.available_gb?.toFixed(1)} GB</div>
                  <div className="small muted">Memory needed vs. available ({budget.basis})</div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* FLAGS */}
      <div className="section">
        <div style={{ padding: '14px 20px' }}>
          {grouped.map(({ cat, items }) => items.length > 0 && (
            <div key={cat} style={{ marginBottom: 14 }}>
              <div className="section-label">{CATEGORY_TITLES[cat]}</div>
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
      </div>

      {/* LAUNCH BAR */}
      <div className="section">
        <div className="section-head">
          <div className="small muted">
            {warned
              ? 'The memory estimate says this won\u2019t fit. You can still try; it may fail to load or crash.'
              : level === 'red'
              ? 'Fix the red items above before launching.'
              : level === 'yellow'
                ? 'You can launch, but read the yellow notes first.'
                : advice ? 'All clear for your hardware.' : 'Checking configuration…'}
          </div>
          <button
            className={`launchbtn ${warned ? 'warned' : level === 'yellow' ? 'caution' : level === 'red' ? 'nogo' : ''}`}
            disabled={launching || (level === 'red' && !warned) || !advice}
            onClick={launch}>
            {launching ? 'Launching…' : warned ? 'BAD IDEA – YOU HAVE BEEN WARNED' : level === 'yellow' ? 'Launch anyway' : 'Launch'}
          </button>
        </div>
      </div>
    </>
  )
}
