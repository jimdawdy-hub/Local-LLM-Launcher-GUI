// Thin fetch wrapper. Backend errors arrive as {"detail": "..."} from FastAPI.
async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch { /* keep fallback */ }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  hardware: () => request('/api/hardware'),
  about: () => request('/api/about'),
  models: () => request('/api/models'),
  search: (q) => request(`/api/models/search?q=${encodeURIComponent(q)}`),
  repoDetail: (repoId) => request(`/api/models/repo/${repoId}`),
  startDownload: (repoId, filename) =>
    request('/api/downloads', { method: 'POST', body: JSON.stringify({ repo_id: repoId, filename }) }),
  downloads: () => request('/api/downloads'),
  catalog: (engine) => request(`/api/catalog/${engine}`),
  advise: (engine, repoId, config) =>
    request('/api/advise', { method: 'POST', body: JSON.stringify({ engine, repo_id: repoId, config }) }),
  presets: (engine, repoId) =>
    request(`/api/presets?engine=${engine}&repo_id=${encodeURIComponent(repoId)}`),
  launch: (engineMode, repoId, config) =>
    request('/api/servers', { method: 'POST', body: JSON.stringify({ engine_mode: engineMode, repo_id: repoId, config }) }),
  servers: () => request('/api/servers'),
  serverLogs: (id, n = 200) => request(`/api/servers/${id}/logs?n=${n}`),
  stopServer: (id) => request(`/api/servers/${id}/stop`, { method: 'POST' }),
  removeServer: (id) => request(`/api/servers/${id}`, { method: 'DELETE' }),
  chat: (id, messages) =>
    request(`/api/servers/${id}/chat`, { method: 'POST', body: JSON.stringify({ messages }) }),
  settings: () => request('/api/settings'),
  saveSettings: (data) => request('/api/settings', { method: 'PUT', body: JSON.stringify(data) }),
  openwebui: () => request('/api/openwebui'),
  launchOpenwebui: () => request('/api/openwebui/launch', { method: 'POST' }),
  stopOpenwebui: () => request('/api/openwebui/stop', { method: 'POST' }),
}

export function gb(bytes) {
  if (bytes == null) return '?'
  return (bytes / 1024 ** 3).toFixed(1)
}
