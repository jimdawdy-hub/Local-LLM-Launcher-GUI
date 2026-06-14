# Handover: Frontend Rebrand to OpsPulse Style

## Project
`/home/jim/Projects/Local-LLM-Launcher-GUI`

## Goal
Rebrand the frontend from the current "AI-generated dark theme" to match the OpsPulse SaaS dashboard aesthetic (reference: https://me.muz.li/orbix-studio/opspulse-ai-operations-compliance-saas-dashboard-design).

## Current State
- **Mockup ready**: `frontend/mockups/07-opspulse.html` — standalone HTML file showing the Dashboard view in the new style. This is the reference implementation to match.
- **No code changes yet** — the mockup is purely visual, not wired into the React app.
- **Working tree is clean** — all previous changes (v0.2.0) are committed and pushed.

## Design Language (from mockup)

### Colors (CSS variables to replace)
```
--bg: #f8f9fc              (light gray background, NOT dark)
--sidebar: #111827          (dark sidebar only)
--surface: #ffffff          (white cards)
--ink: #111827              (near-black text)
--ink-2: #4b5563            (secondary text)
--ink-3: #9ca3af            (muted text)
--ink-4: #e5e7eb            (borders)
--accent: #2563eb           (blue, used sparingly)
--accent-light: #eff6ff     (light blue bg)
--go: #059669               (green, slightly muted)
--go-bg: #ecfdf5
--caution: #d97706          (amber)
--caution-bg: #fffbeb
--nogo: #dc2626             (red)
--nogo-bg: #fef2f2
--border: #e5e7eb
```

### Typography
- **Font**: Plus Jakarta Sans (display + body) — NOT Space Grotesk, NOT IBM Plex Sans
- **Mono**: JetBrains Mono
- **Weights**: 400/500/600/700/800
- Load from Google Fonts

### Layout
- **Dark sidebar** (240px) with: logo, nav with badges, status footer
- **Light main area** with sticky top bar (breadcrumb + actions)
- **Metric cards**: 4-column grid, colored top border (3px), icon accent
- **Data tables**: striped headers, status badges with dots
- **Section cards**: white bg, subtle shadow, header with title+subtitle+actions
- **Ring/dial gauges**: SVG circles for VRAM budget (compliance-score style)

### Key Components to Restyle
1. **Sidebar (.rail)** — dark bg, logo block, nav with icons+badges, status footer
2. **Top bar** — new, sticky, with breadcrumb + action buttons
3. **Metric cards** — replace current GPU bars with OpsPulse-style cards
4. **VRAM gauge** — replace segmented bar with ring/dial SVG
5. **Section cards** — replace `.panel` with `.section` style (header+subtitle+actions)
6. **Data table** — for running servers (replace current list)
7. **Status badges** — dot + label pattern (green/amber/red/neutral)
8. **Buttons** — primary (blue), secondary (bordered), ghost
9. **Flag rows** — keep the traffic-light system but restyle to match table rows
10. **Toast** — restyle to match new surface/colors
11. **Modal** — restyle overlay + modal to match new surface

## Files to Modify

### CSS
- `frontend/src/theme.css` — complete rewrite of CSS variables and component styles

### React Components
- `frontend/src/components.jsx` — Led, Badge, Toast, VramGauge, FitVerdict
- `frontend/src/views/Dashboard.jsx` — metric cards, engine list, running servers, Open WebUI panel
- `frontend/src/views/Models.jsx` — model cards, search, download rows
- `frontend/src/views/Launch.jsx` — flag rows, presets, launch button, gauge
- `frontend/src/views/Servers.jsx` — server list, log viewer
- `frontend/src/views/Settings.jsx` — settings form

### New Components Needed
- `TopBar` — sticky header with breadcrumb + actions
- `MetricCard` — the OpsPulse-style stat card
- `RingGauge` — SVG circular progress indicator
- `DataTable` — table with styled headers and badge cells
- `StatusBadge` — dot + label badge component

## Implementation Order
1. **theme.css** — rewrite all CSS variables, add new component classes
2. **components.jsx** — update Led, Badge, Toast, VramGauge (→ RingGauge), FitVerdict
3. **Dashboard.jsx** — restyle to match mockup (metric cards, table, ring gauge)
4. **TopBar component** — extract from layout, add breadcrumb support
5. **App.jsx/layout** — wire TopBar, update frame structure
6. **Launch.jsx** — restyle flag rows, presets, launch button
7. **Models.jsx** — restyle model cards, search, downloads
8. **Servers.jsx** — restyle server list, logs
9. **Settings.jsx** — restyle form
10. **Modal + Toast** — restyle overlays
11. **Build & test** — `cd frontend && npm run build`, copy to static/
12. **Verify** — run launcher, check all views in browser

## Current Theme (to be replaced)
```css
--bg: #0e1420           (dark)
--panel: #161e2e        (dark panels)
--ink: #e8edf7          (light text on dark)
--action: #6e9bff       (blue)
--font-display: 'Space Grotesk'
--font-body: 'IBM Plex Sans'
```

## Sidebar Nav Structure (keep same tabs)
Dashboard, Models, Launch, Servers, Settings

## Status Footer in Sidebar
Show: active server count + VRAM usage (2.3 / 32 GB)

## Build Process
```bash
cd /home/jim/Projects/Local-LLM-Launcher-GUI/frontend
npm run build
# Output goes to src/local_llm_launcher/static/
```

## Testing
```bash
cd /home/jim/Projects/Local-LLM-Launcher-GUI
python3.12 -m local_llm_launcher.__main__ --no-browser
# Open http://localhost:8765
```

## Do NOT
- Push to GitHub (local only)
- Change any Python backend code
- Change API endpoints or data shapes
- Add new npm dependencies (use existing React + CSS)
- Use Space Grotesk, Inter, or IBM Plex Sans
- Use purple gradients or dark-on-dark panels
