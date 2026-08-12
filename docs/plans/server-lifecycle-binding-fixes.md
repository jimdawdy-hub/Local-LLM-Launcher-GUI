---
title: Local LLM Launcher server lifecycle & binding fixes - Plan
date: 2026-08-11
type: fix
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: legacy-requirements
execution:
  working_branch: upgrade
  build_lane: testing
  tdd: test-first
---

# Local LLM Launcher server lifecycle & binding fixes - Plan

## Goal Capsule

Make the Local LLM Launcher's server lifecycle correct and safe: a launched model server listens only on loopback by default (with an explicit LAN toggle), exposes only loopback hosts to the FastAPI reverse proxy, auto-increments its port correctly inside the real command, survives PID reuse after reboot, deletes the temporary HF-token env file it creates for Docker, and returns a clear error when a stop request cannot stop the server.

## Product Contract

### Summary

The launcher (GitHub issue tracker, Local-LLM-Launcher-GUI) has six failure modes in server launch/lifecycle:

1.  Launching a server with an occupied port auto-increments the port *after* the command (argv) is built, so the server still runs on the wrong port. (Issue 1)
2.  After a reboot, PID reuse makes a stopped server look alive; `stop` can terminate an unrelated process. (Issue 2)
3.  All launched servers and Open WebUI bind `0.0.0.0`, exposing model APIs on the LAN/wire without the user asking. (Issue 3)
4.  The API is vulnerable to DNS-rebinding attacks from other websites. (Issue 4)
5.  vLLM Docker launches write the credentials/HF token to a temporary `.env` file that is never deleted. (Issue 5)
6.  `POST /servers/{id}/stop` returns 404 even when the server exists but fails to stop, hiding the real failure. (Issue 6)

### Problem Frame

The frontend only records what it was told, and pytest is green, but the *servers actually spawned* disagree with what was recorded: wrong port in argv, stale PID, public bind, leaked token file, and error semantics that hide failures. This is a lifecycle-correctness (server side) defect set, not a UI defect.

### Requirements

**R1 (Issue 1) — Port auto-increment must land in the real command.** When `ServerManager.launch` auto-increments a busy port, the resolved port must be applied to the command (argv) the server actually receives. Non-negotiable observable: a launched server whose preferred port is occupied must be served on a different, actually-free port; the `Server` record's port, and the `--port`/`-p` argument in its argv, must both equal that resolved port.

**R2 (Issue 3) — Loopback by default.** vLLM native, vLLM docker, llama.cpp, and Open WebUI must bind `127.0.0.1` by default, not `0.0.0.0` (and not the any-interface docker default). Non-negotiable observable: the listening socket of a freshly launched server binds only loopback.

**R3 (Issue 3) — LAN toggle.** A single settings boolean (default off) exposes servers on `0.0.0.0` so features can reach them across the LAN. The setting must be persisted, returned in `GET /api/settings`, settable via `PUT/PATCH /api/settings`, and honored by every engine launch and Open WebUI launch. When off, behavior equals R2.

**R4 (Issue 4) — Host header protection.** The FastAPI app must reject requests whose Host header is not a loopback host (`127.0.0.1`, `localhost`). Non-negotiable observable: a request with `Host: evil.example` is rejected (4xx); normal loopback requests keep working.

**R5 (Issue 2) — PID identity.** A server revived from disk (or viewable across restarts of the GUI) whose PID is now recycled must be reported as *not running*, and `stop` must refuse to kill it. Non-negotiable observable: `is_running()` false and `stop()` no-op (or safe) for a dead/recycled PID; true running detection still works via process start time.

**R6 (Issue 5) — Docker env-file cleanup.** The temporary env file written for vLLM docker launches (HF token) must be deleted when the server stops, is removed from the registry, or fails to start. Non-negotiable observable: after stop/remove/failed-launch, the temp env file no longer exists on disk.

**R7 (Issue 6) — Stop error semantics.** `POST /servers/{id}/stop` returns 404 only when the id is unknown; a known server that fails to stop (process won't die) returns a distinct error (409/500 style), not 404.

### Acceptance Examples

- **A1 (R1):** Bind a port X; launch llama.cpp requesting X (or any model). `POST /api/servers` returns a server whose `port` is a free port ≠ X; `GET /api/servers` shows the same port; the spawned process's argv contains `--port <that free port>`.
- **A2 (R2):** Launch each of the 4 engines. `ss -tlnp` (or equivalent) shows the listening address as `127.0.0.1:<port>`, never `0.0.0.0:<port>` or `*:<port>` or `:::<port>`.
- **A3 (R3):** With `lan_access` off (default), A2 holds. Toggle `lan_access` on via `PUT /api/settings`; relaunch; listening address becomes `0.0.0.0:<port>`. Toggle back off; relaunch; loopback again.
- **A4 (R4):** `GET /api/about` with `Host: evil.example` → 4xx. `GET /api/about` with default loopback Host → 200.
- **A5 (R5):** Manually write a stale `servers.json` record pointing at a recycled PID (e.g. a tiny long-lived process started after reboot). `GET /api/servers` reports it not running; `POST .../stop` does not kill the innocent process; the innocent process is still alive afterward (check with `ps`).
- **A6 (R6):** Launch vLLM docker with an HF token configured; note the env file path in the record/argv; stop the server; assert the file is gone on disk.
- **A7 (R7):** `POST /api/servers/does-not-exist/stop` → 404. Launch a real server; make its process unkillable (e.g. SIGSTOP/`uninterruptible`); `POST .../stop` → non-404 error.

### Scope Boundaries

In scope: the six issues above, plus the LAN toggle that makes the default-loopback change usable across a LAN, plus the regression tests that lock each behavior in.

Out of scope (explicitly NOT fixed here):
- IPv6 binding behavior chosen by the engines (we only guarantee loopback vs any-interface on IPv4 as far as the engine CLI allows).
- UI features beyond the LAN toggle checkbox on the Settings screen.
- Issues numbered 7+ in the tracker.
- Firewall / OS-level network policy.
- The physical-collocation and shared-workspace concerns of other Freecase projects — this is the Local-LLM-Launcher-GUI repo only.

## Planning Contract

### Known Technical Details

**KTD1 (settled): the settings key is `lan_access`, a boolean, default `False`.**
Alternative (per-launch host flags) was rejected: it added surface area to the launch UI, duplicated the choice per launch, and did not match the "one switch" semantics the issue implied. Live binding decision in `api_launch` (and Open WebUI launch) from the current settings value, passed into each engine's `build` and `openwebui.launch`.

**KTD2 (settled): port resolution happens before `build_spec` reads config.**
In `ServerManager.launch`, resolve `config["port"]` (default from the engine, busy → increment) *before* calling `build_spec`, and pass the resolved port as part of config to build. Because `build` reads `config.get("port", ...)` and `build_args_and_env` emits `--port` only when the key is present, ensuring `config["port"]` is always set means the resolved port always lands in argv. Keep the existing `port_in_use`/`find_free_port` helpers.

**KTD3 (settled): PID identity via `psutil.Process(pid).create_time()`.**
Store `pid_create_time` on `LocalServer` and persist it in `to_record`/`from_record`. In `is_running()`/`stop()` when `self.process is None` (revived-from-record path), compare the live process's `create_time()` to the stored one; mismatch (or no such process) → not running / refuse kill. Legacy records without `pid_create_time` fall back to the current behavior (PID-only check), since we cannot recover more identity for them. `psutil` is already in `pyproject`.

**KTD4 (settled): `TrustedHostMiddleware(allowed_hosts=["127.0.0.1", "localhost"])`.**
Added to `create_app()` in `app.py`. Because the middleware rejects non-listed Hosts, the two `TestClient(create_app())` fixtures must pass `base_url="http://127.0.0.1"` (default is `http://testserver`, which would be rejected → all tests 403).

**KTD5 (settled): env-file path travels in the spec and lifecycle lives with the Server.**
`vllm_docker.build` already writes its env file via `tempfile.mkstemp(prefix="llml-env-", suffix=".env")` and references it in argv. Add `env_file` to the returned spec dict. `LocalServer` stores the path. Deletion happens in three places: (a) `LocalServer.stop` after the process dies; (b) `ServerManager.remove` for a server being discarded; (c) a failed launch path (start returned/raised before running). No other code owns this file.

**KTD6 (settled): stop keeps `ServerManager.stop -> bool`, route distinguishes cases.**
`api_server_stop` keeps 404 only when `servers.get(server_id) is None`. When the server exists, call `stop()`; on `False` return 409 with a message directing the user to the log. `ServerManager.stop` keeps its existing boolean contract; no registry-level exception type introduced.

**KTD7 (load-bearing code facts):**
- `catalog.defaults("vllm")["port"] = 8000`, `catalog.defaults("llamacpp")["port"] = 8080`; Open WebUI `DEFAULT_PORT = 3000` in `openwebui.py`.
- The flag catalogs have **no `host` flag** (only `port`, whose help text mentions localhost). Per KTD1 we do **not** add a `host` flag to the catalogs (it would surface in the Launch UI); we inject host into the engine argv directly in each builder.
- `hf_token` maps to `env_var: HF_TOKEN` in `flags_vllm.json`, so config `hf_token` becomes the docker env-file contents for vLLM docker (and `-e`/env for native).
- vLLM native `build_args_and_env` emits only keys present in config; so `config["port"]` set in KTD2 automatically yields `--port <resolved>`.
- Docker `-p HOST_PORT:CONTAINER_PORT` binds any-interface by default (`0.0.0.0`); prefixing the host address (`-p 127.0.0.1:PORT:8000`) restricts it.
- llama.cpp server binds loopback by default, but `--host` must be passed to reach `0.0.0.0` on LAN toggle; vLLM binds `0.0.0.0` by default, so `--host 127.0.0.1` is required to restrain it when LAN is off.
- `TestClient` default `base_url="http://testserver"` — incompatible with KTD4; both test fixtures change to `base_url="http://127.0.0.1"`.
- Two environments for verification: `.venv` created at repo root (`python3.14`, editable install, `105 passed` baseline with 1 StarletteDeprecationWarning about TestClient/httpx), and the frontend which has **no `node_modules`** yet (must `npm install` before `npm run build`; no commit of `node_modules`).

### High-Level Technical Design

The change is deliberately flat and localized; every fix touches exactly one seam:

1. **registry.py** — `launch()` resolves the final port into `config` before building the server (KTD2). `ServerManager.remove()` deletes a server's env file (KTD6/KTD5).
2. **config.py** — `DEFAULTS` gains `lan_access: False`; `Settings.update` iterates `DEFAULTS` keys so the new key is persisted automatically; no other change (roundtrip already proven in `test_settings_roundtrip_masks_token`).
3. **models.py (if settings models exist there) / anywhere SettingsUpdate is defined** — add `lan_access: bool = False` so PUT/PATCH accept it (KTD1). (*Author to confirm exact file: search `SettingsUpdate`/`SettingsPatch`.*)
4. **engines/base.py** — `LocalServer.__init__` picks up `pid_create_time` and `env_file`; `is_running`/`stop` use create-time identity when process is None; `to_record`/`from_record` persist both fields.
5. **engines/vllm_native.py** — `--host` from config (default loopback) added to argv when the key exists.
6. **engines/vllm_docker.py** — host-prefixed `-p` binding from config; env-file path recorded in spec (`env_file`).
7. **engines/llamacpp.py** — `--host` from config (default loopback) added when the key exists.
8. **openwebui.py** — `launch(host=...)` + `--host` passed to the serve command; default loopback.
9. **api.py** — `api_launch` injects `config["host"]` from settings (`0.0.0.0` if `lan_access` else `127.0.0.1`); openwebui launch passes the same; `api_server_stop` returns 404 only for unknown id, 409 when `stop()` is False.
10. **app.py** — `TrustedHostMiddleware` registered with loopback-only allowed hosts (KTD4).
11. **frontend/src/views/Settings.jsx** — LAN checkbox bound to `settings.lan_access`, included in the `save()` payload.
12. **frontend/src/api.js** — `saveSettings` already sends arbitrary bodies; verify it sends `lan_access`. If it whitelists keys, add it. (*Author to confirm*.)
13. **tests** across `test_registry.py`, `test_engines.py`, `test_openwebui.py`, `test_api.py` — new cases per Requirements; existing `TestClient` fixtures gain `base_url="http://127.0.0.1"`.

Data flow for host binding (LAN toggle):
`GET/PUT settings.lan_access` → `api_launch` reads `settings.data["lan_access"]` → `config["host"] = "0.0.0.0" | "127.0.0.1"` → engine `build` reads `config.get("host", "127.0.0.1")` → argv / docker `-p` carries the bind → frontend/settings UI toggles it.

### Sequencing

Safe order (each step keeps the suite green when its new tests pass):

1. **U1 (R1)** registry port resolution + regression tests. Self-contained.
2. **U2 (R5)** `LocalServer` pid_create_time identity + revive/stop tests. Touches base only.
3. **U3 (R2, R3)** host/default binding: config key + settings models + engine argv + api injection + openwebui + Settings.jsx. Larger, but each engine is a one-liner argv change with its own test.
4. **U4 (R6)** env-file lifecycle in docker builder, base, registry + tests.
5. **U5 (R4)** `TrustedHostMiddleware` + fix the two TestClient fixtures + host-rejection test.
6. **U6 (R7)** stop 404/409 semantics + tests.
7. **Build + full verify** (backend suite, frontend unit/build) and lint per repo conventions.

### Implementation Units

**U1 — Port resolution before build (R1).** `registry.py`: resolve config port into `config` prior to `build_spec`; tests bind an occupied port and assert resolved port + argv contain it.

**U2 — PID identity (R5).** `engines/base.py` + `LocalServer` record/from_record; tests: run a short-lived marker process, record its pid+create_time, poison a record with a recycled pid, assert not-running and stop-refused with the innocent process still alive.

**U3 — Loopback default + LAN toggle (R2, R3).** Config (`lan_access`), settings model, the 4 engine builders' host handling, `api_launch`/openwebui host injection, Settings.jsx checkbox; tests assert argv/`-p` per engine with LAN off and on.

**U4 — Env-file lifecycle (R6).** `vllm_docker.build` spec `env_file`; `LocalServer.env_file` + delete in stop/remove/failed-launch; tests: launch a docker server (or fake build) with token, stop, assert file gone.

**U5 — Host-header protection (R4).** `app.py` middleware; both TestClient fixtures `base_url`; test `Host: evil.example` → 4xx.

**U6 — Stop 404/409 (R7).** `api_server_stop` case split; tests: unknown → 404; unkillable known server → 409.

## Verification Contract

- Backend: `.venv/bin/python -m pytest -q  # baseline 105 passed`; must stay green and grow by the new cases.
- Frontend: `npm install && npm run build` must succeed; no `node_modules` committed.
- Manual smoke (optional per reviewer): launch each engine engine-mode, check listening address counts as loopback vs LAN toggle with `ss -tlnp`.
- Acceptance examples A1–A7 are each backed by an automated test.

## Definition of Done

- All of R1–R7 true and locked by regression tests (A1–A7 green).
- Suite green (`105 + new`), frontend builds, no unrelated cleanup in the diff.
- Change stays in the fix scope: settings (`lan_access`), the 4 engines, local server base, registry, API stop/launch, app middleware, Settings UI, and their tests.
- No new runtime dependency (psutil, already present).
- Handed over per lfg: work → simplify → review → apply → residual → browser tests → push/PR.