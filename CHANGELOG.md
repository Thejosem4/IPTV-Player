# Changelog - IPTV-Player

All notable changes will be documented in this file.

## [7.3] - 2026-11-07

### Added
- **In-Memory HLS Ring Buffer**: `handle_hls_file()` now caches the last 12 `.ts` segments in an `OrderedDict` per stream. Cache hits eliminate disk I/O entirely, reducing segment serve latency from 15–45ms to <0.5ms.
- **Real Network Telemetry in AI**: `ThrottleDetector` now has `record_latency(url, ms)`, `get_latency(url)`, `get_speed(url)`, and `_probe_domain_latency(domain)`. RTT is measured via TCP socket connect; download speed tracked as EMA per domain. Replaces the three hardcoded `latency=0` callsites that made the AI blind to real network conditions.
- **Range / Seeking Support for VOD**: `proxy_parallel_download()` now parses the browser's `Range: bytes=X-Y` header and responds with `206 Partial Content` + `Content-Range`. Seeking in VOD streams no longer restarts from byte 0.
- **Auto-Activation of Parallel VOD**: Parallel multi-connection download is now enabled automatically for files >2MB that support HTTP Range. No longer requires the `?parallel=true` query parameter that the frontend never sent.

### Changed
- **`async_vod_downloader.py` fully rewritten**: `PieceBuffer` class removed (was accumulating GB in RAM). New architecture uses a bounded `queue.Queue(maxsize=16)` as a sliding window between async producer and sync writer. Max RAM per stream: ~8MB regardless of file size.
- **`download_and_stream()` new signature**: Now keyword-only: `(url, wfile, start_byte=0, end_byte=None, content_length=None, num_conn=8, ai_headers_func=None, ai_jitter_func=None, chunk_size=…)`. Full Range support for seeks.
- **`predict_optimal_config()` signature**: Parameters `latency` and `speed_kbps` now default to `None` and are auto-resolved from `throttle_detector` telemetry when not explicitly provided.
- **HLS segment wait granularity**: Poll interval reduced from `sleep(0.1)×30` to `sleep(0.05)×60` (same 3s window, finer response to late segments).
- **Small range requests (<2MB) delegated**: Moov atom / metadata fetches in VOD are routed through `proxy_range_request()` to avoid parallel-download overhead.
- **Server banner**: Updated to v7.3.

### Fixed
- **AI always received `latency=0`**: Three separate callsites in `iptv-server.py` hardcoded `latency=0`; replaced with auto-resolution via `throttle_detector.get_latency(url)`.
- **`uuid` import scope**: `import uuid` moved to module-level to prevent potential `NameError` in async paths.
- **Hardcoded `C:/IPTV_Cache/` paths**: Two occurrences replaced with `os.path.join(CACHE_DIR, …)` and `tempfile.gettempdir()` for cross-platform correctness.

---

## [7.2] - 2026-06-06

### Added
- **Async VOD Engine**: New `backend/async_vod_downloader.py` module using `aiohttp` for non-blocking parallel downloads. Streams bytes directly to the client socket (`wfile`) without buffering entire files in RAM.
- **Connection Pooling**: `aiohttp.TCPConnector` with dynamic limits and 60s keepalive for sustained high-throughput downloads.
- **CLAUDE.md**: Local development context file (excluded from Git) for AI assistant orientation.

### Changed
- **ThrottleDetector `MAX_PIECES`**: Increased from 5 to 32, enabling massively parallel chunk downloads for VOD content.
- **Jitter Delay**: Reduced base delay from 80ms to 20ms between pieces for faster async pipeline startup.
- **VOD Download Pipeline**: `proxy_parallel_download` in `iptv-server.py` now delegates to `async_vod_downloader.download_and_stream()` instead of blocking `urllib` threads.
- **Scheduler Fix**: `ai_scheduler.py` now delegates playlist updates to `refresh_one_playlist()`, ensuring disk cache (`.json.gz`) and SSE notifications are always synchronized.
- **README.md**: Rewritten for v7.0 architecture, explaining privacy-first local AI training and async streaming.

### Fixed
- **Stale Playlist Cache**: Scheduler was downloading M3U but not saving to disk cache or notifying the frontend via SSE. Now uses the unified server pipeline.
- **UTF-8 Encoding**: Cleaned corrupted emoji/characters across `iptv-server.py`, `iptv_ai_core.py`, and `iptv-player.html` that caused rendering issues on some systems.
- **Code Block Formatting**: Fixed broken markdown fenced code blocks in `README.md`.
- **Hardcoded Cache Paths**: Replaced non-existent `C:/IPTV_Cache/` in `/api/ai-stats` and ABR stream dirs with dynamic paths.
- **Dependency Drift & Version Fixes**: Cleaned phantom `flask` dependency, fixed invalid `scikit-learn` version, and explicitly declared `aiohttp`, `lightgbm`, `requests`, and `onnxruntime` in `requirements.txt`.
- **CDN Modernization**: Upgraded `hls.js` to 1.5.17 for improved buffer and memory management, and pinned `Chart.js` 4.4.7.

### Security
- **`.gitignore` hardened**: Added `CLAUDE.md`, `*.zip`, `.planning/`, and `listas*.txt` to prevent leaking local AI context, IPTV credentials, and large backup archives to the public repository.

---

## [7.1] - 2026-04-19

### Fixed
- **Static file serving**: Server now `chdir`s to `frontend/` before serving, fixing 404 on `iptv-player.html`
- **Cache corruption handling**: `load_cache_gz()` now has try/except with fallback to legacy `.json` file if gzip is corrupted
- **Atomic cache writes**: `save_cache_gz()` now writes to `.tmp` then uses `os.replace()` to prevent truncated `.json.gz` files
- **M3U download resilience**: `download_m3u()` now reads in 512KB chunks with total timeout of 180s and progress reporting every 5MB

### Changed
- **Download timeout**: Socket-level timeout reduced from 120s to 30s per chunk; total download timeout is now 180s (3 min) controlled by code logic
- **Download progress**: Added `📥 Descargando: X MB @ Y KB/s` output during M3U download

### Documentation
- Added this CHANGELOG.md
- README updated to reflect v7.1

---

## [7.0] - 2026-04-?? (Initial Release)

### Added
- AI-Powered Streaming Engine with Evolutionary Brain
- Neural prediction for stream quality and buffer optimization
- HLS proxy with FFmpeg and automatic stream restart
- M3U playlist parser with gzip compression
- AI scheduler for daily maintenance and model evolution
- AI decision logging to `logs/ai_decisions.log`
- Stream watchdog that restarts dead streams automatically
- Stream sharing (multiple clients on one FFmpeg process)
- VOD HLS streaming with transcoding support