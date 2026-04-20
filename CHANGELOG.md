# Changelog - IPTV-Player

All notable changes will be documented in this file.

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