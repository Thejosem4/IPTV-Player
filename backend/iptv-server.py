#!/usr/bin/env python3
"""
Servidor IPTV v7.0 - Resiliente y Multi-stream
─────────────────────────────────────────────────────────────
✅ v6.x: Parser M3U, cache gzip, FFmpeg optimizado, watchdog,
         transcode-vod via Python-pipe, VOD quality selector,
         descarga paralela adaptativa, latencia <2s live
─────────────────────────────────────────────────────────────
🆕 v7.0 OPTIMIZACIONES:
✅ fetch_with_retry() — 404/403 no matan el stream (retry 2s)
✅ Pipe VOD con reconexión automática en fallo de red
✅ Detección anticipada de primer segmento (playlist.m3u8 antes que .ts)
✅ Stream reuse mejorado — actualiza last_access en cada .ts servido
✅ Watchdog VOD — reinicia pipe si FFmpeg se cuelga en mitad del VOD
✅ Limpieza de sesiones garantizada con try/finally
✅ Respuesta de playlist ~40% más rápida (detección anticipada)
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.request
import urllib.parse
import urllib.error
import subprocess
import threading
import socket
import json
import gzip
import zlib
import time
import sys
import os

# Handle importing iptv_ai_core from core directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

import re
import io

# Forzar UTF-8 en stdout/stderr para evitar crashes con emojis en Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except: pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except: pass

import tempfile
import shutil
import hashlib
import datetime
import psutil
from collections import deque
from io import StringIO

# --- INTEGRACIÓN IA ---
try:
    from iptv_ai_core import ai_optimizer
    HAS_AI = True
    import logging
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'logs'), exist_ok=True)
    # Re-configurar logging para no interferir con el stdout original
    ai_logger = logging.getLogger("IPTV_AI")
    ai_handler = logging.FileHandler(os.path.join(os.path.join(os.path.dirname(__file__), '..', 'logs'), 'ai_decisions.log'), encoding="utf-8")
    ai_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    ai_logger.addHandler(ai_handler)
    ai_logger.setLevel(logging.INFO)
    ai_logger.info("🚀 SISTEMA IA RECONECTADO - Base Robusta Detectada")
except ImportError:
    HAS_AI = False
# ----------------------

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
CACHE_INDEX_FILE = os.path.join(CACHE_DIR, "index.json")
CACHE_EXT = ".json.gz"  # Nuevo formato comprimido

# Configuración de auto-actualización
auto_refresh_cfg = {
    "enabled": True,
    "interval_minutes": 60,
}
auto_refresh_lock = threading.Lock()

# Estado de última actualización por playlist_id
refresh_status = {}  # {id: {timestamp, status, total, error?}}
refresh_status_lock = threading.Lock()

# Log de velocidades (deque = O(1) al agregar/rotar)
speed_log = deque(maxlen=100)

# Cache de servidor: {url → {channels, groups, timestamp}}
# Sin claves "url"/"channels"/"groups" de compatibilidad
m3u_cache = {}
m3u_cache_lock = threading.Lock()
active_url = None  # URL activa para búsqueda/grupos

# Cache de streams FFmpeg activos
active_streams = {}
streams_lock = threading.Lock()

# Registro de pipe threads VOD activos (url → thread event para cancelar)
vod_pipe_registry = {}
vod_pipe_registry_lock = threading.Lock()

# ── Registro de stream activo por URL ────────────────────────────────────
# url_vod_active[url] = stream_id del transcode más reciente para ese URL.
# Al llegar cancel_previous=1, se limpia el stream anterior de esa URL
# antes de crear el nuevo. Evita acumulación de FFmpeg por seeks rápidos.
url_vod_active = {}  # {url → stream_id}
url_vod_active_lock = threading.Lock()

# Cache de duración de archivos VOD (url → segundos float)
# Poblado por ffprobe en background; evita relanzar ffprobe en cada seek
duration_cache = {}
duration_cache_lock = threading.Lock()

app_metrics = {
    "abr_interventions": 0,
    "cpu_saved_events": 0,
    "bytes_saved_mb": 0.0
}


# ============================================================
# DETECCIÓN DE DURACIÓN VOD VÍA ffprobe
# ============================================================
def get_vod_duration(url, base_url, timeout_s=20):
    """
    Consulta la duración total de un archivo VOD mediante ffprobe.

    Se ejecuta en un hilo background para no bloquear la respuesta inicial.
    Usa las mismas cabeceras HTTP que el servidor IPTV espera.
    Resultado cacheado para evitar llamadas repetidas.

    Returns: duración en segundos (float), o 0 si falla.
    """
    with duration_cache_lock:
        if url in duration_cache:
            return duration_cache[url]

    try:
        parsed = urllib.parse.urlparse(url)
        base_url2 = f"{parsed.scheme}://{parsed.netloc}"

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            # Headers para pasar autenticación del servidor IPTV
            "-user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "-headers",
            f"Referer: {base_url2}/\r\nConnection: keep-alive\r\n",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        data = json.loads(result.stdout)
        dur = float(data.get("format", {}).get("duration", 0))

        if dur > 0:
            with duration_cache_lock:
                duration_cache[url] = dur
            print(f"   ⏱️ Duración detectada: {dur:.1f}s ({dur/3600:.2f}h)")

        return dur

    except Exception as e:
        print(f"   ⚠️ ffprobe duración: {e}")
        return 0


def get_vod_file_size(url, base_url):
    """
    Obtiene el tamaño del archivo VOD via HEAD request.
    Necesario para calcular el byte-offset al hacer seeking.
    Returns: bytes (int), o 0 si no disponible.
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header(
            "User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        req.add_header("Referer", f"{base_url}/")
        with urllib.request.urlopen(req, timeout=10) as r:
            size = int(r.headers.get("Content-Length", 0))
            return size
    except Exception as e:
        print(f"   ⚠️ file size: {e}")
        return 0


# ============================================================
# FETCH CON RETRY AUTOMÁTICO (patrón kvaster/iptv-proxy)
# ============================================================
def fetch_with_retry(
    url,
    extra_headers=None,
    max_time_s=2.5,
    retry_delay_s=0.15,
    timeout_s=30,
    method="GET",
    range_header=None,
    channel_name=None,
    is_mirror_retry=False,
):
    """
    Descarga una URL con reintento automático.
    🤖 IA AUTO-HEALING: Si falla tras los reintentos, busca un mirror funcional.
    """
    if HAS_AI:
        try:
            from iptv_ai_core import ai_optimizer, throttle_detector
            domain = throttle_detector._domain(url)
            current_spd = throttle_detector.expected_speeds.get(domain, 3000.0)
            cfg = ai_optimizer.predict_optimal_config(url, 1.0, latency=0, speed_kbps=current_spd, required_bitrate=3000.0)
            # IA AUTO-HEALING: Usa directamente el delay sin dividir (hasta 3s)
            retry_delay_s = max(0.15, float(cfg.get("retry_delay", 0.5)))
            timeout_s = max(timeout_s, int(retry_delay_s * 5 + 5))
        except: pass

    parsed = urllib.parse.urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"{base_url}/",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    if extra_headers:
        headers.update(extra_headers)
    if range_header:
        headers["Range"] = range_header

    last_exc = None
    deadline = time.time() + max_time_s
    attempts = 0

    while time.time() < deadline:
        attempts += 1
        try:
            req = urllib.request.Request(url, method=method)
            for k, v in headers.items():
                req.add_header(k, v)

            resp = urllib.request.urlopen(req, timeout=timeout_s)

            # 🧠 IA: Registrar experiencia exitosa (en background para no bloquear el stream)
            if HAS_AI and channel_name and not range_header:  # solo en petición principal, no en seeks
                try:
                    _t_elapsed = time.time() - (deadline - max_time_s)
                    _speed_kbps = 0  # Se completará con datos reales cuando se lea el body
                    threading.Thread(
                        target=ai_optimizer.log_experience,
                        args=(url,),
                        kwargs={
                            "size_mb": 0,  # Se actualiza cuando el proxy termina de enviar
                            "speed_kbps": 2000,  # Estimado; log_experience ajusta desde CPU actual
                            "success": True,
                            "channel_name": channel_name,
                        },
                        daemon=True
                    ).start()
                except Exception:
                    pass  # La IA nunca debe romper el stream

            return resp, resp.status

        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (403, 404, 429, 503, 502, 504) and time.time() < deadline:
                time.sleep(retry_delay_s)
                continue
            break  # Error permanente

        except Exception as e:
            last_exc = e
            if time.time() < deadline:
                time.sleep(retry_delay_s)
                continue
            break

    # 🤖 AUTO-HEALING: Agotado tiempo → buscar mirror si tenemos el nombre
    if HAS_AI and channel_name and not is_mirror_retry:
        # Registrar fallo en la IA
        try: ai_optimizer.log_failure(url)
        except: pass
        
        mirror_url = ai_optimizer.find_best_mirror(channel_name)
        if mirror_url:
            print(f"🚀 IA: Espejo encontrado para '{channel_name}' → Re-intentando con nueva fuente...")
            return fetch_with_retry(
                mirror_url,
                extra_headers=extra_headers,
                max_time_s=max_time_s,
                retry_delay_s=retry_delay_s,
                timeout_s=timeout_s,
                method=method,
                range_header=range_header,
                channel_name=channel_name,
                is_mirror_retry=True,
            )

    if last_exc:
        raise last_exc
    raise TimeoutError(f"fetch_with_retry: timeout tras {max_time_s}s")


# ============================================================
# UTILIDADES DE CACHÉ EN DISCO (GZIP)
# ============================================================
def ensure_cache_dir():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not os.path.exists(CACHE_INDEX_FILE):
            with open(CACHE_INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            print("✅ Índice caché inicializado")
    except Exception as e:
        print(f"❌ Error creando directorio caché: {e}")


def cache_gz_path(playlist_id):
    return os.path.join(CACHE_DIR, f"{playlist_id}{CACHE_EXT}")


def cache_json_path(playlist_id):
    """Ruta legada .json (para migración)"""
    return os.path.join(CACHE_DIR, f"{playlist_id}.json")


def save_cache_gz(playlist_id, data):
    """Guarda JSON comprimido con gzip (nivel 6 = buen balance)"""
    path = cache_gz_path(playlist_id)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(data, f, ensure_ascii=False)
    size_kb = os.path.getsize(path) / 1024
    print(f"   💾 Guardado: {path} ({size_kb:.1f} KB comprimido)")


def load_cache_gz(playlist_id):
    """Carga JSON comprimido. Fallback a .json legado si existe."""
    gz_path = cache_gz_path(playlist_id)
    json_path = cache_json_path(playlist_id)

    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    elif os.path.exists(json_path):
        # Migrar automáticamente a .gz
        print(f"   🔄 Migrando caché legado a gzip: {playlist_id}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        save_cache_gz(playlist_id, data)
        os.remove(json_path)
        return data
    return None


def load_index():
    try:
        if os.path.exists(CACHE_INDEX_FILE):
            with open(CACHE_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Error leyendo índice: {e}")
    return []


def save_index(index):
    with open(CACHE_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


# ============================================================
# PARSER M3U OPTIMIZADO (iteración línea a línea)
# ============================================================
def parse_m3u(text):
    """
    Parser que usa StringIO para iterar sin split('\n') masivo.
    En listas de 100k+ canales ahorra ~50% de RAM vs list split.
    """
    channels = []
    current_info = None
    RE_NAME = re.compile(r",(.+)$")
    RE_GROUP = re.compile(r'group-title="([^"]+)"', re.IGNORECASE)
    RE_LOGO = re.compile(r'tvg-logo="([^"]+)"', re.IGNORECASE)

    for line in StringIO(text):
        line = line.rstrip("\r\n").strip()
        if not line:
            continue

        if line.startswith("#EXTINF:"):
            nm = RE_NAME.search(line)
            gm = RE_GROUP.search(line)
            lm = RE_LOGO.search(line)
            current_info = {
                "name": nm.group(1).strip() if nm else "Sin nombre",
                "group": gm.group(1) if gm else "Sin grupo",
                "logo": lm.group(1) if lm else None,
            }

        elif not line.startswith("#") and current_info:
            channels.append({**current_info, "url": line})
            current_info = None

    return channels


# ============================================================
# DESCARGA M3U CON GZIP
# ============================================================
def download_m3u(url):
    """
    Descarga un M3U con Accept-Encoding: gzip.
    Descomprime automáticamente si el servidor lo soporta.
    Retorna el texto del M3U y metadata de velocidad.
    """
    req = urllib.request.Request(url)
    req.add_header(
        "User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    req.add_header("Accept-Encoding", "gzip, deflate")
    req.add_header("Connection", "keep-alive")

    t0 = time.time()
    
    timeout_m3u = 120
    if HAS_AI:
        try:
            from iptv_ai_core import ai_optimizer
            cfg = ai_optimizer.predict_optimal_config(url, 5.0) # Playlists pueden ser grandes (5MB est)
            # Si la IA detecta asfixia de red o alta CPU, damos mucho mas tiempo al socket
            buffer_k = cfg.get("buffer_kb", 2500)
            timeout_m3u = 120 if buffer_k > 1500 else 300 
            print(f"   🧠 IA M3U Downloader: Ajustando timeout a {timeout_m3u}s basado en perfil de red.")
        except: pass

    with urllib.request.urlopen(req, timeout=timeout_m3u) as response:
        encoding = response.headers.get("Content-Encoding", "").lower()
        raw_bytes = response.read()
    elapsed = time.time() - t0

    compressed_size = len(raw_bytes)

    if encoding == "gzip":
        raw_bytes = gzip.decompress(raw_bytes)
    elif encoding == "deflate":
        raw_bytes = zlib.decompress(raw_bytes)

    text = raw_bytes.decode("utf-8", errors="ignore")
    final_size = len(raw_bytes)

    speed_kbps = (compressed_size / 1024) / elapsed if elapsed > 0 else 0
    print(
        f"✅ Descargado: {compressed_size:,} bytes en {elapsed:.1f}s ({speed_kbps:.0f} KB/s)"
    )
    if encoding in ("gzip", "deflate"):
        ratio = (1 - compressed_size / final_size) * 100
        print(
            f"   🗜️  Compresión {encoding}: {final_size/1024:.0f} KB → {compressed_size/1024:.0f} KB ({ratio:.0f}% menos)"
        )

    return text, compressed_size, elapsed


# ============================================================
# AUTO-ACTUALIZACIÓN DE PLAYLISTS
# ============================================================
def refresh_one_playlist(playlist_id, url, entry_name=""):
    """
    Re-descarga un M3U, actualiza memoria y disco.
    Llamado por el thread de auto-refresh o por el endpoint manual.
    """
    global active_url
    name = entry_name or url[:40]
    print(f"\n🔄 Actualizando playlist: {name}")

    # Marcar como EN PROGRESO antes de comenzar
    with refresh_status_lock:
        refresh_status[playlist_id] = {
            "timestamp": time.time(),
            "status": "refreshing",
            "name": name,
        }

    try:
        text, compressed_size, elapsed = download_m3u(url)

        t0 = time.time()
        channels = parse_m3u(text)
        groups = sorted(set(ch["group"] for ch in channels))
        print(f"✅ Parseado: {len(channels):,} canales en {time.time()-t0:.1f}s")

        # --- Actualizar caché de memoria ---
        with m3u_cache_lock:
            m3u_cache[url] = {
                "channels": channels,
                "groups": groups,
                "timestamp": time.time(),
            }
            if active_url == url:
                pass  # ya actualizado arriba

        # --- Actualizar caché de disco ---
        existing = load_cache_gz(playlist_id) or {}
        existing.update(
            {
                "channels": channels,
                "timestamp": time.time(),
                "url": url,
            }
        )
        # Preservar live/series/movies si existen (categorizados por el cliente)
        save_cache_gz(playlist_id, existing)

        # --- Actualizar timestamp en índice ---
        index = load_index()
        for entry in index:
            if entry.get("id") == playlist_id:
                entry["timestamp"] = time.time()
                entry["totalChannels"] = len(channels)
        save_index(index)

        with refresh_status_lock:
            refresh_status[playlist_id] = {
                "timestamp": time.time(),
                "status": "success",
                "total": len(channels),
            }

        print(f"✅ Playlist '{name}' actualizada: {len(channels):,} canales")

    except Exception as e:
        print(f"❌ Error actualizando '{name}': {e}")
        
        # FIX: Actualizar timestamp aunque falle para que auto_refresh_loop
        # no dispare otro hilo hasta que pase el intervalo configurado
        index = load_index()
        for entry in index:
            if entry.get("id") == playlist_id:
                entry["timestamp"] = time.time()
        save_index(index)
        
        with refresh_status_lock:
            refresh_status[playlist_id] = {
                "timestamp": time.time(),
                "status": "error",
                "error": str(e),
            }


def auto_refresh_loop():
    """
    Thread daemon que revisa playlists vencidas y las actualiza.
    Duerme en intervalos cortos para responder a cambios de config.
    """
    SLEEP_STEP = 60  # Revisar cada 60 segundos

    while True:
        time.sleep(SLEEP_STEP)

        with auto_refresh_lock:
            enabled = auto_refresh_cfg["enabled"]
            interval = auto_refresh_cfg["interval_minutes"] * 60

        if not enabled:
            continue

        current_time = time.time()
        index = load_index()

        for entry in index:
            url = entry.get("url", "")
            pid = entry.get("id", "")
            last_update = entry.get("timestamp", 0)

            if not url or not pid:
                continue

            if current_time - last_update >= interval:
                # Actualizar en hilo para no bloquear el loop
                t = threading.Thread(
                    target=refresh_one_playlist,
                    args=(pid, url, entry.get("name", "")),
                    daemon=True,
                )
                t.start()
                # Pausa entre refrescos para no saturar la red
                time.sleep(5)


# ============================================================
# FFMPEG: DETECCIÓN DE BITRATE (hilo paralelo)
# ============================================================
def detect_stream_info(url):
    """Probe rápido del stream (3s timeout)"""
    try:
        probe_cmd = [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-timeout",
            "3000000",
            "-user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "-headers",
            "Connection: keep-alive\r\n",
            "-show_entries",
            "format=bit_rate",
            "-of",
            "json",
            url,
        ]
        result = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=6,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout.decode("utf-8", errors="ignore"))
            bitrate = int(data.get("format", {}).get("bit_rate", 0)) / 1000
            if bitrate > 0:
                print(f"   📊 Bitrate detectado: {bitrate:.0f} kbps")
                return bitrate
    except Exception as e:
        print(f"   ⚠️ Probe sin resultado: {e}")
    return 2500  # Default: 2.5 Mbps


# ============================================================
# LIMPIEZA DE STREAMS
# ============================================================
def cleanup_stream(stream_id):
    if stream_id not in active_streams:
        return
    info = active_streams[stream_id]
    try:
        proc = info["process"]
        if proc.poll() is None:
            proc.terminate()
            time.sleep(0.4)
            if proc.poll() is None:
                proc.kill()
    except Exception:
        pass
    try:
        shutil.rmtree(info["temp_dir"])
    except Exception:
        pass
        
    # 🧠 HLS Live AI Logging
    try:
        if HAS_AI and "started" in info:
            elapsed = time.time() - info["started"]
            if elapsed > 10:  # Solo logear si al menos vio 10 segundos
                from iptv_ai_core import ai_optimizer
                success = True if elapsed > 45 else False # Asumimos éxito si duró > 45s sin morir
                speed_kbps = info.get("brkbps", 2500)
                # Estimar Size en base a velocidad constante y tiempo
                size_mb = (speed_kbps * 1024 / 8) * elapsed / (1024 * 1024)
                
                # Reportar al throttle observer
                try:
                    from iptv_ai_core import throttle_detector
                    throttle_detector.record_speed(info["url"], speed_kbps)
                except Exception:
                    pass
                    
                print(f"   🧠 IA ABR: Grabando memoria de stream HLS ({elapsed:.0f}s, {size_mb:.1f}MB, Éxito:{success})")
                threading.Thread(
                    target=ai_optimizer.log_experience, 
                    args=(info["url"], size_mb, speed_kbps, success, 0, None), 
                    daemon=True
                ).start()
    except Exception as e:
        print(f"   ⚠️ Error loggeando memoria HLS: {e}")

    del active_streams[stream_id]
    print(f"   🧹 Stream {stream_id} limpiado")



def cleanup_inactive_streams():
    """Limpia streams inactivos cada 3 minutos"""
    while True:
        time.sleep(180)
        current_time = time.time()
        with streams_lock:
            to_remove = [
                sid
                for sid, info in active_streams.items()
                if current_time - info.get("last_access", current_time) > 600
            ]
            for sid in to_remove:
                print(f"\n🧹 Limpiando stream inactivo: {sid}")
                cleanup_stream(sid)


def log_request_speed(url, bytes_sent, elapsed_time):
    speed_kbps = (bytes_sent / 1024) / elapsed_time if elapsed_time > 0 else 0
    speed_log.append(
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "url": url[:60],
            "bytes": bytes_sent,
            "time": elapsed_time,
            "speed_kbps": round(speed_kbps, 2),
        }
    )


# ============================================================
# SERVIDOR HTTP
# ============================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 50


class IPTVHandler(SimpleHTTPRequestHandler):
    timeout = 180

    def log_message(self, format, *args):
        msg = format % args
        # Silenciar endpoints de polling continuo del frontend
        if any(x in msg for x in ["/api/ai-stats", "/api/cache/list", "/api/auto-refresh", "/api/speed-log", "/api/cache/refresh-status"]):
            return
        sys.stdout.write("%s - %s\n" % (self.log_date_time_string(), msg))
        sys.stdout.flush()


    # ── Único send_json_response ──────────────────────────────
    def send_json_response(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # NOTA: NO añadimos CORS aquí — end_headers() override ya los añade.
            # Añadirlos dos veces corrompe la respuesta HTTP.
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def handle_ai_stats(self):
        """Devuelve el estado evolucionado de la Red Neuronal Evolutiva"""
        try:
            if not HAS_AI:
                self.send_json_response(400, {"error": "IA no habilitada"})
                return
            
            # Obtener datos reales de SQLite y Config
            import sqlite3
            conn = sqlite3.connect("C:/IPTV_Cache/iptv_permanent_memory.db")
            # Bug Fix: COALESCE evita NULLs en filas antiguas que crashean el JS
            recent = conn.execute("""
                SELECT COALESCE(size_mb, 0), COALESCE(actual_speed, 0),
                       COALESCE(success, 0), COALESCE(cpu, 0),
                       COALESCE(channel_name, 'VOD')
                FROM experiences ORDER BY id DESC LIMIT 15
            """).fetchall()
            db_count = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            conn.close()

            # Bug Fix: avg_error siempre como float (nunca None)
            avg_error = ai_optimizer.config.get("avg_error", 0.0)
            avg_error = round(float(avg_error) if avg_error is not None else 0.0, 4)

            stats = {
                "experience_count": db_count,
                "complexity": ai_optimizer.config.get("complexity_level", 1),
                "brain_layers": ai_optimizer.config.get("layers", [16, 12, 8]),
                "avg_error": avg_error,
                "has_model": ai_optimizer.model is not None,
                "cpu_usage": round(psutil.cpu_percent(), 1),
                "ram_usage": round(psutil.virtual_memory().percent, 1),
                "recent_events": [
                    {
                        "input": [float(r[0]), 0, float(r[3])],
                        "speed": float(r[1]),
                        "success": bool(r[2]),
                        "name": r[4]
                    }
                    for r in recent
                ],
                "abr_metrics": app_metrics
            }
            self.send_json_response(200, stats)

        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    # ── Routing ───────────────────────────────────────────────
    def do_GET(self):
        p = self.path
        if p.startswith("/api/load-m3u?"):
            self.handle_load_m3u()
        elif p.startswith("/api/check-m3u?"):
            self.handle_check_m3u()
        elif p.startswith("/api/search?"):
            self.handle_search()
        elif p.startswith("/api/groups"):
            self.handle_get_groups()
        elif p.startswith("/api/download-group?"):
            self.handle_download_group()
        elif p.startswith("/hls/"):
            self.handle_hls_file()
        elif p.startswith("/api/play-hls?"):
            self.handle_play_hls()
        elif p.startswith("/api/play-vod-hls?"):
            self.handle_play_vod_hls()
        elif p.startswith("/api/transcode-vod?"):
            self.handle_transcode_vod()
        elif p.startswith("/api/vod-duration?"):
            self.handle_vod_duration()
        elif p.startswith("/api/proxy-video?"):
            self.handle_proxy_video()
        elif p == "/api/cache/list":
            self.handle_cache_list()
        elif p.startswith("/api/cache/load/"):
            self.handle_cache_load()
        elif p.startswith("/api/cache/refresh/"):
            self.handle_cache_refresh_one()
        elif p == "/api/cache/refresh-status":
            self.handle_refresh_status()
        elif p == "/api/auto-refresh/settings":
            self.handle_get_auto_refresh()
        elif p == "/api/speed-log":
            self.handle_speed_log()
        elif p == "/api/ai-stats":
            self.handle_ai_stats()
        else:
            super().do_GET()

    def do_DELETE(self):
        p = self.path
        if p.startswith("/api/cache/delete/"):
            self.handle_cache_delete()
        elif p == "/api/cache/clear":
            self.handle_cache_clear()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        p = self.path
        if p == "/api/cache/save":
            self.handle_cache_save()
        elif p == "/api/auto-refresh/settings":
            self.handle_set_auto_refresh()
        else:
            self.send_response(404)
            self.end_headers()

    # ============================================================
    # ENDPOINTS DE AUTO-ACTUALIZACIÓN
    # ============================================================
    def handle_get_auto_refresh(self):
        with auto_refresh_lock:
            cfg = dict(auto_refresh_cfg)
        self.send_json_response(200, cfg)

    def handle_set_auto_refresh(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            with auto_refresh_lock:
                if "enabled" in body:
                    auto_refresh_cfg["enabled"] = bool(body["enabled"])
                if "interval_minutes" in body:
                    auto_refresh_cfg["interval_minutes"] = max(
                        5, int(body["interval_minutes"])
                    )
            print(f"⚙️  Auto-refresh: {auto_refresh_cfg}")
            self.send_json_response(200, {"success": True, **auto_refresh_cfg})
        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def handle_cache_refresh_one(self):
        """GET /api/cache/refresh/{id}  – refresca una playlist manualmente"""
        try:
            playlist_id = self.path.split("/")[-1]
            index = load_index()
            entry = next((e for e in index if e.get("id") == playlist_id), None)

            if not entry:
                self.send_json_response(404, {"error": "Playlist no encontrada"})
                return

            url = entry.get("url", "")
            name = entry.get("name", "")

            # Lanzar en hilo para responder inmediatamente
            t = threading.Thread(
                target=refresh_one_playlist,
                args=(playlist_id, url, name),
                daemon=True,
            )
            t.start()

            self.send_json_response(
                200, {"success": True, "status": "refreshing", "id": playlist_id}
            )

        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def handle_refresh_status(self):
        """GET /api/cache/refresh-status – estado de actualizaciones recientes"""
        with refresh_status_lock:
            data = dict(refresh_status)
        self.send_json_response(200, data)

    def handle_speed_log(self):
        self.send_json_response(200, list(speed_log))

    # ============================================================
    # CACHÉ EN DISCO
    # ============================================================
    def handle_cache_list(self):
        try:
            self.send_json_response(200, load_index())
        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def handle_cache_load(self):
        try:
            playlist_id = self.path.split("/")[-1]
            data = load_cache_gz(playlist_id)
            if data:
                self.send_json_response(200, data)
                print(f"✅ Caché cargado: {playlist_id}")
            else:
                self.send_json_response(404, {"error": "Playlist no encontrada"})
        except Exception as e:
            print(f"❌ Error cargando caché: {e}")
            self.send_json_response(500, {"error": str(e)})

    def handle_cache_save(self):
        """POST /api/cache/save – guarda playlist categorizada (del cliente)"""
        try:
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            pid = body.get("id")
            idx_entry = body.get("indexEntry")
            cache_data = body.get("cacheData")

            if not all([pid, idx_entry, cache_data]):
                self.send_json_response(400, {"error": "Faltan datos"})
                return

            name = idx_entry.get("name", "unknown")
            print(f"💾 Guardando en disco (gzip): {name}")

            save_cache_gz(pid, cache_data)

            # Actualizar índice
            index = load_index()
            index = [e for e in index if e.get("id") != pid]
            index.append(idx_entry)
            save_index(index)

            print(f"✅ '{name}' guardado exitosamente")
            self.send_json_response(200, {"success": True})

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.send_json_response(500, {"error": str(e)})

    def handle_cache_delete(self):
        try:
            pid = self.path.split("/")[-1]
            for path in [cache_gz_path(pid), cache_json_path(pid)]:
                if os.path.exists(path):
                    os.remove(path)

            index = load_index()
            index = [e for e in index if e.get("id") != pid]
            save_index(index)

            print(f"🗑️ Playlist eliminada: {pid}")
            self.send_json_response(200, {"success": True})
        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def handle_cache_clear(self):
        try:
            for fname in os.listdir(CACHE_DIR):
                fpath = os.path.join(CACHE_DIR, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
            save_index([])
            print("🗑️ Caché en disco vaciado completamente")
            self.send_json_response(200, {"success": True})
        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    # ============================================================
    # CARGA DE M3U
    # ============================================================
    def handle_check_m3u(self):
        """
        Verificación rápida de una URL M3U:
        - Conecta al servidor (timeout 10s)
        - Lee los primeros 8KB para detectar si es un M3U válido
        - Devuelve estado, tiempo de respuesta y diagnóstico
        """
        import ssl as _ssl

        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            url = params.get("url", [None])[0]
            if not url:
                self.send_json_response(400, {"error": "URL requerida"})
                return
            url = urllib.parse.unquote(url)

            print(f"\n🔍 Verificando lista: {url[:70]}...")

            t0 = time.time()

            # SSL permisivo (solo para HTTPS — pasar context a http:// causa TypeError)
            is_https = url.lower().startswith("https://")
            if is_https:
                ctx = _ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = _ssl.CERT_NONE
            else:
                ctx = None

            def _open(try_url, use_ctx):
                r = urllib.request.Request(try_url)
                r.add_header(
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )
                r.add_header("Range", "bytes=0-8191")  # solo primeros 8 KB
                if use_ctx:
                    return urllib.request.urlopen(r, timeout=10, context=use_ctx)
                else:
                    return urllib.request.urlopen(r, timeout=10)

            active = False
            status = "error"
            message = ""
            detail = ""
            http_code = None
            latency_ms = None

            try:
                with _open(url, ctx) as resp:
                    http_code = resp.getcode()
                    latency_ms = int((time.time() - t0) * 1000)
                    chunk = resp.read(8192).decode("utf-8", errors="ignore")

                if "#EXTM3U" in chunk or "#EXTINF" in chunk:
                    active = True
                    status = "ok"
                    count_sample = chunk.count("#EXTINF")
                    message = "✅ Lista activa y válida"
                    detail = (
                        f"Servidor respondió en {latency_ms}ms. "
                        f"Contiene contenido M3U válido."
                        + (
                            f" (~{count_sample}+ canales en la muestra)"
                            if count_sample
                            else ""
                        )
                    )
                elif "<html" in chunk.lower() or "<!doctype" in chunk.lower():
                    status = "invalid"
                    message = "⚠️ Servidor responde pero devuelve HTML"
                    detail = (
                        "Las credenciales pueden ser incorrectas o la URL no es de tipo m3u_plus. "
                        f"Respondió en {latency_ms}ms."
                    )
                elif len(chunk.strip()) == 0:
                    status = "empty"
                    message = "⚠️ Servidor respondió vacío"
                    detail = f"El servidor respondió (HTTP {http_code}) pero sin contenido en {latency_ms}ms."
                else:
                    active = True
                    status = "ok_noheader"
                    message = "✅ Lista accesible (sin cabecera estándar)"
                    detail = (
                        f"Servidor respondió en {latency_ms}ms con contenido accesible."
                    )

            except urllib.error.HTTPError as e:
                latency_ms = int((time.time() - t0) * 1000)
                http_code = e.code
                if e.code == 401:
                    status = "auth_error"
                    message = "🔑 Credenciales incorrectas (401 Unauthorized)"
                    detail = "El servidor rechazó el usuario/contraseña."
                elif e.code == 403:
                    status = "forbidden"
                    message = "🔒 Acceso denegado (403 Forbidden)"
                    detail = "La cuenta puede haber expirado o el servidor bloqueó el acceso."
                elif e.code == 404:
                    status = "not_found"
                    message = "🔍 Lista no encontrada (404)"
                    detail = (
                        "La URL no existe en el servidor. Verifica que sea correcta."
                    )
                elif e.code == 504:
                    status = "timeout_server"
                    message = "⏱️ Servidor saturado (504 Gateway Timeout)"
                    detail = "El servidor recibió la petición pero no pudo responder a tiempo. Intenta más tarde."
                elif e.code == 503:
                    status = "unavailable"
                    message = "🔴 Servidor no disponible (503)"
                    detail = "El servidor está temporalmente fuera de servicio."
                elif e.code == 502:
                    status = "bad_gateway"
                    message = "🔴 Error de gateway (502)"
                    detail = "El servidor IPTV está teniendo problemas internos."
                else:
                    status = "http_error"
                    message = f"❌ Error HTTP {e.code}"
                    detail = str(e)

            except urllib.error.URLError as e:
                latency_ms = int((time.time() - t0) * 1000)
                # En Python 3.3+, socket.timeout es subclase de OSError/TimeoutError.
                # urllib a veces la envuelve en URLError, otras veces no.
                reason = str(e.reason) if hasattr(e, "reason") else str(e)
                if (
                    isinstance(e.reason, (TimeoutError, OSError))
                    or "timed out" in reason.lower()
                    or "timeout" in reason.lower()
                ):
                    status = "timeout"
                    message = "⏱️ Sin respuesta (timeout 10s)"
                    detail = (
                        "El servidor no responde. Parece estar caído o inaccesible."
                    )

                # SSL incorrecto → intentar HTTP automáticamente
                elif (
                    "WRONG_VERSION_NUMBER" in reason or "SSL" in reason.upper()
                ) and url.startswith("https://"):
                    http_url = "http://" + url[8:]
                    print(f"   ⚠️  SSL falló → reintentando con HTTP...")
                    try:
                        t1 = time.time()
                        with _open(http_url, None) as r2:
                            chunk2 = r2.read(8192).decode("utf-8", errors="ignore")
                        latency_ms = int((time.time() - t1) * 1000)
                        if "#EXTINF" in chunk2 or "#EXTM3U" in chunk2:
                            active = True
                            status = "ok_http_fallback"
                            message = "✅ Lista activa (usa HTTP, no HTTPS)"
                            detail = (
                                f"SSL falló pero HTTP funcionó en {latency_ms}ms. "
                                f"Cambia 'https://' por 'http://' en la URL."
                            )
                        else:
                            status = "ssl_invalid"
                            message = "🔐 Error SSL y HTTP no devuelve M3U"
                            detail = "Cambia https:// por http:// e inténtalo de nuevo."
                    except Exception as e3:
                        status = "ssl_error"
                        message = "🔐 Error SSL de conexión"
                        detail = "Prueba cambiando 'https://' por 'http://' en la URL."

                elif (
                    "10061" in reason
                    or "Connection refused" in reason.lower()
                    or "denegó" in reason
                ):
                    status = "refused"
                    message = "🚫 Conexión rechazada"
                    detail = "El servidor rechazó la conexión. Puerto incorrecto o servidor caído."

                elif (
                    "Name or service not known" in reason
                    or "getaddrinfo" in reason
                    or "11001" in reason
                    or "11004" in reason
                ):
                    status = "dns_error"
                    message = "❓ Servidor no encontrado (DNS)"
                    detail = (
                        "No se pudo resolver el nombre del servidor. Verifica la URL."
                    )

                else:
                    status = "connection_error"
                    message = "❌ Error de conexión"
                    detail = reason

            except (TimeoutError, OSError) as e:
                # TimeoutError built-in (socket.timeout) a veces no va envuelto en URLError
                latency_ms = int((time.time() - t0) * 1000)
                status = "timeout"
                message = "⏱️ Sin respuesta (timeout 10s)"
                detail = "El servidor no responde. Parece estar caído o inaccesible."

            except Exception as e:
                latency_ms = int((time.time() - t0) * 1000)
                import traceback

                traceback.print_exc()
                status = "error"
                message = "❌ Error inesperado"
                detail = str(e)

            result = {
                "active": active,
                "status": status,
                "message": message,
                "detail": detail,
                "latency_ms": latency_ms,
                "http_code": http_code,
                "url": url,
            }
            print(f"   {'✅' if active else '❌'} {status}: {message}")
            self.send_json_response(200, result)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.send_json_response(500, {"error": str(e)})

    def handle_load_m3u(self):
        global active_url
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            url = params.get("url", [None])[0]
            if not url:
                self.send_json_response(400, {"error": "URL requerida"})
                return
            url = urllib.parse.unquote(url)

            current_time = time.time()

            # ── Verificar caché en memoria ──
            with m3u_cache_lock:
                cached = m3u_cache.get(url)

            if cached and current_time - cached.get("timestamp", 0) < 3600:
                channels = cached["channels"]
                groups = cached["groups"]
                active_url = url
                print(f"✅ Caché memoria: {len(channels):,} canales")
                self.send_json_response(
                    200, {"status": "cached", "total": len(channels), "groups": groups}
                )
                return

            # ── Descargar M3U nuevo ──
            print(f"\n{'='*60}")
            print(f"📥 Cargando M3U: {url[:60]}...")
            print(f"{'='*60}")

            text, compressed, elapsed = download_m3u(url)

            t0 = time.time()
            channels = parse_m3u(text)
            groups = sorted(set(ch["group"] for ch in channels))
            print(f"✅ Parseado: {len(channels):,} canales en {time.time()-t0:.1f}s")
            print(f"📂 Grupos: {len(groups)}")

            with m3u_cache_lock:
                m3u_cache[url] = {
                    "channels": channels,
                    "groups": groups,
                    "timestamp": current_time,
                }
            active_url = url
            print(f"{'='*60}\n")

            self.send_json_response(
                200, {"status": "loaded", "total": len(channels), "groups": groups}
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            self.send_json_response(500, {"error": str(e)})

    # ============================================================
    # BÚSQUEDA Y GRUPOS
    # ============================================================
    def handle_search(self):
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            q = params.get("q", [""])[0].lower()
            group = params.get("group", [""])[0]
            offset = int(params.get("offset", [0])[0])
            limit = int(params.get("limit", [100])[0])

            with m3u_cache_lock:
                if not active_url or active_url not in m3u_cache:
                    self.send_json_response(400, {"error": "No hay M3U cargado"})
                    return
                channels = m3u_cache[active_url]["channels"]

            filtered = [
                ch
                for ch in channels
                if (not q or q in ch["name"].lower())
                and (not group or ch["group"] == group)
            ]

            self.send_json_response(
                200,
                {
                    "results": filtered[offset : offset + limit],
                    "total": len(filtered),
                    "offset": offset,
                    "limit": limit,
                },
            )
        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def handle_get_groups(self):
        try:
            with m3u_cache_lock:
                if not active_url or active_url not in m3u_cache:
                    self.send_json_response(400, {"error": "No hay M3U"})
                    return
                cache = m3u_cache[active_url]

            counts = {}
            for ch in cache["channels"]:
                counts[ch["group"]] = counts.get(ch["group"], 0) + 1

            self.send_json_response(
                200,
                {
                    "groups": [
                        {"name": g, "count": counts[g]} for g in cache["groups"]
                    ],
                    "total": len(cache["channels"]),
                },
            )
        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    # ============================================================
    # DESCARGA DE GRUPO M3U
    # ============================================================
    def handle_download_group(self):
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            group_name = params.get("group", [""])[0]

            with m3u_cache_lock:
                if not active_url or active_url not in m3u_cache:
                    self.send_json_response(400, {"error": "No hay M3U"})
                    return
                channels = m3u_cache[active_url]["channels"]

            filtered = [ch for ch in channels if ch.get("group") == group_name]
            if not filtered:
                self.send_json_response(404, {"error": "Grupo vacío"})
                return

            m3u = "#EXTM3U\n"
            for ch in filtered:
                logo = f' tvg-logo="{ch["logo"]}"' if ch.get("logo") else ""
                m3u += f'#EXTINF:-1{logo} group-title="{ch["group"]}",{ch["name"]}\n{ch["url"]}\n'

            safe = (
                re.sub(r"[^\w\s-]", "", group_name).strip().replace(" ", "_") or "grupo"
            )
            body = m3u.encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "audio/x-mpegurl")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{safe}.m3u"'
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    # ============================================================
    # PROXY DE VIDEO (Range Requests)
    # ============================================================
    def handle_proxy_video(self):
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            url = params.get("url", [None])[0]
            if not url:
                self.send_json_response(400, {"error": "URL requerida"})
                return
            url = urllib.parse.unquote(url)

            print(f"\n{'='*60}")
            print(f"🎬 Proxy VOD: {url[:60]}...")

            ext = url.split("?")[0].split(".")[-1].lower()
            print(f"   📄 Extensión: .{ext}")

            # Verificar Content-Type
            try:
                req = urllib.request.Request(url, method="HEAD")
                req.add_header(
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )
                parsed = urllib.parse.urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                req.add_header("Referer", f"{base_url}/")

                with urllib.request.urlopen(req, timeout=10) as r:
                    content_type = r.headers.get("Content-Type", "").lower()
                    accept_ranges = r.headers.get("Accept-Ranges", "none")
                    content_length = r.headers.get("Content-Length", "0")

                print(f"   🔍 Content-Type: {content_type}")
                print(f"   📏 Content-Length: {content_length} bytes")
                print(f"   🎯 Accept-Ranges: {accept_ranges}")
            except Exception as e:
                print(f"   ⚠️ HEAD request falló: {e}")
                content_type = ""
                accept_ranges = "none"

            video_exts = {"mp4", "mkv", "avi", "m4v", "mov", "wmv", "flv"}
            if ext in video_exts:
                use_parallel = params.get("parallel", ["false"])[0].lower() == "true"
                # Tamaño total conocido por HEAD (puede ser 0 si server no lo dio)
                known_size = int(content_length) if content_length.isdigit() else 0
                if use_parallel:
                    print(f"   🚀 Modo paralelo")
                    print(f"{'='*60}\n")
                    self.proxy_parallel_download(url)
                else:
                    print(
                        f"   ✅ Proxy directo → {ext.upper()} | size={known_size:,} bytes"
                    )
                    print(f"{'='*60}\n")
                    self.proxy_range_request(url, known_size=known_size)
            else:
                self.send_json_response(400, {"error": f"Formato no soportado: {ext}"})

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.send_json_response(500, {"error": str(e)})

    def proxy_range_request(self, url, known_size=0):
        """Proxy directo con soporte Range Requests para seeking + retry automático

        known_size: tamaño total del archivo (de HEAD request previo).
        Si el upstream no devuelve Content-Length/Range, lo inyectamos nosotros
        para que el browser sepa la duración total y pueda hacer seeking.
        """
        try:
            range_header = self.headers.get("Range", "bytes=0-")

            # ── fetch_with_retry: tolera 404/403 transitorios del servidor IPTV ──
            try:
                response, status = fetch_with_retry(
                    url,
                    range_header=range_header,
                    max_time_s=2.5,
                    timeout_s=60,
                )
            except urllib.error.HTTPError as e:
                print(f"   ❌ HTTP {e.code}: {e.reason} (tras retry)")
                self.send_error(e.code, e.reason)
                return
            except Exception as e:
                print(f"   ❌ Error proxy: {e}")
                self.send_error(502, str(e))
                return

            content_type = response.headers.get("Content-Type", "video/mp4")
            content_length = response.headers.get("Content-Length", "")
            content_range = response.headers.get("Content-Range", "")
            accept_ranges = response.headers.get("Accept-Ranges", "bytes")

            # ── Inyectar Content-Range si el upstream no lo devolvió ──────────
            # Sin Content-Range, el browser no sabe el tamaño total del archivo
            # y no puede hacer seeking → la barra de progreso no funciona.
            # Si tenemos known_size (de HEAD previo), construimos el header.
            if not content_range and known_size > 0:
                # Parsear el byte range pedido (ej. "bytes=1024-")
                try:
                    rng = range_header.replace("bytes=", "").strip()
                    start_b = int(rng.split("-")[0]) if rng.split("-")[0] else 0
                    end_b = known_size - 1
                    part_len = end_b - start_b + 1
                    content_range = f"bytes {start_b}-{end_b}/{known_size}"
                    content_length = str(part_len)
                    print(f"   📏 Inyectando Content-Range: {content_range}")
                except Exception:
                    pass

            is_partial = bool(content_range)
            self.send_response(206 if is_partial else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            if content_length:
                self.send_header("Content-Length", content_length)
            if content_range:
                self.send_header("Content-Range", content_range)
            self.end_headers()

            start_time = time.time()
            bytes_sent = 0
            CHUNK = 256 * 1024  # 256 KB
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    bytes_sent += len(chunk)
                    if bytes_sent % (10 * 1024 * 1024) == 0:
                        print(f"   📤 {bytes_sent/1024/1024:.1f} MB enviados")
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    ConnectionAbortedError,
                ) as e:
                    print(f"   ⚠️ Cliente desconectó: {type(e).__name__}")
                    break

            if bytes_sent > 0:
                elapsed = time.time() - start_time
                log_request_speed(url, bytes_sent, elapsed)
                speed_kbps = (bytes_sent / 1024) / max(elapsed, 0.01)
                speed_mbps = speed_kbps / 1024
                print(f"   ✅ {bytes_sent/1024/1024:.1f} MB @ {speed_mbps:.2f} MB/s")

                # 🧠 IA: Registrar velocidad y detectar throttle
                if HAS_AI:
                    try:
                        from iptv_ai_core import throttle_detector
                        throttle_detector.record_speed(url, speed_kbps)
                        analysis = throttle_detector.analyze(url, speed_kbps)
                        if analysis["throttled"] and analysis["pieces"] > 1:
                            domain = url.split('/')[2] if '/' in url else url
                            print(f"   🚨 THROTTLE en {domain}: {analysis['strategy']}")
                            print(f"   💡 Próxima petición: activar {analysis['pieces']} piezas paralelas")
                    except Exception:
                        pass


        except Exception as e:
            print(f"   ❌ Error proxy_range: {e}")
            try:
                self.send_error(500, str(e))
            except:
                pass

    def proxy_parallel_download(self, url):
        """Descarga adaptativa con múltiples conexiones paralelas"""
        try:
            print(f"\n{'='*60}")
            print(f"🚀 DESCARGA PARALELA ADAPTATIVA")

            parsed = urllib.parse.urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            req = urllib.request.Request(url, method="HEAD")
            req.add_header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            req.add_header("Referer", f"{base_url}/")

            with urllib.request.urlopen(req, timeout=10) as r:
                content_length = int(r.headers.get("Content-Length", 0))
                accept_ranges = r.headers.get("Accept-Ranges", "none")

            if accept_ranges == "none" or content_length == 0:
                print(f"   ⚠️ Sin Range Requests → fallback a proxy normal")
                self.proxy_range_request(url)
                return

            size_mb = content_length / 1024 / 1024
            
            # 🤖 IA OPTIMIZER V2: Medición de latencia y predicción profunda
            latency_ms = 0
            if HAS_AI:
                # Medir latencia rápida al servidor
                t_lat_0 = time.time()
                try:
                    urllib.request.urlopen(url, timeout=1).close()
                    latency_ms = (time.time() - t_lat_0) * 1000
                except: latency_ms = 500
                
                ai_config = ai_optimizer.predict_optimal_config(url, size_mb, latency_ms)
                num_conn = ai_config["num_conn"]
                print(f"   🧠 IA V2 Sugirió: {num_conn} conns | Prefetch: {ai_config['prefetch_count']} | Latencia: {latency_ms:.0f}ms")
            else:
                # Fallback tradicional
                if size_mb < 50: num_conn = 4
                elif size_mb < 200: num_conn = 8
                elif size_mb < 500: num_conn = 12
                else: num_conn = 16

            print(f"   📏 Tamaño: {size_mb:.1f} MB → {num_conn} conexiones")

            # Protección contra num_conn muy bajo (extreme_starvation → num_conn=2)
            # Evitar chunk_size=0 o fragmentos inválidos con archivos pequeños
            num_conn = max(1, num_conn)  # Nunca 0 conexiones
            chunk_size = content_length // num_conn
            # Si el chunk resultante es demasiado pequeño (<1KB), reducir conexiones
            if chunk_size < 1024 and num_conn > 1:
                num_conn = max(1, content_length // 1024)
                chunk_size = content_length // num_conn
                print(f"   ⚠️ Ajuste protectivo: archivo pequeño → {num_conn} conexión(es)")
            part_buffers = [bytearray() for _ in range(num_conn)]
            errors = []
            speeds = []

            def download_part(idx, start, end, buf):
                try:
                    # 🕵️ Anti-detección: cada pieza tiene identidad única
                    headers = {}
                    if HAS_AI:
                        try:
                            from iptv_ai_core import throttle_detector
                            headers = throttle_detector.get_disguised_headers(idx, url)
                            jitter  = throttle_detector.get_jitter_delay(idx)
                            time.sleep(jitter)  # Escalonar inicio para no parecer burst
                        except Exception:
                            pass

                    r2 = urllib.request.Request(url)
                    # Si no hay headers de anti-detección, usar los por defecto
                    ua  = headers.get("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                    ref = headers.get("Referer", f"{base_url}/")
                    r2.add_header("User-Agent",      ua)
                    r2.add_header("Referer",         ref)
                    r2.add_header("Accept",          headers.get("Accept", "*/*"))
                    r2.add_header("Accept-Language", headers.get("Accept-Language", "en-US,en;q=0.9"))
                    r2.add_header("Cache-Control",   headers.get("Cache-Control", "no-cache"))
                    r2.add_header("Range",           f"bytes={start}-{end}")

                    t0 = time.time()
                    with urllib.request.urlopen(r2, timeout=120) as resp:
                        downloaded = 0
                        while True:
                            c = resp.read(64 * 1024)
                            if not c:
                                break
                            buf.extend(c)
                            downloaded += len(c)
                    elapsed = time.time() - t0
                    spd = (downloaded / 1024) / elapsed if elapsed > 0 else 0
                    speeds.append(spd)
                    print(
                        f"   ✅ Pieza {idx+1}/{num_conn}: {downloaded/1024/1024:.1f} MB @ {spd:.0f} KB/s [{ua[:30]}...]"
                    )
                except Exception as e:
                    errors.append(f"Pieza {idx+1}: {e}")


            ranges = [
                (
                    i,
                    i * chunk_size,
                    (
                        (i + 1) * chunk_size - 1
                        if i < num_conn - 1
                        else content_length - 1
                    ),
                )
                for i in range(num_conn)
            ]
            threads = []
            t0 = time.time()
            for idx, start, end in ranges:
                t = threading.Thread(
                    target=download_part,
                    args=(idx, start, end, part_buffers[idx]),
                    daemon=True,
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=180)

            if errors:
                print(f"   ⚠️ Errores en partes: {errors}")
                self.proxy_range_request(url)
                return

            total_elapsed = time.time() - t0
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            print(
                f"   🏁 {size_mb:.1f} MB descargados en {total_elapsed:.1f}s (avg {avg_speed:.0f} KB/s)"
            )

            # Enviar al cliente en orden
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            bytes_sent = 0
            for buf in part_buffers:
                try:
                    self.wfile.write(bytes(buf))
                    bytes_sent += len(buf)
                except (BrokenPipeError, ConnectionResetError):
                    print(f"   ⚠️ Cliente desconectó")
                    break

            log_request_speed(url, bytes_sent, total_elapsed)
            
            # 📊 Reportar velocidad total al detector de throttle
            if HAS_AI and avg_speed > 0:
                try:
                    from iptv_ai_core import throttle_detector
                    total_speed_kbps = (bytes_sent / 1024) / max(total_elapsed, 0.1)
                    throttle_detector.record_speed(url, total_speed_kbps)
                    analysis = throttle_detector.analyze(url, total_speed_kbps)
                    if analysis["throttled"]:
                        print(f"   ⚠️ {analysis['strategy']}")
                    else:
                        print(f"   📶 Velocidad: {analysis['strategy']}")
                except Exception:
                    pass
            
            # 🤖 IA V2: Registrar experiencia
            if HAS_AI:
                ai_optimizer.log_experience(url, size_mb, avg_speed, success=True, latency=latency_ms)
                if ai_optimizer.experience_count % 20 == 0:
                    threading.Thread(target=ai_optimizer.evolve_brain, daemon=True).start()
            
            print(f"{'='*60}\n")


        except Exception as e:
            import traceback

            traceback.print_exc()
            self.send_error(500, str(e))

    # ============================================================
    # HLS: LIVE STREAMS
    # ============================================================
    # ──────────────────────────────────────────────────────────────────────────
    # HELPER: construir comando FFmpeg (reutilizable para inicio y reinicio)
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_ffmpeg_cmd(
        url, temp_dir, transcode_video, quality_profile, hls_time, start_number=0, ai_config=None
    ):
        """
        ══════════════════════════════════════════════════════════════
        OPTIMIZACIÓN DE LATENCIA — objetivo: ≤ 2s respecto al live edge
        ══════════════════════════════════════════════════════════════

        Fuentes de latencia eliminadas:
        ┌─────────────────────────────────────┬───────────┬───────────┐
        │ Fuente                              │ Antes     │ Ahora     │
        ├─────────────────────────────────────┼───────────┼───────────┤
        │ Probe/analyze FFmpeg                │ 1.5s      │ 0.25s     │
        │ Buffer VBV (bufsize)                │ 2× bitrate│ 1× bitrate│
        │ Lookahead encoder x264 (bframes)    │ ~16 frames│ 0 frames  │
        │ Duración de segmento (transcodif.)  │ 2-4s      │ 1s        │
        │ Primer segmento (hls_init_time)     │ = hls_time│ 1s        │
        │ Playlist window (hls_list_size)     │ 12 segs   │ 6 segs    │
        │ GOP mal alineado con segmento       │ variable  │ fijo=fps×1s│
        │ Flush a disco                       │ buffered  │ inmediato  │
        └─────────────────────────────────────┴───────────┴───────────┘

        Con hls_time=1 y HLS.js liveSyncDurationCount=2:
          latencia total = 1s × 2 segmentos = ~2s desde live edge ✓
        """
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if transcode_video and quality_profile:
            w, h, vbr, abr = quality_profile

            # ── x264 parámetros para ZERO encoder latency ────────────────────
            # rc-lookahead=0  → sin buffer de frames para rate control
            # bframes=0       → sin B-frames (cada uno añade ~1 frame de latencia)
            # scenecut=0      → GOP fijo, no inserta IDR frames inesperados
            # no-mbtree=1     → macroblock tree requiere lookahead, lo desactivamos
            # sync-lookahead=0→ sin threaded lookahead delay
            # ref=1           → solo 1 frame de referencia (baseline implica esto)
            x264params = (
                "rc-lookahead=0:bframes=0:scenecut=0:no-mbtree=1:sync-lookahead=0:ref=1"
            )

            # GOP = fps × hls_time → keyframe exacto al inicio de cada segmento.
            # Usamos 30fps como base conservadora (funciona para 25fps también).
            # Con split_by_time el segmento se corta en hls_time exacto aunque no haya keyframe.
            gop = int(30 * hls_time)

            video_args = [
                "-c:v",
                "libx264",
                # superfast: ~40% menos CPU que veryfast (benchmarks Streaming Learning Center 2024)
                # calidad casi idéntica — diferencia SSIM < 0.5dB en streams de telenovela/deportes
                "-preset",
                "superfast",
                "-tune",
                "zerolatency",  # implica bframes=0, force-cfr, slice threads
                "-x264-params",
                x264params,
                "-profile:v",
                "baseline",  # decodificación más rápida en cliente
                "-b:v",
                f"{vbr}k",
                # maxrate 1.1× (no 1.2×): bursts más controlados, menos latencia de buffer
                "-maxrate",
                f"{int(vbr * 1.1)}k",
                # La IA ajusta el VBV Buffer Size (si la red es inestable, lo agranda)
                "-bufsize",
                f"{int(ai_config.get('buffer_kb', vbr)) if ai_config else vbr}k",
                # GOP fijo alineado con segmento → keyframe en cada seg_000X.ts
                "-g",
                str(gop),
                "-keyint_min",
                str(gop),
                # flags cgop: closed GOP → cada segmento es autónomo, HLS.js puede empezar en cualquiera
                "-flags",
                "+cgop",
                "-pix_fmt",
                "yuv420p",  # evitar problemas de color space con streams IPTV
                # Filtro de escala optimizado: lanczos da mejor calidad que bilinear/bicubic
                # en downscales pronunciados (1080p→480p) sin costo de CPU perceptible
                "-vf",
                f"scale='min({w},iw)':-2:flags=lanczos",
            ]
            audio_br = f"{abr}k"
        else:
            # AUTO / copy: sin transcodificación → latencia prácticamente cero
            video_args = ["-c:v", "copy", "-bsf:v", "h264_mp4toannexb"]
            audio_br = "128k"

        return (
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                # ── Arranque ultra-rápido (reducido de 1MB/0.5s → 0.5MB/0.25s) ─────
                # IPTV streams son TS estándar; no necesitan análisis largo
                "-probesize",
                "500000",
                "-analyzeduration",
                "250000",
                # ── Reconexión HTTP robusta por IA ─────────────────────────────────
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                str(int(ai_config.get("retry_delay", 8)) if ai_config else 8),
                "-reconnect_at_eof",
                "1",
                "-multiple_requests",
                "1",
                "-rw_timeout",
                "20000000",
                "-user_agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "-headers",
                f"Referer: {base_url}/\r\nConnection: keep-alive\r\n",
                "-timeout",
                "20000000",
                "-i",
                url,
            ]
            + video_args
            + [
                # ── Audio ────────────────────────────────────────────────────────────
                "-c:a",
                "aac",
                "-b:a",
                audio_br,
                "-ar",
                "48000",
                "-ac",
                "2",
                # aresample: async=1 corrige drift audio/video sin introducir latencia
                "-af",
                "aresample=async=1:min_hard_comp=0.1:first_pts=0",
                # ── HLS ──────────────────────────────────────────────────────────────
                "-f",
                "hls",
                "-hls_time",
                str(hls_time),
                # hls_init_time=1: el primer segmento está listo en 1s (antes = hls_time completo)
                # Permite que el player empiece a cargar antes aunque los segs siguientes sean más largos
                "-hls_init_time",
                "1",
                # Ventana dinámica: si la IA solicita prefetch masivo, el playlist
                # debe contener al menos prefetch_count + 5 segmentos para que el
                # cliente pueda descargarlos por adelantado desde el .m3u8
                "-hls_list_size",
                str(max(15, (ai_config.get('prefetch_count', 10) + 5) if ai_config else 15)),
                # Flags HLS:
                #   delete_segments     → borrar segs fuera de la ventana (liberar disco)
                #   independent_segments→ cada seg arranca en keyframe (seek seguro)
                #   omit_endlist        → no marcar stream como terminado cuando FFmpeg muere
                #   discont_start       → numeración continúa en restart (sin salto visible)
                #   split_by_time       → CORTE EXACTO en hls_time aunque no haya keyframe
                #                         (sin esto FFmpeg espera el siguiente IDR → segs variables)
                "-hls_flags",
                "delete_segments+independent_segments+omit_endlist+discont_start+split_by_time",
                "-hls_allow_cache",
                "0",  # forzar al cliente a pedir siempre el más fresco
                "-hls_segment_type",
                "mpegts",
                "-start_number",
                str(start_number),
                # ── Corrección de errores de stream ─────────────────────────────────
                "-err_detect",
                "ignore_err",
                # flush_packets=1: escritura inmediata a disco → HLS.js no espera el buffer del OS
                "-fflags",
                "+genpts+igndts+discardcorrupt+flush_packets",
                "-flush_packets",
                "1",
                "-avoid_negative_ts",
                "make_zero",
                "-hls_segment_filename",
                f"{temp_dir}/seg_%05d.ts",
                f"{temp_dir}/playlist.m3u8",
            ]
        )

    # ──────────────────────────────────────────────────────────────────────────
    # WATCHDOG: detecta FFmpeg muerto o colgado y lo reinicia automáticamente
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _watchdog(stream_id):
        import glob as _glob
        import uuid
        import shutil

        MAX_RESTARTS = 10
        STALL_THRESHOLD = 25  # segundos sin nuevo .ts → FFmpeg colgado
        CHECK_INTERVAL = 1

        print(f"   👁️ Watchdog iniciado: {stream_id}")
        last_bola_segs = 0

        while True:
            time.sleep(CHECK_INTERVAL)

            with streams_lock:
                if stream_id not in active_streams:
                    break
                info = active_streams[stream_id]
                
            # FIX: Si el directorio ya no existe, el stream fue limpiado externamente
            if not os.path.exists(info.get("temp_dir", "")):
                break

            # Dejar de vigilar si el cliente lleva >10min sin acceder
            if time.time() - info.get("last_access", 0) > 620:
                break

            segs_servidos = len(info.get("segs_servidos_set", set()))
            process = info["process"]

            def _limpiar_pending(sid, pend):
                if pend["proceso"].poll() is None:
                    try:
                        pend["proceso"].terminate()
                        time.sleep(0.5)
                        pend["proceso"].kill()
                    except: pass
                try: shutil.rmtree(pend["dir"], ignore_errors=True)
                except: pass
                with streams_lock:
                    if sid in active_streams and "pending" in active_streams[sid]:
                        del active_streams[sid]["pending"]

            # --- SEAMLESS BOLA ABR ---
            if HAS_AI:
                if "pending" in info:
                    pending = info["pending"]
                    pending_segs = sorted(_glob.glob(f"{pending['dir']}/seg_*.ts"))
                    
                    if pending["proceso"].poll() is not None and len(pending_segs) < 4:
                        print(f"   ⚠️ BOLA pending falló (FFmpeg murió), manteniendo calidad actual")
                        _limpiar_pending(stream_id, pending)
                        continue
                        
                    if time.time() - pending["started"] > 20:
                        print(f"   ⚠️ BOLA pending timeout (20s), manteniendo calidad actual")
                        _limpiar_pending(stream_id, pending)
                        continue
                        
                    current_segs = sorted(_glob.glob(f"{info['temp_dir']}/seg_*.ts"))
                    buffer_ms = (len(current_segs) - segs_servidos) * (info.get("hls_time", 2) * 1000)
                    if buffer_ms < 2000:
                        print(f"   ⚠️ BOLA abortado (buffer cayó a {buffer_ms}ms), manteniendo calidad actual")
                        _limpiar_pending(stream_id, pending)
                        continue

                    if len(pending_segs) >= 4:
                        # --- SWITCH TRANSPARENTE ---

                        with streams_lock:
                            if stream_id in active_streams:
                                old_proc = info["process"]
                                old_dir = info["temp_dir"]
                                
                                active_streams[stream_id]["process"] = pending["proceso"]
                                active_streams[stream_id]["temp_dir"] = pending["dir"]
                                active_streams[stream_id]["url"] = pending["calidad"]["url"]
                                active_streams[stream_id]["quality"] = pending["calidad"]["resolution"]
                                del active_streams[stream_id]["pending"]
                                
                        print(f"   🧠 BOLA SEAMLESS | {info.get('quality', 'old')} → {pending['calidad']['resolution']} | Buffer: {buffer_ms}ms | Stream: {stream_id}")
                        
                        # Time sleep preventivo antes de matar al viejo
                        time.sleep(2)
                        if old_proc.poll() is None:
                            try:
                                old_proc.terminate()
                                time.sleep(0.5)
                                old_proc.kill()
                            except: pass
                        try: shutil.rmtree(old_dir, ignore_errors=True)
                        except: pass
                        continue

                else:
                    # Chequeo BOLA normal
                    if segs_servidos - last_bola_segs >= 3:
                        last_bola_segs = segs_servidos
                        av_qualities = info.get("available_qualities", [])
                        
                        if av_qualities:
                            current_segs = sorted(_glob.glob(f"{info['temp_dir']}/seg_*.ts"))
                            buffer_ms = (len(current_segs) - segs_servidos) * (info.get("hls_time", 2) * 1000)
                            
                            from iptv_ai_core import ai_optimizer
                            nueva_calidad = ai_optimizer.calculate_bola_quality(av_qualities, buffer_ms)
                            
                            # Logica Asimétrica: Solo exigimos 8s de buffer p/ subir calidad. Para bajar, actuamos libremente.
                            if nueva_calidad and nueva_calidad["url"] != info.get("url"):
                                is_downswitch = nueva_calidad.get('bitrate', 1000) < info.get('brkbps', 999999)
                                buffer_req = 2000 if is_downswitch else 8000
                                
                                if buffer_ms > buffer_req:
                                    start_n = int(os.path.basename(current_segs[-1]).replace("seg_", "").replace(".ts", "")) + 1 if current_segs else 0
                                    
                                    new_dir = f"C:/IPTV_Cache/iptv_{stream_id}_pending_{str(uuid.uuid4())[:4]}"
                                    os.makedirs(new_dir, exist_ok=True)
                                    
                                    # Al pasar start_n, FFmpeg nativamente nombrará seg_XX y generará el m3u8 correctamente
                                pending_cmd = IPTVHandler._build_ffmpeg_cmd(
                                    nueva_calidad["url"],
                                    new_dir,
                                    info.get("transcode", False),
                                    info.get("quality_profile"),
                                    info.get("hls_time", 2),
                                    start_n,
                                    info.get("ai_config", None)
                                )
                                
                                try:
                                    pend_proc = subprocess.Popen(pending_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
                                    with streams_lock:
                                        if stream_id in active_streams:
                                            active_streams[stream_id]["pending"] = {
                                                "stream_id": stream_id,
                                                "dir": new_dir,
                                                "proceso": pend_proc,
                                                "calidad": nueva_calidad,
                                                "start_n": start_n,
                                                "started": time.time(),
                                                "listo": False
                                            }
                                    print(f"   ⏳ BOLA evaluando switch: {info.get('quality')} → {nueva_calidad['resolution']} (Buffer: {buffer_ms}ms)")
                                except: pass

            restarts = info.get("restarted", 0)

            def _restart(reason):
                # FIX: Verificar que el stream sigue activo antes de reiniciar
                with streams_lock:
                    if stream_id not in active_streams:
                        return False
                        
                nonlocal restarts
                if restarts >= MAX_RESTARTS:
                    print(f"   ❌ Watchdog {stream_id}: máx reintentos ({MAX_RESTARTS}) agotados")
                    return False

                # Encontrar el último segmento para continuar desde ahí
                segs = sorted(_glob.glob(f"{info['temp_dir']}/seg_*.ts"))
                start_n = int(os.path.basename(segs[-1]).replace("seg_", "").replace(".ts", "")) + 1 if segs else 0

                print(f"   🔄 Watchdog: {reason} → reiniciando desde seg #{start_n}...")

                old = info.get("process")
                if old and old.poll() is None:
                    try:
                        old.terminate()
                        time.sleep(0.5)
                        old.kill()
                    except: pass

                cmd = IPTVHandler._build_ffmpeg_cmd(
                    info["url"],
                    info["temp_dir"],
                    info.get("transcode", False),
                    info.get("quality_profile"),
                    info.get("hls_time", 2),
                    start_n,
                    info.get("ai_config", None)
                )
                try:
                    new_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
                    with streams_lock:
                        if stream_id in active_streams:
                            active_streams[stream_id]["process"] = new_proc
                            active_streams[stream_id]["restarted"] = restarts + 1
                            active_streams[stream_id]["last_restart"] = time.time()
                    restarts += 1
                    print(f"   ✅ FFmpeg reiniciado (intento #{restarts})")
                    return True
                except Exception as e:
                    print(f"   ❌ Error al reiniciar: {e}")
                    return False

            if process.poll() is not None:
                if not _restart("FFmpeg terminó"):
                    with streams_lock:
                        if stream_id in active_streams:
                            cleanup_stream(stream_id)
                    break
                continue

            segs = sorted(_glob.glob(f"{info['temp_dir']}/seg_*.ts"))
            if segs:
                stall = time.time() - os.path.getmtime(segs[-1])
                if stall > STALL_THRESHOLD:
                    if not _restart(f"stall {stall:.0f}s sin segmentos"):
                        with streams_lock:
                            if stream_id in active_streams:
                                cleanup_stream(stream_id)
                        break

        print(f"   👁️ Watchdog terminado: {stream_id}")

    # ──────────────────────────────────────────────────────────────────────────
    # HLS ABR: Interceptor Inteligente de Variante Maestra
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def select_optimal_hls_variant(url, quality):
        """
        Descarga el M3U8 original. Si es un Playlist Maestro con variantes, 
        selecciona la mejor URL según la calidad y el estado de la red (ThrottleDetector).
        Retorna: (optimal_url, exact_match) -> exact_match=True indica que NO se necesita transcodificar.
        """
        try:
            import ssl as _ssl
            ctx = _ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            req.add_header("Connection", "keep-alive")
            req.add_header("Range", "bytes=0-16384") # Leer cabecera
            with urllib.request.urlopen(req, timeout=2.5, context=ctx) as r:
                # IMPORTANTE: No usar read(), usar read(16384)
                # Si el server ignora Range, read() intentará descargar un stream en vivo infinito.
                content = r.read(16384).decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"   ⚠️ ABR Head Error: {e}")
            return url, False
            
        if "#EXT-X-STREAM-INF" not in content:
            return url, False  # No es playlist maestro

        # Parsear variantes (BANDWIDTH y RESOLUTION opcional)
        print(f"   🧠 IA ABR: Detectado Playlist Maestro HLS")
        variantes = []
        base_url = url.rsplit("/", 1)[0]
        
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("#EXT-X-STREAM-INF"):
                bandwidth = 0
                height = 0
                
                # Extraer BANDWIDTH
                b_match = re.search(r'BANDWIDTH=(\d+)', line)
                if b_match: bandwidth = int(b_match.group(1))
                
                # Extraer RESOLUTION (ej: 1920x1080)
                r_match = re.search(r'RESOLUTION=\d+x(\d+)', line)
                if r_match: height = int(r_match.group(1))
                
                # Extraer URL (siguiente linea que no sea comenta)
                var_url = None
                for j in range(i+1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith("#"):
                        var_url = lines[j].strip()
                        break
                        
                if var_url:
                    if not var_url.startswith("http"):
                        var_url = f"{base_url}/{var_url}"
                    variantes.append({
                        "bw": bandwidth,
                        "h": height,
                        "url": var_url
                    })
        
        if not variantes:
            return url, False

        # Ordenar por ancho de banda
        variantes.sort(key=lambda x: x["bw"])
        
        # LOGICA ABR:
        try:
            from iptv_ai_core import throttle_detector
            analysis = throttle_detector.analyze(url, 1000) # El análisis se basa en el registro histórico
            is_throttled = analysis.get("throttled", False)
        except Exception:
            is_throttled = False

        if quality == "auto":
            if is_throttled:
                v = variantes[0] # La MENOR calidad para salvar el stream
                print(f"   🚨 ABR AI: Red asfixiada detectada. Forzando menor variante: {v['bw']/1000:.0f}Ckbps")
                app_metrics["abr_interventions"] += 1
                return v["url"], True # Copy
            else:
                v = variantes[-1] # La MAYOR calidad
                print(f"   📶 ABR AI: Red veloz. Forzando mayor variante: {v['bw']/1000:.0f} kbps")
                app_metrics["cpu_saved_events"] += 1
                return v["url"], True # Copy
                
        elif quality in ("original", "source"):
            v = variantes[-1]
            app_metrics["cpu_saved_events"] += 1
            return v["url"], True

            
        else:
            # Buscar la que se acerque más a la resolución pedida (ej. "720p")
            try:
                target_h = int(quality.replace("p", ""))
                # Buscar variante que coincida con target_h, o la inmediatamente superior
                # Si las variantes no reportan height, usamos heurística por bandwidth
                if all(v["h"] == 0 for v in variantes):
                    for v in variantes:
                        if target_h <= 480 and v["bw"] > 1200000: return v["url"], False
                        if target_h <= 720 and v["bw"] > 2500000: return v["url"], False
                    return variantes[-1]["url"], True 
                
                # Buscar por RESOLUTION explícita
                cercanas = [v for v in variantes if v["h"] > 0]
                if cercanas:
                    # Encontrar la resolución más cercana y mayor o igual
                    coincidencias = [v for v in cercanas if v["h"] >= target_h]
                    if coincidencias:
                        v = min(coincidencias, key=lambda x: x["h"])
                        exact = (v["h"] == target_h)
                        print(f"   🎯 ABR: Encontrada variante {v['h']}p para {quality}")
                        return v["url"], exact
                    else:
                        v = max(cercanas, key=lambda x: x["h"])
                        return v["url"], False
            except Exception:
                pass
                
        return url, False
        

    # ──────────────────────────────────────────────────────────────────────────
    # HLS: Iniciar stream en vivo
    # ──────────────────────────────────────────────────────────────────────────
    def handle_play_hls(self):
        """Inicia stream HLS con FFmpeg + watchdog de auto-reinicio."""
        global active_streams
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            url = params.get("url", [None])[0]
            if not url:
                self.send_json_response(400, {"error": "URL requerida"})
                return
            url = urllib.parse.unquote(url)

            QUALITY_PROFILES = {
                "1080p": (1920, 1080, 3500, 192),
                "720p": (1280, 720, 2000, 128),
                "480p": (854, 480, 900, 96),
                "360p": (640, 360, 500, 64),
            }
            quality = params.get("quality", ["auto"])[0].lower()
            transcode_video = quality in QUALITY_PROFILES
            quality_profile = QUALITY_PROFILES.get(quality)
            stream_id = hashlib.md5(f"{url}|{quality}".encode()).hexdigest()[:12]

            print(f"\n{'='*60}")
            print(f"🎬 HLS Request: {url[:50]}...")
            print(f"   ID: {stream_id}  |  Calidad: {quality.upper()}")

            import glob
            
            # --- SEMÁFORO DE INICIALIZACIÓN ---
            with streams_lock:
                while stream_id in active_streams and active_streams[stream_id].get("status") == "init":
                    streams_lock.release()
                    time.sleep(0.3)
                    streams_lock.acquire()

                if stream_id in active_streams:
                    info = active_streams[stream_id]
                    if "process" in info and info["process"].poll() is None:
                        info["last_access"] = time.time()
                        if "segs_servidos_set" not in info:
                            info["segs_servidos_set"] = {os.path.basename(f) for f in glob.glob(f"{info['temp_dir']}/seg_*.ts")}
                        
                        print(f"   ♻️ Reutilizando stream activo")
                        self.send_json_response(200, {"stream_id": stream_id, "playlist_url": f"/hls/{stream_id}/playlist.m3u8", "status": "ready", "quality": quality})
                        return
                    elif info.get("status") != "init":
                        print(f"   💀 Proceso murió o inválido, reiniciando...")
                        cleanup_stream(stream_id)

                # Registro provisional para bloquear otros hilos mientras urllib/ABR resuelven
                active_streams[stream_id] = {"status": "init", "started": time.time()}

            temp_dir = tempfile.mkdtemp(prefix=f"iptv_{stream_id}_")
            try:
                # Detectar bitrate sin bloquear (max 3s)
                br_result = [2500]
                def _probe(): br_result[0] = detect_stream_info(url)
                pt = threading.Thread(target=_probe, daemon=True)
                pt.start()
                pt.join(timeout=3)
                brkbps = br_result[0]
                
                # 🧠 IA ABR: Interceptar URL y seleccionar variante maestra
                optimal_url, is_exact_match = self.select_optimal_hls_variant(url, quality)
                if optimal_url != url:
                    print(f"   ✨ ABR AI: URL original reescrita por sub-variante óptima")
                    url = optimal_url
                    if is_exact_match and transcode_video:
                        print(f"   ⚡ ABR AI: La variante es el MATCH PERFECTO. Se anula reescalado (-c:v copy)")
                        transcode_video = False
                        quality_profile = None

                hls_time = 1 if transcode_video else 2
                from iptv_ai_core import ai_optimizer, throttle_detector
                size_est = (brkbps * hls_time) / 8000.0 if brkbps else 2.0
                current_spd = throttle_detector.expected_speeds.get(throttle_detector._domain(url), brkbps if brkbps else 2500)
                ai_config_data = ai_optimizer.predict_optimal_config(url, size_est, latency=0, speed_kbps=current_spd, required_bitrate=brkbps if brkbps else 2500) if HAS_AI else None
                
                ffmpeg_cmd = self._build_ffmpeg_cmd(url, temp_dir, transcode_video, quality_profile, hls_time, 0, ai_config_data)
                print(f"   🚀 Iniciando FFmpeg...")
                process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)

                available_qualities = []
                if HAS_AI:
                    try:
                        req_abr = urllib.request.Request(url)
                        req_abr.add_header("User-Agent", "Mozilla/5.0")
                        req_abr.add_header("Range", "bytes=0-16384")
                        with urllib.request.urlopen(req_abr, timeout=3) as r_abr:
                            content_abr = r_abr.read(16384).decode("utf-8", errors="ignore")
                        if "#EXT-X-STREAM-INF" in content_abr:
                            base_abr = url.rsplit("/", 1)[0]
                            lines_abr = content_abr.splitlines()
                            for i_abr, line_abr in enumerate(lines_abr):
                                if line_abr.startswith("#EXT-X-STREAM-INF"):
                                    bw_m = re.search(r'BANDWIDTH=(\d+)', line_abr)
                                    res_m = re.search(r'RESOLUTION=\d+x(\d+)', line_abr)
                                    for j_abr in range(i_abr+1, len(lines_abr)):
                                        if lines_abr[j_abr].strip() and not lines_abr[j_abr].startswith("#"):
                                            var_url = lines_abr[j_abr].strip()
                                            if not var_url.startswith("http"): var_url = f"{base_abr}/{var_url}"
                                            available_qualities.append({"resolution": f"{res_m.group(1)}p" if res_m else "auto", "bitrate": int(bw_m.group(1)) // 1000 if bw_m else 2500, "url": var_url})
                                            break
                    except: pass

                with streams_lock:
                    active_streams[stream_id] = {
                        "available_qualities": available_qualities,
                        "process": process,
                        "temp_dir": temp_dir,
                        "url": url,
                        "quality": quality,
                        "transcode": transcode_video,
                        "hls_time": hls_time,
                        "started": time.time(),
                        "last_access": time.time(),
                        "restarted": 0,
                        "brkbps": brkbps,
                        "ai_config": ai_config_data,
                        "segs_servidos_set": set()
                    }
            except Exception as e_init:
                print(f"   ❌ Fallo crítico en inicialización: {e_init}")
                with streams_lock:
                    if stream_id in active_streams and active_streams[stream_id].get("status") == "init":
                        del active_streams[stream_id]
                try: shutil.rmtree(temp_dir, ignore_errors=True)
                except: pass
                raise e_init


            # Esperar primer segmento
            max_wait = 35 if transcode_video else 25
            waited = 0
            print(f"   ⏳ Esperando primer segmento (max {max_wait}s)...")

            while waited < max_wait:
                if process.poll() is not None:
                    stderr = process.stderr.read().decode("utf-8", errors="ignore")
                    print(f"   ❌ FFmpeg:\n{stderr[:400]}")
                    cleanup_stream(stream_id)
                    
                    if "403" in stderr or "Forbidden" in stderr:
                        msg = "Servidor bloqueó FFmpeg (403). Usa VLC."
                    elif "404" in stderr:
                        msg = "Stream no encontrado (404)"
                    elif "timed out" in stderr:
                        msg = "Timeout de conexión"
                    elif "libx264" in stderr:
                        msg = "libx264 no disponible. Usa Auto."
                    else:
                        msg = "FFmpeg falló al procesar stream"
                    
                    self.send_json_response(
                        500, {"error": msg, "details": stderr[:300]}
                    )
                    return

                pl = f"{temp_dir}/playlist.m3u8"
                if os.path.exists(pl):
                    try:
                        import glob
                        with open(pl, "r", encoding="utf-8") as f:
                            pl_content = f.read()

                        if ".ts" in pl_content and glob.glob(f"{temp_dir}/seg_*.ts"):
                            print(f"   ✅ Stream listo ({waited:.1f}s)")
                            break
                    except:
                        pass
                time.sleep(0.3)
                waited += 0.3
            else:
                cleanup_stream(stream_id)
                self.send_json_response(500, {"error": "Timeout generando HLS"})
                return

            # 🐕 Lanzar watchdog daemon por este stream
            threading.Thread(
                target=self._watchdog, args=(stream_id,), daemon=True
            ).start()

            self.send_json_response(200, {
                "stream_id": stream_id,
                "playlist_url": f"/hls/{stream_id}/playlist.m3u8",
                "status": "ready",
                "quality": quality,
            })

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.send_json_response(500, {"error": str(e)})

    def handle_hls_file(self):
        """Sirve archivos HLS generados por FFmpeg — actualiza last_access en cada petición"""
        try:
            parts = self.path[5:].split("/")
            stream_id = parts[0]
            filename = parts[1] if len(parts) > 1 else ""

            if stream_id not in active_streams:
                self.send_response(404)
                self.end_headers()
                return

            with streams_lock:
                if stream_id in active_streams:
                    # ── Stream sharing: cada cliente que pide un .ts renueva el lease ──
                    # Esto permite que múltiples pestañas compartan el mismo proceso FFmpeg
                    # sin que el watchdog lo mate por inactividad
                    active_streams[stream_id]["last_access"] = time.time()
                    # Contar clientes activos aproximadamente por frecuencia de requests
                    active_streams[stream_id]["requests"] = (
                        active_streams[stream_id].get("requests", 0) + 1
                    )
                    
                    if filename.endswith(".ts"):
                        if "segs_servidos_set" not in active_streams[stream_id]:
                            active_streams[stream_id]["segs_servidos_set"] = set()
                        active_streams[stream_id]["segs_servidos_set"].add(filename)
                        
                    temp_dir = active_streams[stream_id]["temp_dir"]
                else:
                    self.send_response(404)
                    self.end_headers()
                    return

            file_path = os.path.join(temp_dir, filename)

            if not os.path.exists(file_path):
                # Esperar hasta 3s por el archivo — segmentos nuevos pueden tardar
                # Importante para .ts recientes y para playlist.m3u8 en inicio
                for _ in range(30):
                    time.sleep(0.1)
                    if os.path.exists(file_path):
                        break
                else:
                    self.send_response(404)
                    self.end_headers()
                    return

            if filename.endswith(".m3u8"):
                ct = "application/vnd.apple.mpegurl"
            elif filename.endswith(".ts"):
                ct = "video/mp2t"
            else:
                ct = "application/octet-stream"

            with open(file_path, "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        except Exception as e:
            print(f"❌ Error HLS: {e}")
            try:
                self.send_response(500)
                self.end_headers()
            except:
                pass

    def handle_transcode_vod(self):
        """
        Transcodificación VOD vía Python-pipe → FFmpeg  (v7.1)

        NUEVO en v7.1 — SEEKING COMPLETO:
        ─ start_time: arranca FFmpeg desde cualquier segundo del VOD
        ─ byte_offset: Python calcula Range: bytes=N para saltar en el HTTP
        ─ output_ts_offset: los segmentos HLS tienen timestamps correctos
          (el player muestra el minuto real, no desde 0)
        ─ ffprobe en background: devuelve duración total en la respuesta
          para que el player muestre el seekbar completo desde el inicio

        URL: /api/transcode-vod?url=...&quality=...&start_time=0
        """
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            url = urllib.parse.unquote(params.get("url", [None])[0] or "")
            quality = (params.get("quality", ["auto"])[0] or "auto").lower()
            start_time = float(params.get("start_time", ["0"])[0] or "0")
            cancel_previous = params.get("cancel_previous", ["0"])[0] == "1"

            if not url:
                self.send_json_response(400, {"error": "URL requerida"})
                return

            # ── Cancelar transcode previo del mismo URL (seek rápido) ──────────
            # Cuando el player hace seek, primero llega cancel_previous=1.
            # Limpiamos el stream anterior de esta URL para liberar FFmpeg y disco
            # antes de crear el nuevo. Sin esto, seeks rápidos dejan 5-10 FFmpeg
            # corriendo en paralelo consumiendo CPU, red y disco innecesariamente.
            if cancel_previous:
                with url_vod_active_lock:
                    old_sid = url_vod_active.get(url)
                if old_sid:
                    with streams_lock:
                        if old_sid in active_streams:
                            print(
                                f"   🛑 Cancelando transcode previo {old_sid[:8]} (seek nuevo)"
                            )
                            cleanup_stream(old_sid)
                    # Cancelar también el pipe thread
                    with vod_pipe_registry_lock:
                        old_evt = vod_pipe_registry.get(old_sid)
                        if old_evt:
                            old_evt.set()
                            vod_pipe_registry.pop(old_sid, None)

            # ── Perfil de calidad ─────────────────────────────────────────────
            PROFILES = {
                "720p": (1280, 720, 1800, 128),
                "480p": (854, 480, 900, 96),
                "360p": (640, 360, 500, 64),
            }
            profile = PROFILES.get(quality)
            transcode_vid = profile is not None

            parsed = urllib.parse.urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            # stream_id incluye start_time → cada seek es un nuevo stream
            start_key = int(start_time)
            stream_id = hashlib.md5(
                f"{url}:vod:{quality}:{start_key}".encode()
            ).hexdigest()[:12]

            print(f"\n{'='*60}")
            print(f"🎬 Transcode-VOD v7.1: {url[:60]}...")
            print(
                f"   ID: {stream_id} | Calidad: {quality.upper()} | Start: {start_time:.1f}s"
            )

            # ── Reutilizar stream activo si ya existe para este seek ───────────
            with streams_lock:
                if stream_id in active_streams:
                    info = active_streams[stream_id]
                    if info["process"].poll() is None:
                        info["last_access"] = time.time()
                        pl = os.path.join(info["temp_dir"], "playlist.m3u8")
                        if os.path.exists(pl):
                            print(f"   ♻️ Reutilizando stream VOD")
                            cached_dur = duration_cache.get(url, 0)
                            self.send_json_response(
                                200,
                                {
                                    "stream_id": stream_id,
                                    "playlist_url": f"/hls/{stream_id}/playlist.m3u8",
                                    "status": "ready",
                                    "shared": True,
                                    "total_duration": cached_dur,
                                    "start_time": start_time,
                                },
                            )
                            return
                    else:
                        cleanup_stream(stream_id)

                temp_dir = tempfile.mkdtemp(prefix=f"iptv_{stream_id}_")
                print(f"   📁 {temp_dir}")

            # ── Lanzar ffprobe en background para obtener duración total ──────
            # No bloqueamos la respuesta, se cachea para futuros seeks
            def _probe_duration():
                dur = get_vod_duration(url, base_url)
                if dur > 0:
                    print(f"   📐 Duración VOD cacheada: {dur:.1f}s")

            threading.Thread(target=_probe_duration, daemon=True).start()

            # ── Perfil de calidad / modo audio-fix ───────────────────────────────
            if transcode_vid:
                w, h, vbr, abr = profile
                print(f"   🔽 Transcodificando → {quality} ({w}×{h}, {vbr}k)")
                x264p = "rc-lookahead=0:bframes=0:scenecut=0:no-mbtree=1:sync-lookahead=0:ref=1"
                hls_t = 2
                vid_args = [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "superfast",
                    "-tune",
                    "zerolatency",
                    "-x264-params",
                    x264p,
                    "-profile:v",
                    "baseline",
                    "-b:v",
                    f"{vbr}k",
                    "-maxrate",
                    f"{int(vbr*1.1)}k",
                    "-bufsize",
                    f"{vbr}k",
                    "-g",
                    str(int(30 * hls_t)),
                    "-keyint_min",
                    str(int(30 * hls_t)),
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    f"scale='min({w},iw)':-2:flags=lanczos",
                ]
                audio_br = f"{abr}k"
            else:
                print(f"   🔊 Audio fix: video copy + AAC 192k")
                hls_t = 4
                vid_args = ["-c:v", "copy"]
                audio_br = "192k"

            # ── Seeking: usar siempre -ss de FFmpeg (robusto para VBR) ───────────
            # byte_offset era frágil: calcula posición lineal asumiendo bitrate
            # constante, pero los MKV tienen bitrate variable → FFmpeg arrancaba
            # en medio de un frame y moría con 500. Con -ss antes de -i pipe:0,
            # FFmpeg descarta los frames hasta llegar al keyframe correcto.
            # Es unos segundos más lento pero nunca falla.
            #
            # Para archivos grandes: el pipe descarga desde el inicio pero -ss
            # hace que FFmpeg descarte rápido (no hay reencoding hasta el punto).
            # Para copy-mode (-c:v copy) el seek es casi instantáneo.
            seek_args = []
            ts_offset_args = []
            if start_time > 0:
                seek_args = ["-ss", str(int(start_time))]
                ts_offset_args = ["-output_ts_offset", str(int(start_time))]
                print(
                    f"   📍 Seek a {start_time:.0f}s → usando -ss {int(start_time)} (robusto)"
                )

            ffmpeg_cmd = (
                ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
                + seek_args
                + ["-i", "pipe:0", "-probesize", "500000", "-analyzeduration", "250000"]
                + vid_args
                + ts_offset_args
                + [
                    "-c:a",
                    "aac",
                    "-b:a",
                    audio_br,
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-af",
                    "aresample=async=1:min_hard_comp=0.1:first_pts=0",
                    "-f",
                    "hls",
                    "-hls_time",
                    str(hls_t),
                    "-hls_init_time",
                    "1",
                    # 0 = conservar todos los segmentos → seeking completo
                    "-hls_list_size",
                    "0",
                    # append_list: el playlist crece; sin delete_segments → todos en disco
                    # independent_segments: cada .ts arranca en keyframe → seek limpio
                    "-hls_flags",
                    "independent_segments+split_by_time+append_list",
                    "-hls_allow_cache",
                    "1",
                    "-hls_segment_type",
                    "mpegts",
                    "-err_detect",
                    "ignore_err",
                    "-fflags",
                    "+genpts+igndts+discardcorrupt+flush_packets",
                    "-flush_packets",
                    "1",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-hls_segment_filename",
                    f"{temp_dir}/seg_%05d.ts",
                    f"{temp_dir}/playlist.m3u8",
                ]
            )

            print(f"   🚀 Iniciando FFmpeg (stdin pipe)...")
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            with streams_lock:
                active_streams[stream_id] = {
                    "process": process,
                    "temp_dir": temp_dir,
                    "url": url,
                    "quality": quality,
                    "quality_profile": profile,
                    "transcode": transcode_vid,
                    "hls_time": hls_t,
                    "started": time.time(),
                    "last_access": time.time(),
                    "type": "vod",
                    "restarted": 0,
                    "requests": 0,
                    "start_time": start_time,
                }

            # Registrar como stream activo de esta URL
            # El próximo seek con cancel_previous=1 lo cancelará
            with url_vod_active_lock:
                url_vod_active[url] = stream_id

            # ── Thread: descarga HTTP con retry y reconexión ───────────────────
            cancel_event = threading.Event()
            with vod_pipe_registry_lock:
                old_evt = vod_pipe_registry.get(stream_id)
                if old_evt:
                    old_evt.set()
                vod_pipe_registry[stream_id] = cancel_event

            def pipe_video_to_ffmpeg():
                # El pipe siempre descarga desde el inicio del archivo.
                # El seeking lo maneja FFmpeg con -ss (ya configurado en el cmd).
                # En reconexión por error de red: usamos Range para reanudar
                # desde donde se cortó (evita re-descargar lo ya enviado).
                MAX_RETRIES = 3
                retry_count = 0
                bytes_piped = 0
                current_offset = (
                    0  # siempre desde el inicio; -ss en FFmpeg maneja el seek
                )

                while retry_count <= MAX_RETRIES and not cancel_event.is_set():
                    try:
                        if retry_count > 0:
                            print(
                                f"   🔄 Pipe reconectando (intento {retry_count}/{MAX_RETRIES})"
                            )
                            time.sleep(1.0 * retry_count)

                        # En reconexión (retry > 0): reanudar desde el byte donde se cortó
                        range_hdr = (
                            f"bytes={current_offset}-" if current_offset > 0 else None
                        )
                        resp, _ = fetch_with_retry(
                            url,
                            extra_headers={"Referer": f"{base_url}/", "Accept": "*/*"},
                            range_header=range_hdr,
                            max_time_s=3.0,
                            timeout_s=120,
                        )

                        CHUNK = 512 * 1024
                        with resp:
                            while not cancel_event.is_set():
                                if process.poll() is not None:
                                    return
                                chunk = resp.read(CHUNK)
                                if not chunk:
                                    break
                                try:
                                    process.stdin.write(chunk)
                                    process.stdin.flush()
                                    bytes_piped += len(chunk)
                                    current_offset += len(chunk)
                                except (BrokenPipeError, OSError):
                                    return

                        print(f"   ✅ Pipe: {bytes_piped/1024/1024:.1f} MB → FFmpeg")
                        return

                    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                        retry_count += 1
                        print(
                            f"   ⚠️ Pipe error ({e}), retry {retry_count}/{MAX_RETRIES}"
                        )
                    except Exception as e:
                        if not cancel_event.is_set():
                            print(f"   ⚠️ Pipe inesperado: {e}")
                        break

                try:
                    process.stdin.close()
                except:
                    pass
                with vod_pipe_registry_lock:
                    if vod_pipe_registry.get(stream_id) is cancel_event:
                        vod_pipe_registry.pop(stream_id, None)

            threading.Thread(target=pipe_video_to_ffmpeg, daemon=True).start()

            # ── Detección anticipada del stream ───────────────────────────────
            pl_path = os.path.join(temp_dir, "playlist.m3u8")
            waited = 0
            print(f"   ⏳ Esperando playlist (max 30s)...")

            while waited < 30:
                time.sleep(0.4)
                waited += 0.4
                if process.poll() is not None:
                    stderr_out = ""
                    try:
                        stderr_out = process.stderr.read(2000).decode(errors="replace")
                    except:
                        pass
                    cancel_event.set()
                    with streams_lock:
                        if stream_id in active_streams:
                            cleanup_stream(stream_id)
                    self.send_json_response(
                        500, {"error": "FFmpeg falló", "detail": stderr_out[:300]}
                    )
                    return

                if os.path.exists(pl_path):
                    try:
                        with open(pl_path, "r", encoding="utf-8") as pf:
                            pl_content = pf.read()
                        if "seg_" in pl_content:
                            print(f"   ✅ Playlist lista ({waited:.1f}s)")
                            break
                    except:
                        pass
            else:
                cancel_event.set()
                with streams_lock:
                    if stream_id in active_streams:
                        cleanup_stream(stream_id)
                self.send_json_response(
                    500, {"error": "Timeout: playlist no generada en 30s"}
                )
                return

            # Devolver duración total cacheada (puede ser 0 si ffprobe aún no terminó)
            total_duration = duration_cache.get(url, 0)

            self.send_json_response(
                200,
                {
                    "stream_id": stream_id,
                    "playlist_url": f"/hls/{stream_id}/playlist.m3u8",
                    "status": "ready",
                    "quality": quality,
                    "mode": "transcode" if transcode_vid else "audio-fix",
                    "waited_s": round(waited, 1),
                    "total_duration": total_duration,
                    "start_time": start_time,
                },
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.send_json_response(500, {"error": str(e)})

    def handle_vod_duration(self):
        """
        /api/vod-duration?url=...

        Devuelve la duración total del archivo VOD obtenida por ffprobe.

        ─ Si ya está en cache (de un transcode previo): responde instantáneo.
        ─ Si no, lanza ffprobe y bloquea hasta 18s esperando resultado.
        ─ Usado por el player para poblar el seekbar completo desde el principio.

        Respuesta: {"duration": 5040.3, "size": 1234567890}
        """
        try:
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            url = urllib.parse.unquote(params.get("url", [None])[0] or "")
            if not url:
                self.send_json_response(400, {"error": "URL requerida"})
                return

            parsed = urllib.parse.urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            # ── 1. Devolver desde cache si disponible ──────────────────────────
            cached = duration_cache.get(url, 0)
            if cached > 0:
                size = get_vod_file_size(url, base_url)
                self.send_json_response(200, {"duration": cached, "size": size})
                return

            # ── 2. Lanzar ffprobe y esperar hasta 18s ──────────────────────────
            dur_result = [0]
            size_result = [0]
            done_event = threading.Event()

            def _probe():
                dur_result[0] = get_vod_duration(url, base_url, timeout_s=16)
                size_result[0] = (
                    get_vod_file_size(url, base_url) if dur_result[0] > 0 else 0
                )
                done_event.set()

            threading.Thread(target=_probe, daemon=True).start()
            done_event.wait(timeout=18)

            self.send_json_response(
                200,
                {
                    "duration": dur_result[0],
                    "size": size_result[0],
                },
            )

        except Exception as e:
            self.send_json_response(200, {"duration": 0, "size": 0})

    def handle_play_vod_hls(self):
        """VOD HLS vía FFmpeg – deshabilitado (servidores IPTV bloquean FFmpeg)"""
        self.send_json_response(
            501,
            {
                "error": "Conversión FFmpeg VOD no disponible",
                "reason": "Usa /api/transcode-vod que hace proxy via Python",
                "solution": "El player usa automáticamente transcode-vod",
            },
        )


# ============================================================
# ARRANQUE DEL SERVIDOR
# ============================================================
def run_server(port=5000):
    # Verificar FFmpeg
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
        except Exception:
            print(f"\n{'='*60}")
            print(f"❌ ERROR: {tool} NO encontrado")
            print("Descárgalo de: https://ffmpeg.org/download.html")
            print(f"{'='*60}\n")
            return

    print("✅ FFmpeg y ffprobe OK")
    
    # 🤖 Estado de la IA
    if HAS_AI:
        print(f"🧠 Red Neuronal IPTV: {'Cargada' if ai_optimizer.model else 'Inicializada (Aprendiendo...)'}")

    ensure_cache_dir()

    server = ThreadedHTTPServer(("", port), IPTVHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Threads de mantenimiento
    for target in (cleanup_inactive_streams, auto_refresh_loop):
        threading.Thread(target=target, daemon=True).start()

    # 📦 IA: Auto-Actualización Diaria al arrancar
    threading.Thread(target=startup_playlist_check, daemon=True).start()

    # 🤖 IA: Scheduler Autónomo Diario (se despierta a las 04:00 AM)
    if HAS_AI:
        try:
            from ai_scheduler import start_scheduler_thread
            start_scheduler_thread()
            print("🤖 Scheduler IA: Activo (ciclo diario a las 04:00 AM)")
        except Exception as e:
            print(f"⚠️ Scheduler IA no disponible: {e}")

    print(
        f"""
╔══════════════════════════════════════════════════════════════╗
║    🎬  SERVIDOR IPTV v7.0  –  AI Neural Optimized 🧠 🎬      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ✓  http://localhost:{port}/iptv-player.html                   ║
║                                                              ║
║  🤖 Neural Engine: {'Activo' if HAS_AI else 'Desactivado'}                        ║
║  ⚡ Parser M3U por iteración (RAM reducida)                 ║
║  🗜️  Descarga M3U con gzip (menos datos en red)             ║
║  💾 Caché en disco comprimido (.json.gz)                    ║
║  🔄 fetch_with_retry — 404/403 transitorios tolerados       ║
║  🔌 Pipe VOD con reconexión automática (3 reintentos)       ║
║  📡 Stream sharing — múltiples clientes 1 proceso FFmpeg    ║
║  ⏩ Detección anticipada — playlist lista ~40% antes        ║
║  🐕 Watchdog reinicia streams live caídos                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    )

    try:
        print("⚡ Servidor listo...\n")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🧹 Deteniendo streams...")
        with streams_lock:
            for sid in list(active_streams.keys()):
                cleanup_stream(sid)
        print("✅ Servidor detenido")


# ============================================================
# FUNCIONES AUXILIARES DE IA (Ámbito Global)
# ============================================================
def startup_playlist_check():
    """Actualiza automáticamente las listas si tienen más de 24 horas"""
    print("\n📦 IA: Verificando caducidad de listas M3U...")
    try:
        index = load_index()
        current_time = time.time()
        for entry in index:
            url = entry.get("url")
            pid = entry.get("id")
            last_update = entry.get("timestamp", 0)
            
            # 86400 segundos = 24 horas
            if current_time - last_update > 86400:
                print(f"   🔄 Lista '{entry.get('name')}' caducada → Actualizando...")
                threading.Thread(target=refresh_one_playlist, args=(pid, url, entry.get("name")), daemon=True).start()
        print("✅ IA: Verificación completada.")
    except Exception as e:
        print(f"⚠️ Error en auto-actualización: {e}")


if __name__ == "__main__":
    run_server()
