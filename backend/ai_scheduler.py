#!/usr/bin/env python3
"""
ai_scheduler.py - Cerebro Autónomo Diario del Sistema IPTV
═══════════════════════════════════════════════════════════
Daemon que se ejecuta en background junto al servidor. Se "despierta"
una vez al día (por defecto 04:00 AM) para ejecutar tareas de mantenimiento:

  1. 📋 ACTUALIZAR PLAYLISTS  — Refresca todas las listas M3U del índice
  2. 🔍 ANALIZAR LOGS IA      — Revisa ai_decisions.log y detecta patrones de fallo
  3. 🧠 EVOLUCIONAR CEREBRO   — Dispara evolve_brain() con todas las experiencias acumuladas
  4. 📊 GENERAR REPORTE       — Escribe un resumen diario en C:/IPTV-Log/daily_report.txt

Uso standalone (para pruebas):
    python ai_scheduler.py --now      # Ejecuta ciclo inmediatamente y sale
    python ai_scheduler.py --daemon   # Corre como daemon (loop infinito)

El servidor iptv-server.py lo arranca automáticamente como thread daemon.
"""

import sys
import io

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except: pass

import os
import time
import json
import sqlite3
import threading
import datetime
import re
import logging
import logging.handlers
import urllib.request
import urllib.parse
import gzip

# ─────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────
CACHE_DIR       = os.path.join(os.path.dirname(__file__), '..', 'cache')
LOG_DIR         = os.path.join(os.path.dirname(__file__), '..', 'logs')
DB_PATH         = os.path.join(CACHE_DIR, "iptv_permanent_memory.db")
CACHE_INDEX     = os.path.join(CACHE_DIR, "index.json")
AI_LOG          = os.path.join(LOG_DIR, "ai_decisions.log")
DAILY_LOG       = os.path.join(LOG_DIR, "daily_report.txt")
SCHEDULER_LOG   = os.path.join(LOG_DIR, "scheduler.log")
WAKE_HOUR       = 7   # Hora de despertar (07:15 AM — PC ya encendida)
WAKE_MINUTE     = 15

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Configurar logger propio
sched_logger = logging.getLogger("AI_SCHEDULER")
handler = logging.handlers.RotatingFileHandler(SCHEDULER_LOG, encoding="utf-8", maxBytes=10_000_000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
sched_logger.addHandler(handler)
sched_logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────
# TAREA 1: ACTUALIZAR PLAYLISTS
# ─────────────────────────────────────────────────
def task_refresh_playlists():
    """Descarga y refresca todas las playlists del índice."""
    sched_logger.info("📋 TAREA 1: Actualizando playlists M3U...")
    results = []

    try:
        if not os.path.exists(CACHE_INDEX):
            sched_logger.info("  Sin índice de playlists — saltando")
            return results

        with open(CACHE_INDEX, "r", encoding="utf-8") as f:
            index = json.load(f)

        if not index:
            sched_logger.info("  Índice vacío — sin playlists que actualizar")
            return results

        for entry in index:
            url  = entry.get("url", "")
            name = entry.get("name", url[:40])
            pid  = entry.get("id", "")

            if not url or not pid:
                continue

            sched_logger.info(f"  🔄 Actualizando: {name}")
            t0 = time.time()

            try:
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                req.add_header("Accept-Encoding", "gzip, deflate")

                with urllib.request.urlopen(req, timeout=120) as resp:
                    encoding = resp.headers.get("Content-Encoding", "").lower()
                    raw = resp.read()

                if encoding == "gzip":
                    raw = gzip.decompress(raw)

                text = raw.decode("utf-8", errors="ignore")
                channel_count = text.count("#EXTINF")
                elapsed = time.time() - t0

                sched_logger.info(f"  ✅ {name}: {channel_count:,} canales en {elapsed:.1f}s")
                results.append({"name": name, "channels": channel_count, "ok": True})

                # Actualizar timestamp en índice
                entry["timestamp"] = time.time()
                entry["totalChannels"] = channel_count

            except Exception as e:
                sched_logger.error(f"  ❌ {name}: {e}")
                results.append({"name": name, "ok": False, "error": str(e)})
                
                # FIX: Actualizar timestamp aunque falle para que el scheduler
                # no reintente esta lista hasta el próximo ciclo diario
                entry["timestamp"] = time.time()

            time.sleep(3)  # Pausa entre descargas

        # Guardar índice actualizado
        with open(CACHE_INDEX, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    except Exception as e:
        sched_logger.error(f"task_refresh_playlists error: {e}")

    return results


# ─────────────────────────────────────────────────
# TAREA 2: ANALIZAR LOGS DE IA
# ─────────────────────────────────────────────────
def task_analyze_ai_logs():
    """
    Lee ai_decisions.log y extrae:
      - Dominios con más fallos
      - Horas con peor rendimiento
      - Error promedio de predicciones
    """
    sched_logger.info("🔍 TAREA 2: Analizando logs de decisiones IA...")
    analysis = {
        "total_decisions": 0,
        "total_failures": 0,
        "worst_domains": [],
        "worst_hours": [],
        "penalization_events": 0,
    }

    if not os.path.exists(AI_LOG):
        sched_logger.info("  Sin log de IA disponible aún — saltando")
        return analysis

    try:
        domain_fails = {}
        hour_fails   = {}
        total_lines  = 0

        with open(AI_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total_lines += 1
                # Contar decisiones
                if "IA DECISIÓN" in line or "IA RAZONAMIENTO" in line:
                    analysis["total_decisions"] += 1
                # Contar errores
                if "IA ERROR" in line or "fallo crítico" in line:
                    analysis["total_failures"] += 1
                    # Extraer dominio del log
                    m = re.search(r'URL: (https?://[^\s/]+)', line)
                    if m:
                        domain = m.group(1)
                        domain_fails[domain] = domain_fails.get(domain, 0) + 1
                # Contar penalizaciones
                if "PENALIZACIÓN" in line:
                    analysis["penalization_events"] += 1
                # Extraer hora del timestamp
                m_hour = re.match(r'(\d{4}-\d{2}-\d{2} (\d{2}):', line)
                if m_hour and "ERROR" in line:
                    h = int(m_hour.group(2))
                    hour_fails[h] = hour_fails.get(h, 0) + 1

        analysis["worst_domains"] = sorted(domain_fails.items(), key=lambda x: -x[1])[:5]
        analysis["worst_hours"]   = sorted(hour_fails.items(),   key=lambda x: -x[1])[:3]

        sched_logger.info(f"  Total decisiones IA : {analysis['total_decisions']:,}")
        sched_logger.info(f"  Fallos detectados   : {analysis['total_failures']}")
        sched_logger.info(f"  Penalizaciones      : {analysis['penalization_events']}")

        if analysis["worst_domains"]:
            sched_logger.info("  Dominios problemáticos:")
            for d, count in analysis["worst_domains"]:
                sched_logger.info(f"    {d}: {count} errores")

    except Exception as e:
        sched_logger.error(f"task_analyze_ai_logs error: {e}")

    return analysis


# ─────────────────────────────────────────────────
# TAREA 3: EVOLUCIONAR CEREBRO
# ─────────────────────────────────────────────────
def task_evolve_brain():
    """Dispara el ciclo de entrenamiento con las experiencias acumuladas desde ayer."""
    sched_logger.info("🧠 TAREA 3: Evolucionando cerebro neural...")

    try:
        # Importar aquí para no bloquear si tiene problemas
        from iptv_ai_core import ai_optimizer

        count_before = ai_optimizer.experience_count
        ai_optimizer.experience_count = ai_optimizer.get_total_experiences()
        count_total = ai_optimizer.experience_count

        sched_logger.info(f"  Experiencias en BD : {count_total:,}")

        if count_total < 20:
            sched_logger.info("  Menos de 20 experiencias — esperando más datos reales")
            return {"evolved": False, "reason": "insufficient_data", "count": count_total}

        ai_optimizer.evolve_brain()

        cfg = ai_optimizer.config
        sched_logger.info(f"  ✅ Evolución completada")
        sched_logger.info(f"     Error actual   : {cfg.get('avg_error', '?'):.4f}")
        sched_logger.info(f"     Capas          : {cfg.get('layers', [])}")
        sched_logger.info(f"     Entrenamientos : {cfg.get('total_trainings', 0)}")

        return {
            "evolved": True,
            "error": cfg.get("avg_error", 0),
            "layers": cfg.get("layers", []),
            "trainings": cfg.get("total_trainings", 0),
            "experiences": count_total,
        }

    except Exception as e:
        sched_logger.error(f"task_evolve_brain error: {e}")
        return {"evolved": False, "error": str(e)}


# ─────────────────────────────────────────────────
# TAREA 4: REPORTE DIARIO
# ─────────────────────────────────────────────────
def task_generate_report(playlist_results, log_analysis, brain_result):
    """Escribe un reporte diario legible en C:/IPTV-Log/daily_report.txt."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        f"  REPORTE DIARIO IPTV IA — {now}",
        "=" * 60,
        "",
        "📋 PLAYLISTS ACTUALIZADAS:",
    ]

    ok = [r for r in playlist_results if r.get("ok")]
    fail = [r for r in playlist_results if not r.get("ok")]
    lines.append(f"   ✅ Exitosas: {len(ok)}  |  ❌ Fallidas: {len(fail)}")
    for r in ok:
        lines.append(f"   • {r['name'][:40]:40s}  {r['channels']:>6,} canales")
    for r in fail:
        lines.append(f"   • {r['name'][:40]:40s}  ERROR: {r.get('error','?')[:40]}")

    lines += [
        "",
        "🔍 ANÁLISIS DE LOGS IA:",
        f"   Decisiones tomadas  : {log_analysis.get('total_decisions', 0):,}",
        f"   Fallos detectados   : {log_analysis.get('total_failures', 0)}",
        f"   Penalizaciones      : {log_analysis.get('penalization_events', 0)}",
    ]
    if log_analysis.get("worst_domains"):
        lines.append("   Dominios problemáticos:")
        for d, c in log_analysis["worst_domains"]:
            lines.append(f"     {d}: {c} errores")

    lines += ["", "🧠 EVOLUCIÓN DEL CEREBRO:"]
    if brain_result.get("evolved"):
        lines += [
            f"   ✅ Entrenamiento completado",
            f"   Error de red        : {brain_result.get('error', 0):.4f}",
            f"   Arquitectura        : {brain_result.get('layers', [])}",
            f"   Total entrenamientos: {brain_result.get('trainings', 0)}",
            f"   Experiencias en BD  : {brain_result.get('experiences', 0):,}",
        ]
    else:
        lines.append(f"   ⏳ No evolucionado: {brain_result.get('reason', brain_result.get('error', 'Ver logs'))}")

    lines += ["", "=" * 60, ""]

    report_text = "\n".join(lines)

    try:
        with open(DAILY_LOG, "a", encoding="utf-8") as f:
            f.write(report_text)
        sched_logger.info(f"📊 Reporte guardado en {DAILY_LOG}")
    except Exception as e:
        sched_logger.error(f"task_generate_report error: {e}")

    return report_text


# ─────────────────────────────────────────────────
# CICLO COMPLETO DEL SCHEDULER
# ─────────────────────────────────────────────────
def run_daily_cycle():
    """Ejecuta el ciclo completo de mantenimiento diario."""
    sched_logger.info("=" * 50)
    sched_logger.info("🌅 DESPERTANDO — Iniciando ciclo de mantenimiento diario")
    sched_logger.info("=" * 50)

    t0 = time.time()

    playlist_results = task_refresh_playlists()
    log_analysis     = task_analyze_ai_logs()
    brain_result     = task_evolve_brain()
    report           = task_generate_report(playlist_results, log_analysis, brain_result)

    elapsed = time.time() - t0
    sched_logger.info(f"✅ Ciclo completado en {elapsed:.1f}s")
    sched_logger.info("=" * 50)

    # También imprimir en consola para que se vea en el servidor
    print("\n" + report)


# ─────────────────────────────────────────────────
# DAEMON LOOP
# ─────────────────────────────────────────────────
STARTUP_DELAY_S = 120   # Esperar 2 min tras arranque para que el servidor estabilice
MIN_INTERVAL_H  = 20    # No re-ejecutar si ya corrió hace menos de 20h

def scheduler_daemon():
    """
    Ejecuta el ciclo de mantenimiento al arrancar el servidor y luego cada 24h.
    No usa hora fija — se adapta a cualquier horario del usuario.
    """
    sched_logger.info(f"🤖 Scheduler IA iniciado — ciclo al arranque + cada 24h")
    sched_logger.info(f"💤 Esperando {STARTUP_DELAY_S}s para que el servidor estabilice...")
    time.sleep(STARTUP_DELAY_S)

    while True:
        # Verificar si ya ejecutó recientemente (evitar doble ejecución si el servidor reinicia)
        last_run = _get_last_run_timestamp()
        elapsed_h = (time.time() - last_run) / 3600

        if elapsed_h < MIN_INTERVAL_H:
            wait_h = MIN_INTERVAL_H - elapsed_h
            sched_logger.info(f"⏩ Ciclo reciente (hace {elapsed_h:.1f}h) — próximo en {wait_h:.1f}h")
            time.sleep(wait_h * 3600)
            continue

        # Ejecutar ciclo completo
        run_daily_cycle()
        _save_last_run_timestamp()

        # Dormir 24h hasta el próximo ciclo
        sched_logger.info("💤 Próximo ciclo en 24h")
        time.sleep(24 * 3600)


def _get_last_run_timestamp():
    """Lee el timestamp de la última ejecución del scheduler."""
    stamp_file = os.path.join(LOG_DIR, "scheduler_last_run.txt")
    try:
        if os.path.exists(stamp_file):
            return float(open(stamp_file).read().strip())
    except Exception:
        pass
    return 0.0


def _save_last_run_timestamp():
    """Guarda el timestamp de la ejecución actual."""
    stamp_file = os.path.join(LOG_DIR, "scheduler_last_run.txt")
    try:
        with open(stamp_file, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        sched_logger.error(f"No se pudo guardar timestamp: {e}")


# ─────────────────────────────────────────────────
# ARRANQUE
# ─────────────────────────────────────────────────
def start_scheduler_thread():
    """
    Arranca el scheduler como thread daemon.
    Llamar desde iptv-server.py al inicio.
    """
    t = threading.Thread(target=scheduler_daemon, daemon=True, name="AI-Scheduler")
    t.start()
    return t


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Scheduler IPTV")
    parser.add_argument("--now",    action="store_true", help="Ejecutar ciclo ahora y salir")
    parser.add_argument("--daemon", action="store_true", help="Correr como daemon permanente")
    args = parser.parse_args()

    if args.now:
        print("🔥 Ejecutando ciclo inmediatamente...")
        run_daily_cycle()
    elif args.daemon:
        print("🤖 Iniciando como daemon (Ctrl+C para parar)...")
        try:
            scheduler_daemon()
        except KeyboardInterrupt:
            print("\n⏹️  Scheduler detenido")
    else:
        print("Uso: python ai_scheduler.py --now | --daemon")
        print("  --now    Ejecuta el ciclo de mantenimiento ahora mismo")
        print("  --daemon Corre en loop permanente (se despierta a las 04:00)")
