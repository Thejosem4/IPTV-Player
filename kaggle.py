#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  KAGGLE CELDA 2 — ENTRENADOR ESPN CON DATOS REALES
  Entrena el cerebro IA usando experiencias reales
  capturadas durante transmisiones ESPN/deportes HD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Perfil ESPN medido (1h real):
    Velocidad promedio : 14,499 KB/s
    Velocidad mínima   : 221 KB/s  
    Velocidad máxima   : 215,167 KB/s
    Latencia promedio  : 214 ms
    Latencia máxima    : 6,370 ms
    Tasa de fallos     : 38.7%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, math, random, json, time, sqlite3, re, warnings
import numpy as np
import joblib
import requests
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────
CACHE_DIR  = "/kaggle/working/IPTV_Cache"
DB_PATH    = os.path.join(CACHE_DIR, "iptv_permanent_memory.db")
GOLD_DB    = os.path.join(CACHE_DIR, "iptv_gold_memory.db")
MODEL_PATH = os.path.join(CACHE_DIR, "iptv_dynamic_brain.pkl")
SCALER_X   = os.path.join(CACHE_DIR, "iptv_scaler_x_v4.pkl")
SCALER_Y   = os.path.join(CACHE_DIR, "iptv_scaler_y_v4.pkl")
CONFIG_PATH = os.path.join(CACHE_DIR, "brain_config.json")
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b"

# ── Perfil ESPN real medido ───────────────────────
ESPN_PROFILE = {
    "speed_avg_kbps":  14499,
    "speed_min_kbps":  221,
    "speed_max_kbps":  215167,
    "latency_avg_ms":  214,
    "latency_max_ms":  6370,
    "failure_rate":    0.387,
    "bitrate_hd_kbps": 9600,
    "bitrate_4k_kbps": 25000,
}

# ── Arquitectura del cerebro ──────────────────────
HIDDEN_LAYERS = (256, 128, 64, 32)


# ══════════════════════════════════════════════════
# A. REGLAS ESPN — derivadas de datos reales
# ══════════════════════════════════════════════════
class ESPNRuleEngine:
    """
    Reglas calibradas con el perfil real de 1 hora de ESPN.
    No son teóricas — reflejan lo que ocurre realmente en
    el servidor www.1play.cool con streams HD deportivos.
    """
    ESPN_HD_KBPS  = 1200   # 9.6 Mbps ÷ 8 = 1200 KB/s
    ESPN_4K_KBPS  = 3125   # 25 Mbps  ÷ 8 = 3125 KB/s
    LAT_NORMAL    = 300    # ms — umbral latencia normal
    LAT_ALTA      = 800    # ms — latencia degradada
    LAT_CRITICA   = 2000   # ms — latencia crítica (vista en ESPN real)

    @classmethod
    def get_targets(cls, speed, latency, cpu, size_mb, success=1, is_critical=0):
        req_kbps = (size_mb * 1024) / 2.0
        ratio    = req_kbps / max(speed, 1.0)

        # ── CASO 1: Red excelente — ESPN fluido (60% del tiempo real)
        if speed > cls.ESPN_HD_KBPS * 5 and latency < cls.LAT_NORMAL:
            t = {"c": 2, "b": 4096,  "d": 0.05, "p": 6}

        # ── CASO 2: Red buena — ESPN normal (promedio real: 14499 KB/s, 214ms)
        elif speed > cls.ESPN_HD_KBPS * 2 and latency < cls.LAT_ALTA:
            t = {"c": 2, "b": 8192,  "d": 0.10, "p": 10}

        # ── CASO 3: Red degradada — latencia alta pero stream posible
        elif speed > cls.ESPN_HD_KBPS and latency < cls.LAT_CRITICA:
            t = {"c": 2, "b": 16384, "d": 0.25, "p": 18}

        # ── CASO 4: Asfixia — velocidad < bitrate requerido (38.7% real)
        elif ratio > 1.2:
            delay    = float(np.clip(0.4 * ratio, 0.3, 2.5))
            prefetch = int(np.clip(6 * ratio, 15, 35))
            t = {"c": 2, "b": 16384, "d": delay, "p": prefetch}

        # ── CASO 5: Latencia caótica (hasta 6370ms visto en ESPN real)
        elif latency > cls.LAT_CRITICA:
            t = {"c": 2, "b": 16384, "d": 1.5, "p": 30}

        # ── CASO 6: Velocidad muy baja (mínimo real: 221 KB/s)
        elif speed < cls.ESPN_HD_KBPS * 0.5:
            t = {"c": 2, "b": 16384, "d": 2.0, "p": 35}

        # ── DEFAULT conservador para streams live
        else:
            t = {"c": 2, "b": 8192, "d": 0.20, "p": 12}

        # Override por fallo real registrado
        if not success or is_critical:
            t["b"] = 16384
            t["d"] = max(t["d"], 0.8)
            t["c"] = 2
            t["p"] = max(t["p"], 20)

        return t


# ══════════════════════════════════════════════════
# B. CARGA DE DATOS REALES
# ══════════════════════════════════════════════════
def load_real_experiences():
    """
    Carga experiencias reales de la BD local.
    Prioriza:
      1. Streams HD (speed > 8 MB/s) — perfil ESPN
      2. Experiencias críticas (fallos reales)
      3. Resto aleatorio para balance
    """
    print("📂 Cargando experiencias reales de la BD...")
    conn = sqlite3.connect(DB_PATH)

    # Streams HD reales (perfil ESPN principal)
    hd = conn.execute("""
        SELECT cpu, ram, latency, actual_speed, size_mb,
               success, is_critical, url_domain
        FROM experiences
        WHERE actual_speed > 8000 AND actual_speed IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 30000
    """).fetchall()

    # Fallos reales (críticos para aprender a recuperarse)
    fallos = conn.execute("""
        SELECT cpu, ram, latency, actual_speed, size_mb,
               success, is_critical, url_domain
        FROM experiences
        WHERE success = 0 AND actual_speed IS NOT NULL AND actual_speed > 0
        ORDER BY RANDOM()
        LIMIT 20000
    """).fetchall()

    # Experiencias normales para balance
    normales = conn.execute("""
        SELECT cpu, ram, latency, actual_speed, size_mb,
               success, is_critical, url_domain
        FROM experiences
        WHERE actual_speed > 0 AND actual_speed IS NOT NULL
          AND actual_speed BETWEEN 1000 AND 8000
        ORDER BY RANDOM()
        LIMIT 20000
    """).fetchall()

    conn.close()

    all_rows = hd + fallos + normales
    random.shuffle(all_rows)

    print(f"  HD streams (>8 MB/s) : {len(hd):,}")
    print(f"  Fallos reales        : {len(fallos):,}")
    print(f"  Normales             : {len(normales):,}")
    print(f"  Total para entrenar  : {len(all_rows):,}")
    return all_rows


def load_gold_experiences():
    """Carga experiencias gold de Kaggle como complemento si existen."""
    if not os.path.exists(GOLD_DB):
        return []
    try:
        conn = sqlite3.connect(GOLD_DB)
        rows = conn.execute("""
            SELECT cpu, ram, latency, actual_speed, size_mb,
                   1 as success, 0 as is_critical, 'kaggle.gold' as url_domain
            FROM gold_experiences
            ORDER BY error_magnitude DESC
            LIMIT 10000
        """).fetchall()
        conn.close()
        print(f"  Kaggle gold          : {len(rows):,}")
        return rows
    except Exception as e:
        print(f"  ⚠️  Gold DB no disponible: {e}")
        return []


# ══════════════════════════════════════════════════
# C. CONSTRUCCIÓN DEL DATASET
# ══════════════════════════════════════════════════
def build_dataset(real_rows, gold_rows):
    """
    Convierte filas de BD en arrays X, Y listos para entrenar.
    Features idénticas a las que usa iptv_ai_core (7 features v5).
    """
    print("\n🔧 Construyendo dataset de entrenamiento...")
    X, Y = [], []
    skipped = 0

    all_rows = real_rows + gold_rows

    for row in all_rows:
        try:
            cpu     = float(row[0] or 30)
            ram     = float(row[1] or 40)
            lat     = float(row[2] or 214)
            speed   = float(row[3] or 1)
            size_mb = float(row[4] or 1.0)
            success    = int(row[5]) if row[5] is not None else 1
            is_critical = int(row[6]) if row[6] is not None else 0

            if speed <= 0 or size_mb <= 0:
                skipped += 1
                continue

            req_kbps = (size_mb * 1024) / 2.0
            ratio    = req_kbps / max(speed, 1.0)

            # ── 7 Features v5 (mismo orden que iptv_ai_core.get_features)
            feat = [
                math.log1p(size_mb),           # 1. log-size
                cpu,                           # 2. CPU %
                ram,                           # 3. RAM %
                math.sqrt(max(0.0, lat)),      # 4. sqrt latency
                ratio,                         # 5. starvation ratio
                (cpu * ratio) / 100.0,         # 6. CPU × ratio
                0.5,                           # 7. bias constante
            ]

            # ── Target según RuleEngine ESPN
            t = ESPNRuleEngine.get_targets(speed, lat, cpu, size_mb, success, is_critical)

            X.append(feat)
            Y.append([
                float(t["c"]),
                math.log(max(1.0, float(t["b"]))),  # log-buffer (como evolve_brain)
                float(t["d"]),
                float(t["p"]),
            ])

        except Exception:
            skipped += 1
            continue

    print(f"  ✅ {len(X):,} muestras válidas | {skipped} omitidas")

    if len(X) < 50:
        raise ValueError(f"Dataset insuficiente: solo {len(X)} muestras válidas")

    return np.array(X), np.array(Y)


# ══════════════════════════════════════════════════
# D. ENTRENAMIENTO PRINCIPAL
# ══════════════════════════════════════════════════
def train(X, Y):
    """
    Entrena MLPRegressor con arquitectura (256,128,64,32).
    Usa early_stopping para evitar sobreajuste.
    """
    print(f"\n🔥 Entrenando red neuronal {HIDDEN_LAYERS}...")
    print(f"   Muestras  : {len(X):,}")
    print(f"   Features  : {X.shape[1]}")
    print(f"   Outputs   : {Y.shape[1]}")

    sx = StandardScaler()
    sy = StandardScaler()
    X_s = sx.fit_transform(X)
    Y_s = sy.fit_transform(Y)

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=2000,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        warm_start=False,
        verbose=False,
        random_state=42,
    )

    t0 = time.time()
    model.fit(X_s, Y_s)
    elapsed = time.time() - t0

    # Reemplazo seguro para capturar el loss
    loss = getattr(model, 'best_loss_', None)
    if loss is None:
        loss = getattr(model, 'loss_', 0.0) # Fallback al loss actual o 0.0
    
    print(f"\n  ✅ Entrenamiento completado en {elapsed:.1f}s")
    print(f"     Loss final     : {loss:.6f}") # Ahora loss nunca será None
    print(f"     Iteraciones    : {model.n_iter_}")

    return model, sx, sy


# ══════════════════════════════════════════════════
# E. REFINAMIENTO CON OLLAMA (casos difíciles)
# ══════════════════════════════════════════════════
def ollama_disponible():
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True
    except Exception:
        return False


def refine_with_ollama(model, sx, sy, X, Y, real_rows, max_corrections=300):
    """
    Ollama solo interviene en los casos donde el modelo comete
    errores en escenarios de asfixia real (ratio > 1.2).
    Máximo 300 correcciones para no saturar la GPU con LLM.
    """
    if not ollama_disponible():
        print("\n⚠️  Ollama no disponible — omitiendo refinamiento LLM")
        return model

    print(f"\n🤖 Refinamiento Ollama sobre casos de asfixia real...")
    corrections_X, corrections_Y = [], []
    revisados = 0
    corregidos = 0

    for row in real_rows:
        if revisados >= 3000 or corregidos >= max_corrections:
            break
        try:
            cpu     = float(row[0] or 30)
            ram     = float(row[1] or 40)
            lat     = float(row[2] or 214)
            speed   = float(row[3] or 1)
            size_mb = float(row[4] or 1.0)

            req_kbps = (size_mb * 1024) / 2.0
            ratio    = req_kbps / max(speed, 1.0)

            # Solo casos de asfixia real — los más difíciles
            if ratio <= 1.2 and lat < 1000:
                revisados += 1
                continue

            feat = [
                math.log1p(size_mb), cpu, ram,
                math.sqrt(max(0.0, lat)), ratio,
                (cpu * ratio) / 100.0, 0.5
            ]

            # Predicción actual del modelo
            pred_s = model.predict(sx.transform([feat]))[0]
            pred   = sy.inverse_transform([pred_s])[0]
            pred_conn = int(np.clip(round(pred[0]), 1, 48))
            pred_buf  = int(np.clip(math.exp(pred[1]), 128, 16384))

            prompt = (
                f"IPTV ESPN Live Stream Expert. Real measured data:\n"
                f"Network speed: {speed:.0f} KB/s | Required: {req_kbps:.0f} KB/s\n"
                f"Latency: {lat:.0f}ms | CPU: {cpu:.0f}% | RAM: {ram:.0f}%\n"
                f"Context: ESPN HD stream (9600kbps bitrate) on real server.\n"
                f"Current model prediction: conn={pred_conn}, buffer={pred_buf}KB\n"
                f"Goal: ZERO buffering on ESPN HD. Correct if needed.\n"
                f"Respond ONLY with valid JSON, nothing else:\n"
                f"{{\"target_num_conn\": N, \"target_buffer\": N, "
                f"\"target_delay\": N.NN, \"target_prefetch\": N}}"
            )

            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.15, "num_predict": 80}
                },
                timeout=40
            )
            match = re.search(r'\{[^}]+\}', res.json()["response"], re.DOTALL)
            if not match:
                revisados += 1
                continue

            j = json.loads(match.group(0))
            corrections_X.append(feat)
            corrections_Y.append([
                float(j["target_num_conn"]),
                math.log(max(1.0, float(j["target_buffer"]))),
                float(j["target_delay"]),
                float(j["target_prefetch"]),
            ])
            corregidos += 1

        except Exception:
            pass
        revisados += 1

    if not corrections_X:
        print("   ⚠️  Sin correcciones Ollama obtenidas")
        return model

    print(f"   ✅ {corregidos} correcciones Ollama obtenidas de {revisados} casos revisados")

    # Reentrenar con correcciones incorporadas
    # --- MEZCLA DE DATOS PARA EVITAR OLVIDO CATASTRÓFICO ---
    # Mezclamos 1,000 muestras del pasado con las correcciones del LLM
    indices = np.random.choice(len(X), size=min(len(X), 1000), replace=False)
    X_mixed = np.vstack([X[indices], np.array(corrections_X)])
    Y_mixed = np.vstack([Y[indices], np.array(corrections_Y)])
    
    X_ns = sx.transform(X_mixed)
    Y_ns = sy.transform(Y_mixed)

    # --- PARCHE CRÍTICO PARA WARM_START ---
    model.warm_start = True
    model.max_iter   = 500
    model.learning_rate_init = 0.0001 # Tasa mucho más baja para no romper lo aprendido
    model.early_stopping = False
    
    # Si best_loss_ es None o se perdió, lo inicializamos con infinito
    # para que la primera iteración del refinamiento siempre sea "la mejor"
    if not hasattr(model, 'best_loss_') or model.best_loss_ is None:
        model.best_loss_ = np.inf
    
    # También reiniciamos el contador de "no mejora" para que no corte antes de tiempo
    if hasattr(model, '_no_improvement_count'):
        model._no_improvement_count = 0
    # ---------------------------------------

    model.fit(X_ns, Y_ns)

    joblib.dump(model, MODEL_PATH)
    print(f"   🔥 Modelo refinado. Loss post-Ollama: {model.loss_:.6f}")
    return model


# ══════════════════════════════════════════════════
# F. GUARDAR MODELOS Y CONFIG
# ══════════════════════════════════════════════════
def save_all(model, sx, sy, n_samples):
    # Captura segura de loss para el reporte
    loss = getattr(model, 'best_loss_', None) or getattr(model, 'loss_', 0.0)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(sx, SCALER_X)
    joblib.dump(sy, SCALER_Y)

    config = {
        "complexity_level": 3,
        "layers": list(HIDDEN_LAYERS),
        "total_trainings": 1,
        "avg_error": float(loss),
        "trained_on": "real_espn_data",
        "samples_used": n_samples,
        "espn_profile": ESPN_PROFILE,
        "timestamp": time.time(),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Archivos guardados en {CACHE_DIR}:")
    for fname in ["iptv_dynamic_brain.pkl", "iptv_scaler_x_v4.pkl",
                  "iptv_scaler_y_v4.pkl", "brain_config.json"]:
        fpath = os.path.join(CACHE_DIR, fname)
        if os.path.exists(fpath):
            kb = os.path.getsize(fpath) / 1024
            print(f"  ✔️  {fname:45s} {kb:8.1f} KB")


# ══════════════════════════════════════════════════
# G. VERIFICACIÓN FINAL
# ══════════════════════════════════════════════════
def verify(model, sx, sy):
    print("\n🧪 Verificación con escenarios ESPN reales:")
    tests = [
        (14499, 214,  50, 1.2, "ESPN HD  — promedio real (14499 KB/s, 214ms)"),
        (221,   2000, 80, 1.2, "ESPN HD  — peor caso real (221 KB/s, 2000ms)"),
        (3125,  500,  60, 4.0, "ESPN 4K  — red degradada (3125 KB/s, 500ms)"),
        (14499, 6370, 90, 1.2, "ESPN HD  — latencia máxima real (6370ms)"),
        (50000, 50,   30, 1.2, "ESPN HD  — red excelente (50 MB/s, 50ms)"),
        (500,   3000, 85, 1.2, "ESPN HD  — crisis extrema (500 KB/s, 3000ms)"),
    ]

    print(f"\n  {'Escenario':<52} {'conn':>4} {'buffer':>7} {'delay':>7} {'prefetch':>8}")
    print("  " + "─" * 80)

    for speed, lat, cpu, size_mb, desc in tests:
        ratio = (size_mb * 1024 / 2.0) / max(speed, 1.0)
        feat  = [
            math.log1p(size_mb), float(cpu), 50.0,
            math.sqrt(float(lat)), ratio, (float(cpu) * ratio) / 100.0, 0.5
        ]
        try:
            pred_s = model.predict(sx.transform([feat]))[0]
            pred   = sy.inverse_transform([pred_s])[0]
            conn   = int(np.clip(round(pred[0]), 1, 4))
            buf    = int(np.clip(math.exp(pred[1]), 128, 16384))
            delay  = float(np.clip(pred[2], 0.01, 3.0))
            pref   = int(np.clip(round(pred[3]), 1, 40))
            print(f"  {desc:<52} {conn:>4} {buf:>6}KB {delay:>6.3f}s {pref:>8}")
        except Exception as e:
            print(f"  {desc:<52} ❌ Error: {e}")


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
def run():
    print("=" * 65)
    print("  🏆 ESPN REAL DATA TRAINER — Kaggle GPU Edition v1.0")
    print("=" * 65)

    # Verificar BD disponible
    if not os.path.exists(DB_PATH):
        print(f"❌ No se encontró {DB_PATH}")
        print("   Asegúrate de haber ejecutado la Celda 1 primero.")
        return

    # 1. Cargar datos
    real_rows = load_real_experiences()
    gold_rows = load_gold_experiences()

    if len(real_rows) < 100:
        print("❌ Datos reales insuficientes para entrenar.")
        return

    # 2. Construir dataset
    X, Y = build_dataset(real_rows, gold_rows)

    # 3. Entrenar
    model, sx, sy = train(X, Y)

    # 4. Refinar con Ollama (casos difíciles)
    model = refine_with_ollama(model, sx, sy, X, Y, real_rows, max_corrections=300)

    # 5. Guardar
    save_all(model, sx, sy, len(X))

    # 6. Verificar
    verify(model, sx, sy)

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ CELDA 2 COMPLETADA — Ejecuta la Celda 3 para
     exportar el cerebro entrenado
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    run()