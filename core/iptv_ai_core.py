import os
import time
import json
import sys
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

import numpy as np
import psutil
import sqlite3
import difflib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import threading
import queue
import logging
import collections
import sys

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_SKL2ONNX = True
except ImportError:
    HAS_SKL2ONNX = False

try:
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Re-configurar logger para consola + archivo
ai_logger = logging.getLogger("IPTV_AI")
ai_logger.setLevel(logging.INFO)

# Evitar handlers duplicados si se recarga el módulo
if not ai_logger.handlers:
    # 1. Archivo
    file_handler = logging.FileHandler(os.path.join(os.path.join(os.path.dirname(__file__), '..', 'logs'), 'ai_decisions.log'), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    ai_logger.addHandler(file_handler)
    
    # 2. Consola en vivo
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    ai_logger.addHandler(console_handler)

# Sobrescribir el logging normal para que logging.info invoque al ai_logger
logging.info = ai_logger.info
logging.error = ai_logger.error
logging.warning = ai_logger.warning

# Configuración de Rutas Dinámicas (Portabilidad v5.0)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DB_PATH = os.path.join(CACHE_DIR, "iptv_permanent_memory.db")
GOLD_DB_PATH = os.path.join(CACHE_DIR, "iptv_gold_memory.db")
MODEL_PATH = os.path.join(CACHE_DIR, "iptv_dynamic_brain.pkl")
LIGHTGBM_MODEL_PATH = os.path.join(CACHE_DIR, "iptv_lgbm_brain.pkl")
ONNX_MODEL_PATH = os.path.join(CACHE_DIR, "iptv_dynamic_brain.onnx")
CONFIG_PATH = os.path.join(CACHE_DIR, "brain_config.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class IPTVEvolutionaryBrain:
    # Número de salidas del vector Y en v6.1:
    # [conn, buf_log, delay, pref, threads, preset, reconnect, stall_prob]
    EXPECTED_N_OUTPUTS = len([2, 0, 1.5, 40, 4, 0, 1, 0.9])
    # Número de features de entrada en v6.1 (se eliminó el sesgo 0.5 constante,
    # reemplazado por la hora del día normalizada: hour/23.0)
    EXPECTED_N_FEATURES = 7  # log_size, cpu, ram, sqrt_lat, ratio, cpu_interaction, hour_norm

    def __init__(self):
        self.lock = threading.Lock()
        self.db_queue = queue.Queue()
        self.init_db()
        self.load_config() # This sets self.config
        self.scaler_x = self.load_scaler_x()
        self.scaler_y = self.load_scaler_y()
        self.lgb_model = self.load_lgbm_model()
        self.model = self.load_model()
        self.onnx_session = self.load_onnx_model()
        self.error_history = []
        self.experience_count = self.get_total_experiences()
        self._last_retrain_count = 0
        
        # Hilo daemon para escrituras asíncronas en SQLite (0ms latencia para el player)
        self._db_writer_thread = threading.Thread(target=self._db_writer_worker, daemon=True)
        self._db_writer_thread.start()

        # Auto-importar memoria Kaggle si la BD local está vacía o escasa
        if self.experience_count < 1000:
            self.experience_count = self.import_gold_memory()
        self.start_auto_retraining()

    def _db_writer_worker(self):
        """Worker thread background que procesa inserciones en la BD sin bloquear hilos de streaming"""
        while True:
            try:
                item = self.db_queue.get()
                if item is None:
                    break
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO experiences 
                    (timestamp, url_domain, full_url, channel_name, size_mb, hour, day_of_week, cpu, ram, latency, 
                     target_num_conn, target_buffer, target_delay, target_prefetch, actual_speed, success, is_critical)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', item)
                conn.commit()
                conn.close()
                self.db_queue.task_done()
            except Exception as e:
                ai_logger.error(f"Error en _db_writer_worker: {e}")

    def load_onnx_model(self):
        """Carga la sesión de inferencia de ONNX Runtime si está disponible"""
        if HAS_ONNX and os.path.exists(ONNX_MODEL_PATH):
            try:
                sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
                ai_logger.info(" Optimizer: Modelo ONNX cargado exitosamente para inferencia ultrarrápida (<50μs)")
                return sess
            except Exception as e:
                ai_logger.warning(f" Optimizer: No se pudo cargar modelo ONNX ({e}). Usando fallback Scikit-Learn.")
        return None

    def export_onnx_model(self):
        """Exporta el modelo Scikit-Learn a ONNX Runtime si skl2onnx está instalado"""
        if not HAS_SKL2ONNX or self.model is None:
            return
        try:
            initial_type = [('float_input', FloatTensorType([None, 7]))]
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
            with open(ONNX_MODEL_PATH, "wb") as f:
                f.write(onnx_model.SerializeToString())
            ai_logger.info(f" Optimizer: Modelo ONNX exportado guardado en {ONNX_MODEL_PATH}")
            self.onnx_session = self.load_onnx_model()
        except Exception as e:
            ai_logger.warning(f" Optimizer: Error exportando a ONNX: {e}")

    def init_db(self):
        """Crea la base de datos de memoria permanente si no existe"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                url_domain TEXT,
                full_url TEXT,
                channel_name TEXT,
                size_mb REAL,
                hour INTEGER,
                day_of_week INTEGER,
                cpu REAL,
                ram REAL,
                latency REAL,
                target_num_conn INTEGER,
                target_buffer INTEGER,
                target_delay REAL,
                target_prefetch INTEGER,
                actual_speed REAL,
                success INTEGER,
                codec TEXT,
                playback_success INTEGER DEFAULT 1,
                is_critical INTEGER
            )
        ''')
        # Migración: añadir columnas si no existen (v6.0)
        cursor.execute("PRAGMA table_info(experiences)")
        cols = [c[1] for c in cursor.fetchall()]
        if "codec" not in cols: cursor.execute("ALTER TABLE experiences ADD COLUMN codec TEXT")
        if "playback_success" not in cols: cursor.execute("ALTER TABLE experiences ADD COLUMN playback_success INTEGER DEFAULT 1")
        conn.commit()
        # Migración simple: verificar si existen las nuevas columnas
        cursor.execute("PRAGMA table_info(experiences)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'channel_name' not in columns:
            cursor.execute("ALTER TABLE experiences ADD COLUMN channel_name TEXT")
        if 'full_url' not in columns:
            cursor.execute("ALTER TABLE experiences ADD COLUMN full_url TEXT")
        conn.commit()
        conn.close()

    def import_gold_memory(self, kaggle_db_path: str = None):
        """
        Importa la memoria de Kaggle (iptv_gold_memory.db / tabla gold_experiences)
        a la BD local (iptv_permanent_memory.db / tabla experiences).
        Si kaggle_db_path es None, usa GOLD_DB_PATH definido en las constantes.
        Solo importa si la BD local tiene menos de 1000 experiencias para no re-importar.
        """
        import random as _rng
        src_path = kaggle_db_path or GOLD_DB_PATH
        if not os.path.exists(src_path):
            print(f" Optimizer: No se encontró BD de Kaggle en {src_path}")
            return 0

        local_count = self.get_total_experiences()
        if local_count >= 1000:
            print(f"ℹ Optimizer: BD local ya tiene {local_count:,} experiencias. Importación omitida.")
            return local_count

        print(f" Optimizer: Importando memoria Kaggle desde {src_path}...")
        src  = sqlite3.connect(src_path)
        dst  = sqlite3.connect(DB_PATH)
        now  = time.time()

        rows = src.execute("""
            SELECT cpu, ram, latency, actual_speed, size_mb,
                   target_num_conn, target_buffer, target_delay, target_prefetch
            FROM gold_experiences
        """).fetchall()
        src.close()

        imported = 0
        for r in rows:
            cpu, ram, lat, spd, sz, conn_t, buf, delay, pref = r
            hour    = _rng.randint(0, 23)
            dow     = _rng.randint(0, 6)
            success = 1 if spd > (sz * 1024 / 2.0) else 0
            is_crit = 0 if success else 1
            ts      = now - _rng.randint(0, 86400 * 30)
            dst.execute("""
                INSERT INTO experiences
                (timestamp, url_domain, full_url, channel_name, size_mb, hour, day_of_week,
                 cpu, ram, latency, target_num_conn, target_buffer, target_delay,
                 target_prefetch, actual_speed, success, is_critical)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (ts, "kaggle.gold", "http://kaggle.gold/vod/0.ts",
                  "Kaggle-Gold", sz, hour, dow, cpu, ram, lat,
                  conn_t, buf, delay, pref, spd, success, is_crit))
            imported += 1

        dst.commit()
        dst.close()
        total = self.get_total_experiences()
        print(f" Optimizer: {imported:,} experiencias Kaggle importadas. Total BD local: {total:,}")
        return total

    def load_config(self):
        """Carga la configuración de la arquitectura de la red"""
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "complexity_level": 3,
                "layers": [256, 128, 64, 32],
                "total_trainings": 0,
                "avg_error": 1.0
            }
            self.save_config()

    def save_config(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f)

    def start_auto_retraining(self, interval_minutes=30):
        """Daemon de reentrenamiento que dispara evolve_brain() cuando hay suficientes datos nuevos."""
        def _daemon():
            while True:
                time.sleep(interval_minutes * 60)
                current_count = self.get_total_experiences()
                nuevos_rows = current_count - getattr(self, "_last_retrain_count", 0)
                if nuevos_rows > 100:
                    try:
                        self.evolve_brain()  # Punto de entrada único de entrenamiento
                        self._last_retrain_count = self.get_total_experiences()
                    except Exception as e:
                        print(f"Error en auto-retraining: {e}")
                        
        t = threading.Thread(target=_daemon, daemon=True)
        t.start()

    # ── NOTA v6.1: _retrain() eliminado ──────────────────────────────────────
    # La función _retrain() fue unificada en evolve_brain() para evitar doble
    # lógica contradictoria y scalers que se reseteaban mutuamente.
    # evolve_brain() es ahora el único punto de entrada de entrenamiento.
    # ─────────────────────────────────────────────────────────────────────────

    def load_scaler_x(self):
        path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'cache'), "iptv_scaler_x_v4.pkl")
        if os.path.exists(path):
            try: return joblib.load(path)
            except: pass
        return StandardScaler()

    def load_scaler_y(self):
        path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'cache'), "iptv_scaler_y_v4.pkl")
        if os.path.exists(path):
            try: return joblib.load(path)
            except: pass
        return StandardScaler()


    def load_lgbm_model(self):
        """Carga el modelo LightGBM (Motor Principal v7.0) si está disponible"""
        if HAS_LIGHTGBM and os.path.exists(LIGHTGBM_MODEL_PATH):
            try:
                lgb_m = joblib.load(LIGHTGBM_MODEL_PATH)
                ai_logger.info(" Optimizer: Modelo LightGBM (Motor Principal v7.0) cargado exitosamente.")
                return lgb_m
            except Exception as e:
                ai_logger.warning(f" Optimizer: Error cargando modelo LightGBM ({e}).")
        return None

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                # Auto-migración: descartar modelo si sus salidas no coinciden con la versión actual
                if hasattr(model, 'n_outputs_') and model.n_outputs_ != self.EXPECTED_N_OUTPUTS:
                    print(f"ℹ Optimizer: Modelo PKL incompatible ({model.n_outputs_} salidas vs "
                          f"{self.EXPECTED_N_OUTPUTS} esperadas). Descartando para reentrenar...")
                    return None
                return model
            except:
                return None
        return None

    def get_features(self, url, size_mb, latency=0, speed_kbps=0.0, required_bitrate=0.0):
        """
        Extrae 7 features para la v6.1 del Cerebro.

        Cambio v6.1: Feature #7 cambiada de sesgo constante (0.5) a hora del día
        normalizada (hour/23.0). El sesgo constante era convertido a 0 por
        StandardScaler (varianza=0), por lo que no aportaba información alguna.
        La hora del día captura patrones de tráfico diurno/nocturno reales.
        """
        import math
        
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        hour_norm = time.localtime().tm_hour / 23.0  # [0.0, 1.0]
        
        # ratio = required_bitrate / speed_kbps — mide presión de ancho de banda
        if speed_kbps <= 5.0:
            ratio = 1.0
        else:
            ratio = required_bitrate / max(speed_kbps, 1.0)
            
        return np.array([[
            math.log1p(size_mb),               # 1. Tamaño logarítmico
            cpu,                               # 2. Carga CPU
            ram,                               # 3. Uso RAM
            math.sqrt(max(0, latency)),        # 4. Latencia suavizada
            ratio,                             # 5. Ratio de asfixia de ancho de banda
            (cpu * ratio) / 100.0,             # 6. Interacción CPU × ratio
            hour_norm                          # 7. Hora del día [0.0-1.0] (era sesgo 0.5)
        ]])

    def calculate_bola_quality(self, available_qualities, buffer_ms, V=0.8):
        """
        Algoritmo BOLA (Buffer Occupancy based Lyapunov Algorithm) para ABR
        :param available_qualities: lista de dicts [{'resolution': '1080p', 'bitrate': 5000}, ...]
        :param buffer_ms: milisegundos de buffer actual reproduciendo
        :param V: Parámetro de agresividad de utilidad
        :return: dict de la calidad ganadora
        """
        if not available_qualities: return None
        import math
        
        # Ordenar por bitrate ascendente
        qualities = sorted(available_qualities, key=lambda x: x.get('bitrate', 1000))
        best_quality = qualities[0]
        max_objective = -float('inf')
        
        # Psi(B): Penalty por cercanía al vacío de buffer (Riesgo de rebuffering)
        # Usamos una hipérbola calibrada: a 2s castiga fuerte, a >8s permite máxima calidad
        psi_b = 1500.0 / (buffer_ms + 1)
        
        for q in qualities:
            bitrate = float(q.get('bitrate', 1000) or 1000)
            
            # Utilidad U(q): Log natural del ancho de banda (Teoría Lyapunov)
            u_q = math.log1p(bitrate)
            
            # Objetivo matemáticamente correcto de Lyapunov BOLA:
            # Maximizar V * Utilidad - Penalización de Buffer
            objective = (V * u_q) - (psi_b * (bitrate / 1000.0))
            
            if objective > max_objective:
                max_objective = objective
                best_quality = q
                
        res = best_quality.get('resolution', 'N/A')
        br = best_quality.get('bitrate', 0)
        print(f"    Optimizer ABR (BOLA) | Buffer: {buffer_ms}ms -> Selección Lyapunov: {res} a {br}kbps")
        return best_quality


    def predict_optimal_config(self, url, size_mb, latency=0, speed_kbps=0.0, required_bitrate=0.0):
        """Predice la mejor configuración basándose en el estado del sistema, red y MEMOROptimizer de fallos"""
        with self.lock:
            sys_cpu = psutil.cpu_percent()
            sys_ram = psutil.virtual_memory().percent
            
            #  SEGUNDO CEREBRO: Memoria Histórica del Servidor IPTV
            domain = url.split('/')[2] if '/' in url else "unknown"
            recent_failures = 0
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM experiences 
                    WHERE url_domain = ? AND success = 0 AND timestamp > ?
                ''', (domain, time.time() - 86400)) # Últimas 24 horas
                recent_failures = cursor.fetchone()[0]
                conn.close()
            except: pass

            pred = None
            engine_used = None

            # ── 1. Intento Motor Principal LightGBM (Gradient Boosted Trees v7.0) ──
            if self.lgb_model is not None:
                try:
                    X_features = self.get_features(url, size_mb, latency, speed_kbps, required_bitrate)
                    pred = self.lgb_model.predict(X_features)[0]
                    engine_used = "LightGBM v7.0 (Motor Principal - Árboles Gradient Boosted)"
                except Exception as lgb_err:
                    ai_logger.warning(f"LightGBM predict falló ({lgb_err}), ejecutando neural fallback")

            # ── 2. Intento Neural Fallback (ONNX Session / MLPRegressor v6.1) ───────
            if pred is None:
                if self.model is None:
                    config = self.expert_fallback(size_mb, recent_failures)
                    reason = f"Lógica Experto | CPU: {sys_cpu}%"
                    if recent_failures > 0: reason += f" | Historial Fallos Servidor: {recent_failures}"
                    logging.info(f" Optimizer RAZONAMIENTO | {reason} | URL: {url[:30]}... | OUT: {config}")
                    return config
                try:
                    X_features = self.get_features(url, size_mb, latency, speed_kbps, required_bitrate)
                    X_scaled = self.scaler_x.transform(X_features)
                    
                    if self.onnx_session is not None:
                        try:
                            input_name = self.onnx_session.get_inputs()[0].name
                            pred_scaled = self.onnx_session.run(None, {input_name: X_scaled.astype(np.float32)})[0][0]
                            engine_used = "ONNX Runtime v6.1 (Neural Fallback)"
                        except Exception as onnx_err:
                            pred_scaled = self.model.predict(X_scaled)[0]
                            engine_used = "MLPRegressor v6.1 (Neural Fallback)"
                    else:
                        pred_scaled = self.model.predict(X_scaled)[0]
                        engine_used = "MLPRegressor v6.1 (Neural Fallback)"

                    pred = self.scaler_y.inverse_transform([pred_scaled])[0]
                except Exception as e:
                    logging.error(f" Optimizer ERROR en predicción: {e}")
                    return self.expert_fallback(size_mb, recent_failures)
            else:
                X_features = self.get_features(url, size_mb, latency, speed_kbps, required_bitrate)

            # Desempaquetar y limpiar predicción (Mapeo dinámico v5.0/v7.0)
            if len(pred) >= 8:
                c_conn, buf_log, delay, pref, th, pr_idx, rec, st_prob = pred
            else:
                c_conn, buf_log, delay, pref = pred
                th, pr_idx, rec, st_prob = 2, 0, 0, 0.1 # Fallbacks seguros v4.0

            import math
            buf_real = math.exp(buf_log) if buf_log < 15 else 16384 # Protección para no desbordar
            
            # Limitamos num_conn drásticamente (max 4) para evitar bloqueos del proveedor IPTV
            num_conn = int(np.clip(round(c_conn), 1, 4))
            buffer_size = int(np.clip(round(buf_real), 2048, 16384))
            retry_delay = float(np.clip(delay, 0.01, 3.0))
            prefetch_count = int(np.clip(round(pref), 1, 40))

            # Parámetros avanzados de FFmpeg/Watchdog (v7.0 Engine)
            ai_threads = int(np.clip(round(th), 1, 4))
            ai_preset_list = ["ultrafast", "superfast", "veryfast", "faster"]
            ai_preset = ai_preset_list[int(np.clip(round(pr_idx), 0, 3))]
            ai_reconnect = 1 if rec > 0.5 else 0
            ai_stall_prob = float(np.clip(st_prob, 0.0, 1.0))
            
            # MEMOROptimizer PREVENTIVA: Castigo a servidores rebeldes
            if recent_failures > 2:
                buffer_size = max(buffer_size, 4096)
                retry_delay = max(retry_delay, 1.5)
                num_conn = max(2, min(num_conn, 4)) # Estrangular conexiones paralelas para no romper el socket remoto
            
            # --- SAFETY OVERRIDE LAYER (ANTI-HALLUCINATION) ---
            # Si el ratio de asfixia (Required / Speed) es alto, forzamos buffer máximo 
            # para evitar cortes, protegiendo al sistema contra modelos mal entrenados.
            x_ratio = X_features[0, 4]
            if x_ratio > 1.2:
                if buffer_size < 16384:
                    buffer_size = 16384
            elif x_ratio < 0.5:
                if buffer_size > 4096:
                    buffer_size = 2048 # Ahorro de RAM preventivo

            config = {
                "num_conn": num_conn,
                "buffer_kb": buffer_size,
                "retry_delay": round(retry_delay, 3),
                "prefetch_count": prefetch_count,
                "threads": ai_threads,
                "preset": ai_preset,
                "reconnect": ai_reconnect,
                "stall_prob": ai_stall_prob
            }
            
            reason = f"Predicción {engine_used or 'IA'} basada en Latencia {latency:.0f}ms y Carga {sys_cpu}%"
            if getattr(self, 'consecutive_errors', 0) > 2:
                config["num_conn"] = max(2, config["num_conn"] // 2)
                reason += " | PENALIZACIÓN errores seguidos"
            if recent_failures > 0:
                reason += f" | MEMOROptimizer SEVERA: {recent_failures} fallos origen 24h"
            
            logging.info(f" Optimizer DECISIÓN | {reason} | URL: {url[:30]}... | DECISIÓN: {config}")
            return config


    def expert_fallback(self, size_mb, failures=0):
        # Fallback mucho más agresivo si sabemos que el server falla mucho
        if failures > 3:
            return {"num_conn": 2, "buffer_kb": 8192, "retry_delay": 5.0, "prefetch_count": 2}
        elif size_mb > 5:
            return {"num_conn": 10, "buffer_kb": 4096, "retry_delay": 2.0, "prefetch_count": 6}
        return {"num_conn": 8, "buffer_kb": 2500, "retry_delay": 1.0, "prefetch_count": 4}

    def log_experience(self, url, size_mb, speed_kbps, success, latency=0, channel_name=None):
        """Guarda la experiencia en la cola asíncrona de SQLite (0ms latencia para el hilo de streaming)"""
        logging.info(f" Optimizer MEMOROptimizer | Registrando evento: {channel_name or url[:30]}... | Velocidad: {speed_kbps:.0f} KB/s | Éxito: {success}")
        t = time.localtime()
        domain = url.split('/')[2] if '/' in url else "unknown"
        
        # Determinar si es una experiencia crítica (ej: fallo o velocidad muy baja)
        is_critical = 1 if (not success or speed_kbps < 1000) else 0
        
        # Targets ideales para el aprendizaje
        target_num_conn = 32 if is_critical else 8
        target_buffer = 4096 if is_critical else 2048
        target_delay = 0.4 if is_critical else 0.05
        target_prefetch = 8 if is_critical else 3

        item = (time.time(), domain, url, channel_name, size_mb, t.tm_hour, t.tm_wday, psutil.cpu_percent(), 
                psutil.virtual_memory().percent, latency, target_num_conn, target_buffer, 
                target_delay, target_prefetch, speed_kbps, 1 if success else 0, is_critical)
        
        self.db_queue.put(item)
        
        self.experience_count += 1
        if self.experience_count % 100 == 0:
            threading.Thread(target=self.evolve_brain, daemon=True).start()

    def find_best_mirror(self, target_name):
        """Busca un mirror similar para un canal que falló"""
        if not target_name:
            return None
            
        print(f" Optimizer: Buscando mirror para '{target_name}'...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Obtener todos los nombres de canales conocidos
        cursor.execute("SELECT DISTINCT channel_name FROM experiences WHERE channel_name IS NOT NULL")
        all_names = [r[0] for r in cursor.fetchall()]
        
        # Buscar coincidencias cercanas
        matches = difflib.get_close_matches(target_name, all_names, n=3, cutoff=0.6)
        
        if not matches:
            conn.close()
            return None
            
        # Para los nombres encontrados, buscar la mejor URL (mayor velocidad y éxito)
        placeholders = ','.join(['?'] * len(matches))
        query = f"""
            SELECT full_url, actual_speed FROM experiences 
            WHERE channel_name IN ({placeholders}) AND success = 1 AND full_url IS NOT NULL
            ORDER BY actual_speed DESC LIMIT 1
        """
        cursor.execute(query, matches)
        res = cursor.fetchone()
        conn.close()
        
        if res:
            print(f" Optimizer: Mirror encontrado: {res[0]} (speed: {res[1]:.0f} KB/s)")
            return res[0]
        return None

    def log_failure(self, url):
        """Marca una URL como no confiable"""
        print(f" Optimizer: Registrando fallo crítico en URL: {url[:60]}...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE experiences SET is_critical = 1, success = 0 
            WHERE full_url = ? OR url_domain = ?
        ''', (url, url.split('/')[2] if '/' in url else url))
        conn.commit()
        conn.close()

    def get_total_experiences(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            res = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            conn.close()
            return res
        except: return 0

    @staticmethod
    def _compute_continuous_labels(size_mb, cpu, ram, latency, speed, hour, success):
        """
        v6.1: Genera targets CONTINUOS usando matemática derivada (inspirada en RuleEngine).

        PROBLEMA ANTERIOR: Los labels eran binarios (2 estados fijos: agresivo/eficiente).
        Una red neuronal entrenada con labels binarios converge a un punto medio estático
        y pierde toda granularidad de decisión.

        SOLUCIÓN: Calcular targets con funciones continuas que varían suavemente con
        cada combinación de condiciones de red, produciendo miles de valores únicos.
        """
        import math

        # ── 1. CONEXIONES — escalado cuadrático con calidad de red ──────────────
        # Representa el ancho de banda como fasor: magnitud = calidad total
        bw_norm = min(speed / 50000.0, 1.0)           # [0,1] normalizado a 50Mbps
        lat_norm = 1.0 / (1.0 + latency / 100.0)      # [0,1] — más lat = peor
        quality2 = bw_norm ** 2 + lat_norm ** 2        # suma cuadrática de componentes
        stress = math.sqrt(max(0.001, cpu / 100) * max(0.001, ram / 100))
        bw_log = math.log2(speed / 1000 + 1)
        conn_raw = 16.0 * quality2 * bw_log * (1 - stress / 2)
        if speed < 1500:
            conn_raw = max(2, min(3, round(speed / 750)))
        if cpu > 85:
            conn_raw = max(2, conn_raw // 2)
        target_conn = float(np.clip(round(conn_raw), 2, 32))
        if not success:
            target_conn = max(2.0, target_conn * 0.6)  # Retroceder en fallos

        # ── 2. BUFFER — Shannon entropy en función de error de paquetes ──────────
        # La pérdida de paquetes estimada es proporcional a la latencia y al ratio de asfixia
        ratio = (2500.0 / max(speed, 1.0)) if speed > 5 else 1.0
        p_error = min(0.99, max(0.01, (1 - min(speed / 50000.0, 0.99)) * min(latency / 500, 1.0)))
        entropy = -(p_error * math.log2(p_error) + (1 - p_error) * math.log2(1 - p_error))
        snr = max(0.1, speed / (1 + latency * 10))
        buf = max(512, min(16384, int(256 + 4096 * (entropy / max(0.1, math.log2(math.log2(1 + snr) + 1))))))
        if speed < 1500 or latency > 500:
            buf = min(16384, int(buf * 1.5))
        target_buf_log = math.log(float(buf))

        # ── 3. DELAY — backoff exponencial TCP-inspired ───────────────────────────
        severity = (2.0 if speed < 500 else 0.0) + (1.5 if latency > 300 else 0.0) + (0.5 if cpu > 80 else 0.0)
        target_delay = float(np.clip(0.02 * math.exp(0.35 * severity), 0.01, 2.0))

        # ── 4. PREFETCH — segmentos para cubrir startup latency ──────────────────
        startup_est = (1.0 / max(0.1, lat_norm)) * (1 + p_error * 3)
        target_prefetch = float(np.clip(math.ceil(startup_est / 2.0), 1, 20))

        # ── 5. THREADS — inverso al estrés del sistema ───────────────────────────
        target_threads = float(np.clip(4 - round(stress * 3), 1, 4))

        # ── 6. PRESET — más lento cuando hay holgura de CPU ─────────────────────
        # 0=ultrafast, 1=superfast, 2=veryfast, 3=faster
        target_preset = float(np.clip(round(stress * 3), 0, 3))

        # ── 7. RECONNECT — necesario si hay riesgo de fallo ─────────────────────
        target_reconnect = 1.0 if (not success or ratio > 1.2 or latency > 300) else 0.0

        # ── 8. STALL PROB — probabilidad de pausa proporcional a la asfixia ─────
        target_stall = float(np.clip(ratio * 0.5 * (1 - lat_norm), 0.0, 1.0))

        return [
            target_conn,
            target_buf_log,
            target_delay,
            target_prefetch,
            target_threads,
            target_preset,
            target_reconnect,
            target_stall
        ]

    def evolve_brain(self):
        """
        v6.1: Punto de entrada ÚNICO de entrenamiento del modelo.

        Cambios respecto a v5.0:
        - Labels continuas (via _compute_continuous_labels) en lugar de 2 estados binarios
        - Feature #7 = hour_norm (hora/23) en lugar de constante 0.5
        - Replay buffer ampliado: 800 recientes + 6000 críticos/aleatorios
        - Regularización mejorada (alpha=0.001) para evitar overfitting
        - early_stopping=True para detener entrenamiento en el punto óptimo
        - Arquitectura CONGELADA — sin auto-expansión de capas
        """
        import math
        print(" Optimizer: Iniciando fase de consolidación de memoria (v6.1)...")
        conn = sqlite3.connect(DB_PATH)

        # Replay Buffer mejorado: más recientes + críticos priorizados
        recent = conn.execute("""
            SELECT url_domain, size_mb, hour, day_of_week, cpu, ram, latency,
                   target_num_conn, target_buffer, target_delay, target_prefetch, success, actual_speed
            FROM experiences ORDER BY id DESC LIMIT 800
        """).fetchall()
        critical = conn.execute("""
            SELECT url_domain, size_mb, hour, day_of_week, cpu, ram, latency,
                   target_num_conn, target_buffer, target_delay, target_prefetch, success, actual_speed
            FROM experiences WHERE is_critical = 1 ORDER BY RANDOM() LIMIT 4000
        """).fetchall()
        random_hist = conn.execute("""
            SELECT url_domain, size_mb, hour, day_of_week, cpu, ram, latency,
                   target_num_conn, target_buffer, target_delay, target_prefetch, success, actual_speed
            FROM experiences ORDER BY RANDOM() LIMIT 2000
        """).fetchall()
        conn.close()

        data = recent + critical + random_hist
        if len(data) < 20:
            print(" Optimizer: Datos insuficientes para entrenar")
            return

        X, Y = [], []
        for d in data:
            try:
                size_mb  = float(d[1] or 0)
                hour     = float(d[2] or 0)
                cpu      = float(d[4] or 0)
                ram      = float(d[5] or 0)
                lat      = float(d[6] or 0)
                success  = bool(d[11])
                speed    = float(d[12] or 1.0)
                hour_norm = hour / 23.0  # Normalizar hora al rango [0,1]

                starvation_ratio = (2500.0 / speed) if speed > 5.0 else 1.0
                cpu_interaction  = (cpu / 100.0) * starvation_ratio

                # v6.1: 7 features — el último es hora normalizada (no sesgo constante)
                X.append([
                    math.log1p(size_mb),         # 1. Tamaño log
                    cpu,                         # 2. CPU
                    ram,                         # 3. RAM
                    math.sqrt(max(0, lat)),      # 4. Latencia suavizada
                    starvation_ratio,            # 5. Ratio de asfixia
                    cpu_interaction,             # 6. Interacción CPU × ratio
                    hour_norm                    # 7. Hora del día [0.0-1.0]
                ])

                # v6.1: Labels CONTINUAS generadas con matemática derivada
                Y.append(self._compute_continuous_labels(
                    size_mb, cpu, ram, lat, speed, hour, success
                ))
            except Exception:
                continue

        if len(X) < 20:
            print(" Optimizer: Datos corruptos insuficientes para entrenar")
            return

        X = np.array(X)
        Y = np.array(Y)

        with self.lock:
            # Verificar compatibilidad de salidas antes de warm_start
            if self.model is not None and hasattr(self.model, 'n_outputs_') and self.model.n_outputs_ != Y.shape[1]:
                ai_logger.warning(f"Modelo existente tiene {self.model.n_outputs_} salidas "
                                  f"vs {Y.shape[1]} esperadas. Reiniciando modelo.")
                self.model = None
                self.scaler_y = StandardScaler()

            self.scaler_x.fit(X)
            self.scaler_y.fit(Y)
            X_scaled = self.scaler_x.transform(X)
            Y_scaled = self.scaler_y.transform(Y)

            # ── Construir o recuperar modelo con parámetros v6.1 ───────────────
            if self.model is None:
                self.model = MLPRegressor(
                    hidden_layer_sizes=tuple(self.config["layers"]),  # [256, 128, 64, 32] CONGELADO
                    activation='relu',
                    solver='adam',
                    max_iter=800,
                    learning_rate='adaptive',        # Reduce LR automáticamente si loss estanca
                    learning_rate_init=0.005,         # Más conservador que 0.01 anterior
                    alpha=0.001,                      # Regularización L2 mejorada (era 0.0001)
                    early_stopping=True,              # Detiene en punto óptimo (antioverfit)
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    warm_start=True
                )
            else:
                # Actualizar hiperparámetros de regularización en modelo existente
                self.model.alpha              = 0.001
                self.model.learning_rate      = 'adaptive'
                self.model.learning_rate_init = 0.005
                self.model.early_stopping     = True
                self.model.n_iter_no_change   = 20
                self.model.warm_start         = True

            self.model.fit(X_scaled, Y_scaled)

            error = self.model.loss_
            print(f" Optimizer: Evolución v6.1 MLP completada. Loss={error:.6f} | Samples={len(X)}")

            # ── 2. Entrenar Motor Principal LightGBM (v7.0) ───────────────────
            if HAS_LIGHTGBM:
                try:
                    base_lgb = lgb.LGBMRegressor(
                        n_estimators=120,
                        learning_rate=0.03,
                        max_depth=5,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        verbosity=-1,
                        n_jobs=-1
                    )
                    self.lgb_model = MultiOutputRegressor(base_lgb)
                    self.lgb_model.fit(X, Y)
                    joblib.dump(self.lgb_model, LIGHTGBM_MODEL_PATH)

                    feature_names = ["log_size", "cpu", "ram", "sqrt_lat", "ratio", "cpu_interaction", "hour_norm"]
                    importances = np.mean([est.feature_importances_ for est in self.lgb_model.estimators_], axis=0)
                    total_imp = np.sum(importances)
                    norm_importances = (importances / total_imp).tolist() if total_imp > 0 else [0.0]*7
                    
                    self.config["primary_engine"] = "LightGBM"
                    self.config["lgb_feature_importances"] = dict(zip(feature_names, [round(x, 4) for x in norm_importances]))
                    print(f" Optimizer: LightGBM v7.0 (Motor Principal) entrenado. Importancias: {self.config['lgb_feature_importances']}")
                except Exception as lgb_train_err:
                    ai_logger.error(f" Optimizer: Error entrenando LightGBM: {lgb_train_err}")

            self.config["total_trainings"] += 1
            self.config["avg_error"] = error
            self.config["last_evolve_samples"] = len(X)
            self.save_config()

            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler_x, os.path.join(CACHE_DIR, "iptv_scaler_x_v4.pkl"))
            joblib.dump(self.scaler_y, os.path.join(CACHE_DIR, "iptv_scaler_y_v4.pkl"))

            # Exportar nuevo ONNX con modelo actualizado
            try:
                self.export_onnx_model()
            except Exception as onnx_err:
                ai_logger.warning(f"Export ONNX falló (no crítico): {onnx_err}")

            print(" Optimizer: Evolución completada. Memoria consolidada.")

    def expand_architecture(self):
        """
        v6.1: DESACTIVADA — no se llama en producción.
        Mantener como método para uso manual en experimentos controlados.
        La auto-expansión causaba overfitting progresivo en el modelo MLP
        ya que añadía capas ante cualquier loss < 0.1 (fácil de lograr con labels binarias).
        """
        ai_logger.warning("expand_architecture() llamada pero DESACTIVADA en v6.1. Ignorando.")

# Instancia Global del Cerebro
ai_optimizer = IPTVEvolutionaryBrain()


# ══════════════════════════════════════════════════════════════════
# THROTTLE DETECTOR — Detección y evasión de limitación de velocidad
# ══════════════════════════════════════════════════════════════════
class ThrottleDetector:
    """
    Detecta cuando un servidor IPTV está limitando (throttling) la velocidad
    de descarga y decide cuántas conexiones paralelas lanzar para compensarlo.

    Algoritmo:
      1. Mantiene una ventana deslizante de las últimas N velocidades por dominio
      2. Detecta throttle si: velocidad_actual < promedio_histórico * THRESHOLD
      3. Calcula el número óptimo de piezas con: n = ceil(velocidad_esperada / velocidad_real)
      4. Limita entre MIN_PIECES y MAX_PIECES para no hacer flood
      5. Decae gradualmente cuando la velocidad se recupera (anti-flapping)

    Anti-detección:
      Cada pieza usa una identidad de cliente diferente para evadir el
      rate-limiting por fingerprint de User-Agent/IP-connection.
    """

    WINDOW_SIZE     = 20    # Número de muestras en la ventana histórica
    THROTTLE_RATIO  = 0.50  # Detectar throttle si velocidad < 50% del historial
    MIN_PIECES      = 1
    MAX_PIECES      = 32    # Aumentado para descargas asíncronas de altísima velocidad
    DECAY_FACTOR    = 0.85  # Al recuperar velocidad, reducir piezas gradualmente

    # Banco de User-Agents: mezcla de browsers, players y bots legítimos
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
        "VLC/3.0.20 LibVLC/3.0.20",
        "Kodi/20.2 (Windows 10.0; Win64; x64) App_Bitness/64 Version/20.2",
        "Mozilla/5.0 (SMART-TV; Linux; Tizen 6.5) AppleWebKit/537.36",
        "ExoPlayerLib/2.19.1 (Linux; U; Android 13; Pixel 7)",
        "stagefright/1.2 (Linux;Android 14)",
        "Mozilla/5.0 (iPad; CPU OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1",
    ]

    # Referers que parecen legítimos según el dominio
    REFERER_TEMPLATES = [
        "https://www.google.com/",
        "https://duckduckgo.com/",
        "https://{domain}/",
        "https://m.{domain}/",
        "https://app.{domain}/player",
    ]

    def __init__(self):
        self.lock            = threading.Lock()
        self.speed_windows   = {}   # domain → deque de velocidades KB/s
        self.piece_counts    = {}   # domain → n piezas activo
        self.throttle_states = {}   # domain → {"throttled": bool, "since": timestamp}
        self.expected_speeds = {}   # domain → velocidad esperada (promedio largo plazo)

    def _domain(self, url: str) -> str:
        try:    return url.split('/')[2]
        except: return "unknown"

    def record_speed(self, url: str, speed_kbps: float):
        """Registra una medición de velocidad para el dominio de la URL."""
        domain = self._domain(url)
        with self.lock:
            if domain not in self.speed_windows:
                self.speed_windows[domain] = collections.deque(maxlen=self.WINDOW_SIZE)
            self.speed_windows[domain].append(speed_kbps)

            # Actualizar velocidad esperada con EMA (Exponential Moving Average)
            if domain not in self.expected_speeds:
                self.expected_speeds[domain] = speed_kbps
            else:
                alpha = 0.15   # Peso de la nueva muestra
                self.expected_speeds[domain] = (
                    alpha * speed_kbps + (1 - alpha) * self.expected_speeds[domain]
                )

    def analyze(self, url: str, current_speed_kbps: float) -> dict:
        """
        Analiza si hay throttle y recomienda acción.
        
        Returns dict:
          throttled     : bool
          pieces        : int (1-5) — piezas paralelas recomendadas
          expected_kbps : float — velocidad esperada sin throttle
          ratio         : float — current/expected
          strategy      : str  — descripción de la estrategia
        """
        domain = self._domain(url)
        with self.lock:
            win   = self.speed_windows.get(domain, collections.deque())
            exp   = self.expected_speeds.get(domain, current_speed_kbps)
            prev_pieces = self.piece_counts.get(domain, 1)

        if len(win) < 3:
            # Sin historial suficiente: asumir sin throttle
            return {"throttled": False, "pieces": 1, "expected_kbps": exp,
                    "ratio": 1.0, "strategy": "learning"}

        ratio = current_speed_kbps / max(exp, 1.0)
        throttled = ratio < self.THROTTLE_RATIO

        if throttled:
            # Calcular cuántas piezas necesitamos:
            # n = ceil(expected / current) — si esperamos 5000 y tenemos 1000, necesitamos 5x
            import math
            raw_pieces = math.ceil(exp / max(current_speed_kbps, 1.0))
            pieces = int(np.clip(raw_pieces, 2, self.MAX_PIECES))

            strategy = f"THROTTLE DETECTADO ({ratio:.0%} de velocidad normal) → {pieces} piezas paralelas"

            with self.lock:
                self.piece_counts[domain]    = pieces
                if not self.throttle_states.get(domain, {}).get("throttled"):
                    self.throttle_states[domain] = {"throttled": True, "since": time.time()}
        else:
            # Velocidad recuperada: decaer número de piezas gradualmente
            decayed = max(1, int(prev_pieces * self.DECAY_FACTOR))
            pieces   = decayed
            strategy = f"Normal ({ratio:.0%} de velocidad esperada)"

            with self.lock:
                self.piece_counts[domain]    = pieces
                self.throttle_states[domain] = {"throttled": False, "since": time.time()}

        return {
            "throttled":     throttled,
            "pieces":        pieces,
            "expected_kbps": exp,
            "ratio":         ratio,
            "strategy":      strategy,
        }

    def get_disguised_headers(self, piece_index: int, base_url: str) -> dict:
        """
        Genera headers únicos para cada pieza para evadir detección por fingerprint.
        
        Técnicas:
          - User-Agent distinto por pieza (pool de 10 identidades)
          - Referer rotativo y semi-legítimo
          - Accept-Language aleatorio
          - Cache-Control variado
          - Jitter temporal (quien llama debe dormir random.uniform(0, 0.3) antes de lanzar)
        """
        import random as _rng
        domain = self._domain(base_url)

        ua = self.USER_AGENTS[piece_index % len(self.USER_AGENTS)]

        # Referer: usa el dominio del servidor para parecer legítimo
        ref_template = self.REFERER_TEMPLATES[piece_index % len(self.REFERER_TEMPLATES)]
        referer = ref_template.replace("{domain}", domain)

        # Accept-Language: varía entre en-US, es-ES, pt-BR, etc.
        languages = ["en-US,en;q=0.9", "es-ES,es;q=0.9,en;q=0.8",
                     "pt-BR,pt;q=0.9", "en-GB,en;q=0.9", "fr-FR,fr;q=0.8,en;q=0.6"]
        accept_lang = languages[piece_index % len(languages)]

        # Cache-Control: algunos dicen no-cache, otros max-age
        cache_opts = ["no-cache", "max-age=0", "no-store", "max-age=3600"]
        cache_ctrl = cache_opts[piece_index % len(cache_opts)]

        return {
            "User-Agent":      ua,
            "Referer":         referer,
            "Accept":          "*/*",
            "Accept-Language": accept_lang,
            "Accept-Encoding": "identity",   # Sin compresión en piezas de video
            "Cache-Control":   cache_ctrl,
            "Connection":      "keep-alive",
            "Pragma":          "no-cache" if piece_index % 2 == 0 else "",
        }

    def get_jitter_delay(self, piece_index: int) -> float:
        """
        Retorna un delay en segundos para escalonar el inicio de cada pieza.
        Evita que N conexiones exactamente simultáneas sean detectadas como burst.
        
        Patrón: delay = piece_index * base_ms + gaussian_noise
        """
        import random as _rng
        base_ms = 0.02   # 20ms entre piezas (Optimizado para asincronía)
        noise   = abs(_rng.gauss(0, 0.01))
        return piece_index * base_ms + noise


# Instancia global del detector de throttle
throttle_detector = ThrottleDetector()
