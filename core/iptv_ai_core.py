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
import logging
import collections
import sys

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

# Configuración de Rutas
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
DB_PATH = os.path.join(CACHE_DIR, "iptv_permanent_memory.db")
MODEL_PATH = os.path.join(CACHE_DIR, "iptv_dynamic_brain.pkl")
SCALER_PATH = os.path.join(CACHE_DIR, "iptv_scaler_v3.pkl")
CONFIG_PATH = os.path.join(CACHE_DIR, "brain_config.json")

os.makedirs(CACHE_DIR, exist_ok=True)

class IPTVEvolutionaryBrain:
    def __init__(self):
        self.lock = threading.Lock()
        self.init_db()
        self.load_config() # This sets self.config
        self.scaler_x = self.load_scaler_x()
        self.scaler_y = self.load_scaler_y()
        self.model = self.load_model()
        self.error_history = []
        self.experience_count = self.get_total_experiences()
        self._last_retrain_count = 0
        self.start_auto_retraining()

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
                is_critical INTEGER
            )
        ''')
        # Migración simple: verificar si existen las nuevas columnas
        cursor.execute("PRAGMA table_info(experiences)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'channel_name' not in columns:
            cursor.execute("ALTER TABLE experiences ADD COLUMN channel_name TEXT")
        if 'full_url' not in columns:
            cursor.execute("ALTER TABLE experiences ADD COLUMN full_url TEXT")
        conn.commit()
        conn.close()

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
        def _daemon():
            while True:
                time.sleep(interval_minutes * 60)
                current_count = self.get_total_experiences()
                nuevos_rows = current_count - getattr(self, "_last_retrain_count", 0)
                if nuevos_rows > 100:
                    try:
                        self._retrain()
                        self._last_retrain_count = self.get_total_experiences()
                    except Exception as e:
                        print(f"Error en auto-retraining: {e}")
                        
        t = threading.Thread(target=_daemon, daemon=True)
        t.start()

    def _retrain(self):
        import math
        # Limpieza de redundancia: np y threading ya están en el global
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT size_mb, cpu, ram, latency, actual_speed
            FROM experiences
            WHERE actual_speed IS NOT NULL AND actual_speed > 0
            ORDER BY is_critical DESC, id DESC LIMIT 8000
        ''')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return

        # --- GUARDIA DE MODELO ---
        if self.model is None:
            layers = self.config.get("layers", [256, 128, 64, 32])
            self.model = MLPRegressor(
                hidden_layer_sizes=layers,
                activation='relu',
                solver='adam',
                max_iter=500,
                warm_start=True
            )
            print(f"   ⚠️ [Cerebro] Modelo inexistente/corrupto. Re-inicializando v5...")

        if not rows:
            return

        raw_X = []
        raw_Y = []
        for rc in rows:
            size_mb, cpu, ram, latency, actual_speed = rc
            ratio = (size_mb * 1024 / 2) / max(actual_speed, 1.0)
            
            x_feat = [
                math.log1p(size_mb),
                cpu,
                ram,
                math.sqrt(max(0, latency)),
                ratio,
                (cpu * ratio) / 100.0,
                0.5
            ]
            raw_X.append(x_feat)
            
            if ratio > 1.2:
                y_feat = [2, math.log(16384), min(2.5, 0.4 * ratio), min(35, 6 * ratio)]
            else:
                y_feat = [8, math.log(2048), 0.05, 4]
            raw_Y.append(y_feat)

        X = np.array(raw_X)
        Y = np.array(raw_Y)

        with self.lock:
            # --- PROTECCIÓN ANTI-DRIFT ---
            # Solo usamos transform() si los scalers ya fueron fitteados por el modelo v5 original.
            # Si son nuevos StandardScaler(), fit_transform() es necesario la PRIMERA VEZ.
            # No fittear escaladores en caliente si ya tienen datos (Evita Deriva / Drift)
            if hasattr(self.scaler_x, 'mean_'):
                X_scaled = self.scaler_x.transform(X)
                Y_scaled = self.scaler_y.transform(Y)
            else:
                X_scaled = self.scaler_x.fit_transform(X)
                Y_scaled = self.scaler_y.fit_transform(Y)
            
            if hasattr(self.model, "warm_start"):
                self.model.warm_start = True
                self.model.max_iter = 500
            
            self.model.fit(X_scaled, Y_scaled)
            
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler_x, os.path.join(os.path.join(os.path.dirname(__file__), '..', 'cache'), "iptv_scaler_x_v4.pkl"))
            joblib.dump(self.scaler_y, os.path.join(os.path.join(os.path.dirname(__file__), '..', 'cache'), "iptv_scaler_y_v4.pkl"))
            
            print(f"   [Cerebro] 🧠 Auto-Retrain completado (n={len(X)})")

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


    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                return joblib.load(MODEL_PATH)
            except:
                return None
        return None

    def get_features(self, url, size_mb, latency=0, speed_kbps=0.0, required_bitrate=0.0):
        """
        Extrae 7 features para la v5 del Cerebro de Combate de 256 neuronas
        """
        import math
        
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        # 5. ratio = required_bitrate / max(speed_kbps, 1.0)  — si speed_kbps <= 5.0 usar 1.0
        if speed_kbps <= 5.0:
            ratio = 1.0
        else:
            ratio = required_bitrate / max(speed_kbps, 1.0)
            
        return np.array([[
            math.log1p(size_mb),               # 1
            cpu,                               # 2
            ram,                               # 3
            math.sqrt(max(0, latency)),        # 4
            ratio,                             # 5
            (cpu * ratio) / 100.0,             # 6
            0.5                                # 7 (sesgo)
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
        print(f"   🧠 IA ABR (BOLA) | Buffer: {buffer_ms}ms -> Selección Lyapunov: {res} a {br}kbps")
        return best_quality


    def predict_optimal_config(self, url, size_mb, latency=0, speed_kbps=0.0, required_bitrate=0.0):
        """Predice la mejor configuración basándose en el estado del sistema, red y MEMORIA de fallos"""
        with self.lock:
            sys_cpu = psutil.cpu_percent()
            sys_ram = psutil.virtual_memory().percent
            
            # 🧠 SEGUNDO CEREBRO: Memoria Histórica del Servidor IPTV
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

            if self.model is None:
                config = self.expert_fallback(size_mb, recent_failures)
                reason = f"Lógica Experto | CPU: {sys_cpu}%"
                if recent_failures > 0: reason += f" | Historial Fallos Servidor: {recent_failures}"
                logging.info(f"🧠 IA RAZONAMIENTO | {reason} | URL: {url[:30]}... | OUT: {config}")
                return config
            try:
                import math
                X_features = self.get_features(url, size_mb, latency, speed_kbps, required_bitrate)
                X_scaled = self.scaler_x.transform(X_features)
                pred_scaled = self.model.predict(X_scaled)[0]
                pred = self.scaler_y.inverse_transform([pred_scaled])[0]
                
                # Desempaquetar y limpiar predicción (buffer logarítmico a real)
                c_conn, buf_log, delay, pref = pred
                
                buf_real = math.exp(buf_log) if buf_log < 15 else 16384 # Protección para no desbordar
                
                # Limitamos num_conn drásticamente (max 4) para evitar bloqueos del proveedor IPTV
                num_conn = int(np.clip(round(c_conn), 1, 4))
                buffer_size = int(np.clip(round(buf_real), 128, 16384))
                retry_delay = float(np.clip(delay, 0.01, 3.0))
                prefetch_count = int(np.clip(round(pref), 1, 40))
                
                # MEMORIA PREVENTIVA: Castigo a servidores rebeldes
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
                    "prefetch_count": prefetch_count
                }
                
                reason = f"Predicción Neuronal basada en Latencia {latency:.0f}ms y Carga {sys_cpu}%"
                if getattr(self, 'consecutive_errors', 0) > 2:
                    config["num_conn"] = max(2, config["num_conn"] // 2)
                    reason += " | PENALIZACIÓN errores seguidos"
                if recent_failures > 0:
                    reason += f" | MEMORIA SEVERA: {recent_failures} fallos origen 24h"
                
                logging.info(f"🧠 IA DECISIÓN | {reason} | URL: {url[:30]}... | DECISIÓN: {config}")
                return config
            except Exception as e:
                logging.error(f"❌ IA ERROR en predicción: {e}")
                return self.expert_fallback(size_mb, recent_failures)

    def expert_fallback(self, size_mb, failures=0):
        # Fallback mucho más agresivo si sabemos que el server falla mucho
        if failures > 3:
            return {"num_conn": 2, "buffer_kb": 8192, "retry_delay": 5.0, "prefetch_count": 2}
        elif size_mb > 5:
            return {"num_conn": 10, "buffer_kb": 4096, "retry_delay": 2.0, "prefetch_count": 6}
        return {"num_conn": 8, "buffer_kb": 2500, "retry_delay": 1.0, "prefetch_count": 4}

    def log_experience(self, url, size_mb, speed_kbps, success, latency=0, channel_name=None):
        """Guarda la experiencia en la memoria permanente (SQLite) con logs de éxito/fallo"""
        logging.info(f"💾 IA MEMORIA | Registrando evento: {channel_name or url[:30]}... | Velocidad: {speed_kbps:.0f} KB/s | Éxito: {success}")
        t = time.localtime()
        domain = url.split('/')[2] if '/' in url else "unknown"
        
        # Determinar si es una experiencia crítica (ej: fallo o velocidad muy baja)
        is_critical = 1 if (not success or speed_kbps < 1000) else 0
        
        # Targets ideales para el aprendizaje
        target_num_conn = 32 if is_critical else 8
        target_buffer = 4096 if is_critical else 1024
        target_delay = 0.4 if is_critical else 0.05
        target_prefetch = 8 if is_critical else 3

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO experiences 
            (timestamp, url_domain, full_url, channel_name, size_mb, hour, day_of_week, cpu, ram, latency, 
             target_num_conn, target_buffer, target_delay, target_prefetch, actual_speed, success, is_critical)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (time.time(), domain, url, channel_name, size_mb, t.tm_hour, t.tm_wday, psutil.cpu_percent(), 
              psutil.virtual_memory().percent, latency, target_num_conn, target_buffer, 
              target_delay, target_prefetch, speed_kbps, 1 if success else 0, is_critical))
        conn.commit()
        conn.close()
        
        self.experience_count += 1
        if self.experience_count % 100 == 0:
            threading.Thread(target=self.evolve_brain, daemon=True).start()

    def find_best_mirror(self, target_name):
        """Busca un mirror similar para un canal que falló"""
        if not target_name:
            return None
            
        print(f"🔍 IA: Buscando mirror para '{target_name}'...")
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
            print(f"✨ IA: Mirror encontrado: {res[0]} (speed: {res[1]:.0f} KB/s)")
            return res[0]
        return None

    def log_failure(self, url):
        """Marca una URL como no confiable"""
        print(f"💔 IA: Registrando fallo crítico en URL: {url[:60]}...")
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

    def evolve_brain(self):
        """Sincronización a 12 features para precisión de Nivel 2"""
        import math
        print("🧠 IA: Iniciando fase de consolidación de memoria...")
        conn = sqlite3.connect(DB_PATH)
        # Replay Buffer: 100 recientes + 1000 históricos aleatorios 
        recent = conn.execute("""
            SELECT url_domain, size_mb, hour, day_of_week, cpu, ram, latency,
                   target_num_conn, target_buffer, target_delay, target_prefetch, success, actual_speed
            FROM experiences ORDER BY id DESC LIMIT 100
        """).fetchall()
        critical = conn.execute("""
            SELECT url_domain, size_mb, hour, day_of_week, cpu, ram, latency,
                   target_num_conn, target_buffer, target_delay, target_prefetch, success, actual_speed
            FROM experiences WHERE is_critical = 1 ORDER BY RANDOM() LIMIT 1000
        """).fetchall()
        conn.close()
        
        data = recent + critical
        if len(data) < 20: return

        X, Y = [], []
        for d in data:
            try:
                domain_hash = hash(d[0]) % 100
                size_mb, hour, dow, cpu, ram = float(d[1] or 0), float(d[2] or 0), float(d[3] or 0), float(d[4] or 0), float(d[5] or 0)
                lat = float(d[6] or 0)
                success = d[11]
                speed = float(d[12] or 1.0)
                
                starvation_ratio = (2500.0 / speed) if speed > 5.0 else 1.0
                cpu_interaction = (cpu / 100.0) * starvation_ratio

                # 🧠 RECONSTRUCCIÓN EXACTA DE LAS 7 FEATURES (v5)
                X.append([
                    math.log1p(size_mb),                   # 1. log-size
                    cpu,                                   # 2. cpu
                    ram,                                   # 3. ram
                    math.sqrt(max(0, lat)),                # 4. sqrt latency
                    starvation_ratio,                      # 5. Ratio asfixia
                    cpu_interaction,                       # 6. CPU Interacción
                    0.5                                    # 7. Constante sesgo
                ])
                
                target_conn = float(d[7] or 8)
                if not success:
                    target_conn = max(2.0, target_conn * 0.5)

                Y.append([
                    target_conn,
                    math.log(max(1.0, float(d[8] or 1024))), # Convertir a logarítmico para retrocompatibilidad
                    float(d[9] or 0.1),
                    float(d[10] or 4),
                ])
            except Exception:
                continue

        if len(X) < 20:
            print("⚠️ IA: Datos insuficientes o corruptos para entrenar")
            return

        X = np.array(X)
        Y = np.array(Y)

        with self.lock:
            self.scaler_x.fit(X)
            self.scaler_y.fit(Y)
            
            X_scaled = self.scaler_x.transform(X)
            Y_scaled = self.scaler_y.transform(Y)
            
            # Si no hay modelo o queremos evolucionar
            if self.model is None:
                self.model = MLPRegressor(
                    hidden_layer_sizes=tuple(self.config["layers"]),
                    activation='relu', solver='adam', max_iter=600,
                    learning_rate='invscaling', learning_rate_init=0.01, power_t=0.5,
                    alpha=0.0001, warm_start=True
                )
            else:
                self.model.warm_start = True
                self.model.alpha = 0.0001

            
            # Entrenamiento Incremental
            self.model.fit(X_scaled, Y_scaled)
            
            # Evaluar si necesitamos expandir el cerebro
            error = self.model.loss_
            print(f"📊 IA: Error actual del cerebro: {error:.6f}")
            
            if error < 0.1 and self.config["total_trainings"] > 5:
                self.expand_architecture()
            
            self.config["total_trainings"] += 1
            self.config["avg_error"] = error
            self.save_config()
            
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler_x, os.path.join(os.path.join(os.path.dirname(__file__), '..', 'cache'), "iptv_scaler_x_v4.pkl"))
            joblib.dump(self.scaler_y, os.path.join(os.path.join(os.path.dirname(__file__), '..', 'cache'), "iptv_scaler_y_v4.pkl"))
            print("✅ IA: Evolución completada. Memoria consolidada.")


    def expand_architecture(self):
        """Añade una nueva capa de neuronas para manejar mayor complejidad"""
        self.config["complexity_level"] += 1
        new_layer = 16 + (self.config["complexity_level"] * 4)
        self.config["layers"].insert(0, new_layer)
        print(f"🚀 IA: ¡AUTO-EXPANSIÓN! Nueva estructura cerebral: {self.config['layers']}")
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(self.config["layers"]),
            activation='relu', solver='adam', max_iter=600,
            learning_rate='invscaling', learning_rate_init=0.01, power_t=0.5,
            alpha=0.0001, warm_start=True
        )
        self.save_config()

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
    MAX_PIECES      = 5
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
        base_ms = 0.08   # 80ms entre piezas
        noise   = abs(_rng.gauss(0, 0.03))
        return piece_index * base_ms + noise


# Instancia global del detector de throttle
throttle_detector = ThrottleDetector()
