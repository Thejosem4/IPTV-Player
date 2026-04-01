#!/usr/bin/env python3
"""
ai_teacher.py — Sistema Teacher-Student v3.0 (Simulador de Exámenes)
═══════════════════════════════════════════════════════════════════════════
Arquitectura "Offline Reinforcement Learning":
  1. Simulador: Genera escenarios de red aleatorios y extremos.
  2. Estudiante: La red neuronal actual (MLPRegressor) intenta resolverlo.
  3. Maestro (RuleEngine/Ollama): Calcula la respuesta matemáticamente perfecta.
  4. Evaluación: Si el estudiante comete un error grave, se reprueba el examen.
  5. Entrenamiento: Al acumular 50 reprobados, se reentrena la red ("replay buffer").
"""

import sys, io, os, time, math, cmath, random, sqlite3, json, hashlib, re
import argparse
import requests
import warnings

# Silenciar warnings de scikit-learn por versiones de features
warnings.filterwarnings("ignore", category=UserWarning)

# Forzar UTF-8 para emojis en consola Windows/Linux
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except: pass

try:
    import numpy as np
    import joblib
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("❌ Faltan dependencias. Ejecuta: pip install numpy scikit-learn joblib requests")
    sys.exit(1)

# ── Configuración ────────────────────────────────────────────────────────
CACHE_DIR      = os.path.join(os.path.dirname(__file__), '..', 'cache')
GOLD_DB        = os.path.join(CACHE_DIR, "iptv_gold_memory.db")
MODEL_PATH     = os.path.join(CACHE_DIR, "iptv_dynamic_brain.pkl")
SCALER_X_PATH  = os.path.join(CACHE_DIR, "iptv_scaler_x_v4.pkl")
SCALER_Y_PATH  = os.path.join(CACHE_DIR, "iptv_scaler_y_v4.pkl")
CONFIG_PATH    = os.path.join(CACHE_DIR, "brain_config.json")

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "qwen2.5:3b"
BATCH_TO_TRAIN = 50  # Exámenes reprobados necesarios para disparar reentrenamiento

# ── Colores ANSI ─────────────────────────────────────────────────────────
class C:
    RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
    RED="\033[91m"; GREEN="\033[92m"; YELLOW="\033[93m"
    BLUE="\033[94m"; MAGENTA="\033[95m"; CYAN="\033[96m"

# ═══════════════════════════════════════════════════════════════════════════
# 1. GENERADOR DE ESCENARIOS SINTÉTICOS
# ═══════════════════════════════════════════════════════════════════════════
class ScenarioGenerator:
    DOMAINS = ["cdn.iptv-pro.com", "free-iptv.org", "edge.ultrahdtv.net", "p2p.iptv-share.net"]
    CHANNELS = ["ESPN HD", "HBO Max 4K", "VOD Movie", "News 24h SD"]

    @classmethod
    def generate(cls):
        """Genera un escenario de red, forzando casos límite el 40% de las veces."""
        mode = random.choices(["normal", "low_speed", "high_latency", "system_stress"], 
                              weights=[0.6, 0.15, 0.15, 0.10])[0]

        if mode == "normal":
            speed = random.uniform(5000, 25000)
            lat = random.uniform(10, 80)
            cpu, ram = random.uniform(10, 50), random.uniform(20, 60)
        elif mode == "low_speed":
            speed = random.uniform(100, 1200) # Colapso de ancho de banda
            lat = random.uniform(50, 150)
            cpu, ram = random.uniform(30, 70), random.uniform(40, 80)
        elif mode == "high_latency":
            speed = random.uniform(2000, 10000)
            lat = random.uniform(300, 1500) # Latencia satelital/P2P
            cpu, ram = random.uniform(40, 80), random.uniform(50, 80)
        else: # system_stress
            speed = random.uniform(1000, 5000)
            lat = random.uniform(50, 200)
            cpu, ram = random.uniform(85, 99), random.uniform(85, 99) # Servidor asfixiado

        return {
            "id": random.randint(1000000, 9999999),
            "url_domain": random.choice(cls.DOMAINS),
            "channel_name": random.choice(cls.CHANNELS),
            "size_mb": random.uniform(0.5, 15.0),
            "hour": random.randint(0, 23),
            "day_of_week": random.randint(0, 6),
            "cpu": cpu,
            "ram": ram,
            "latency": lat,
            "actual_speed": speed,
            "success": 0, # Asumimos fallo para obligar a buscar corrección
            "is_critical": 1 if speed < 1500 or lat > 500 else 0,
            "mode": mode
        }

# ═══════════════════════════════════════════════════════════════════════════
# 2. INFERENCIA DEL ESTUDIANTE (RED NEURONAL)
# ═══════════════════════════════════════════════════════════════════════════
class AIStudent:
    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.load_brain()

    def load_brain(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_X_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.scaler_x = joblib.load(SCALER_X_PATH)
                self.scaler_y = joblib.load(SCALER_Y_PATH)
            except Exception as e:
                print(f"{C.RED}Error cargando cerebro del estudiante: {e}{C.RESET}")
                self.model = None

    def predict(self, rec):
        """Devuelve la predicción del estudiante. Si no hay modelo, devuelve un baseline malo."""
        if not self.model or not hasattr(self.scaler_x, 'mean_'):
            return {"c": 12, "b": 512, "d": 0.05, "p": 2} # Dummy malo para forzar aprendizaje

        # 12 Features (idéntico al núcleo)
        size_mb, hour, dow = rec["size_mb"], rec["hour"], rec["day_of_week"]
        cpu, ram, lat = rec["cpu"], rec["ram"], rec["latency"]
        domain_hash = hash(rec["url_domain"]) % 100

        features = [
            math.log1p(size_mb),
            math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),   math.cos(2 * math.pi * dow / 7),
            cpu, ram,
            (cpu * ram) / 10000,
            domain_hash,
            math.sqrt(max(0, lat)),
            math.log1p(max(0, lat)),
            0.5
        ]

        try:
            X_scaled = self.scaler_x.transform([features])
            pred_scaled = self.model.predict(X_scaled)[0]
            pred = self.scaler_y.inverse_transform([pred_scaled])[0]

            return {
                "c": int(np.clip(round(pred[0]), 2, 48)),
                "b": int(np.clip(round(math.exp(pred[1])), 128, 8192)),
                "d": float(np.clip(pred[2], 0.01, 2.0)),
                "p": int(np.clip(round(pred[3]), 1, 30))
            }
        except Exception:
            # Fallback seguro en caso de dimensiones corruptas de la matriz guardada
            return {"c": 8, "b": 1024, "d": 0.1, "p": 4}

# ═══════════════════════════════════════════════════════════════════════════
# 3. EL MAESTRO (RULE ENGINE / OLLAMA)
# ═══════════════════════════════════════════════════════════════════════════
class RuleEngine:
    BW_MAX = 50000.0
    LAT_REF = 100.0

    @classmethod
    def get_perfect_targets(cls, rec):
        speed, lat = rec["actual_speed"], rec["latency"]
        cpu, ram = rec["cpu"], rec["ram"]
        
        # 1. Conexiones (BOLA)
        if speed < 1500:
            conn = max(2, min(3, round(speed / 750)))
        else:
            q2 = abs(complex(min(speed/cls.BW_MAX, 1.0), 1.0/(1.0+lat/cls.LAT_REF)))**2
            stress = math.sqrt(max(0.001, cpu/100) * max(0.001, ram/100))
            bw_log = math.log2(speed/1000 + 1)
            conn = int(max(2, min(48, round(16.0 * q2 * bw_log * (1 - stress/2)))))
            if cpu > 85: conn = max(2, conn // 2)

        # 2. Buffer (Shannon)
        p_error = min(0.99, max(0.01, (1 - min(speed/cls.BW_MAX, 0.99)) * min(lat/500, 1.0)))
        entropy = -(p_error*math.log2(p_error) + (1-p_error)*math.log2(1-p_error))
        snr = max(0.1, speed / (1 + lat*10))
        buf = int(max(128, min(8192, 256 + 4096 * (entropy / max(0.1, math.log2(math.log2(1+snr)+1))))))
        if speed < 1500 or lat > 500: buf = int(min(8192, buf * 1.5))

        # 3. Delay & Prefetch
        severity = (2.0 if speed<500 else 0.0) + (1.5 if lat>300 else 0.0) + (0.5 if cpu>80 else 0.0)
        delay = round(max(0.01, min(2.0, 0.02 * math.exp(0.35 * severity))), 4)
        prefetch = int(max(1, min(15, math.ceil(((1.0/(1.0/(1.0+lat/cls.LAT_REF))) * (1+p_error*3))/2.0))))

        return {"c": conn, "b": buf, "d": delay, "p": prefetch}

class OllamaTeacher:
    @staticmethod
    def ask(rec, student_pred):
        prompt = (f"Rol: Experto ABR IPTV. Regla: Sacrificar resolución por fluidez.\n"
                  f"Escenario: Speed {rec['actual_speed']:.0f}KB/s, Lat {rec['latency']:.0f}ms, CPU {rec['cpu']:.0f}%.\n"
                  f"El estudiante sugirió: conn={student_pred['c']}, buffer={student_pred['b']}.\n"
                  f"Corrige aplicando Shannon y Lyapunov. Responde SOLO JSON con claves: target_num_conn, target_buffer, target_delay, target_prefetch.")
        try:
            r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, 
                                                "options": {"temperature": 0.1, "num_predict": 128, "num_ctx": 512}}, timeout=10)
            text = r.json().get("response", "")
            m = re.search(r'\{.*?\}', text, re.DOTALL)
            if m:
                j = json.loads(m.group(0))
                return {"c": j.get("target_num_conn", 8), "b": j.get("target_buffer", 1024), 
                        "d": j.get("target_delay", 0.1), "p": j.get("target_prefetch", 4)}
        except: pass
        return None

# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUADOR DE EXÁMENES
# ═══════════════════════════════════════════════════════════════════════════
class ExamEvaluator:
    @staticmethod
    def grade(rec, student, teacher):
        """Evalúa si el estudiante reprobó basándose en desviaciones críticas."""
        speed = rec["actual_speed"]
        lat = rec["latency"]
        
        errors = []
        # Error fatal 1: Muchas conexiones en red asfixiada
        if speed < 1500 and student["c"] > teacher["c"] + 2:
            errors.append(f"Exceso conn en red lenta ({student['c']} vs {teacher['c']})")
        
        # Error fatal 2: Buffer insuficiente en latencia alta
        if lat > 300 and student["b"] < teacher["b"] * 0.7:
            errors.append(f"Buffer muy bajo en alta lat ({student['b']} vs {teacher['b']})")
            
        # Error fatal 3: No hacer backoff en fallo crítico
        if rec["is_critical"] and student["d"] < teacher["d"] * 0.5:
            errors.append(f"Delay insuficiente para shock ({student['d']:.2f} vs {teacher['d']:.2f})")
            
        # Error general: desviación de conexiones cuando es obvia
        elif abs(student["c"] - teacher["c"]) > max(3, teacher["c"] * 0.3):
            errors.append(f"Conexiones ineficientes ({student['c']} vs {teacher['c']})")

        # Calcular magnitud de error general para PER (Prioritized Experience Replay)
        # Se penaliza fuertemente la diferencia porcentual entre lo predicho y lo ideal
        err_c = abs(student["c"] - teacher["c"]) / max(1, teacher["c"])
        err_b = abs(student["b"] - teacher["b"]) / max(1, teacher["b"])
        err_d = abs(student["d"] - teacher["d"]) / max(0.1, teacher["d"])
        error_magnitude = float(err_c + err_b + err_d)

        passed = len(errors) == 0
        return passed, errors, error_magnitude

# ═══════════════════════════════════════════════════════════════════════════
# 5. BASE DE DATOS Y ENTRENADOR
# ═══════════════════════════════════════════════════════════════════════════
class GoldDB:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        c = sqlite3.connect(GOLD_DB)
        c.execute("""CREATE TABLE IF NOT EXISTS gold_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT, original_id INTEGER, timestamp_analysis REAL,
            url_domain TEXT, channel_name TEXT, size_mb REAL, hour INTEGER, day_of_week INTEGER,
            cpu REAL, ram REAL, latency REAL, actual_speed REAL, success INTEGER, is_critical INTEGER,
            orig_target_num_conn INTEGER, orig_target_buffer INTEGER, orig_target_delay REAL, orig_target_prefetch INTEGER,
            target_num_conn INTEGER, target_buffer INTEGER, target_delay REAL, target_prefetch INTEGER,
            analisis_matematico TEXT, estrategia_fluidez TEXT, llm_model TEXT, raw_response TEXT, source TEXT, error_magnitude REAL)""")
        
        # Migraciones v 2.0 -> 3.0 por si cambian columnas
        try: c.execute("ALTER TABLE gold_experiences ADD COLUMN signature TEXT")
        except: pass
        try: c.execute("ALTER TABLE gold_experiences ADD COLUMN error_magnitude REAL DEFAULT 0.0")
        except: pass
        c.commit(); c.close()

    def save_lesson(self, rec, student, teacher, source, error_magnitude=0.0):
        # Generar firma para consistencia de datos si iptv_ai_core la busca
        sig = hashlib.md5(f"{int(rec['actual_speed']//500)}_{int(rec['latency']//50)}_{int(rec['cpu']//10)}".encode()).hexdigest()
        
        c = sqlite3.connect(GOLD_DB)
        c.execute("""INSERT INTO gold_experiences (
            original_id, timestamp_analysis, url_domain, channel_name, size_mb, hour, day_of_week,
            cpu, ram, latency, actual_speed, success, is_critical, signature, error_magnitude,
            orig_target_num_conn, orig_target_buffer, orig_target_delay, orig_target_prefetch,
            target_num_conn, target_buffer, target_delay, target_prefetch, source
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            rec["id"], time.time(), rec["url_domain"], rec["channel_name"], rec["size_mb"], rec["hour"], rec["day_of_week"],
            rec["cpu"], rec["ram"], rec["latency"], rec["actual_speed"], rec["success"], rec["is_critical"], sig, error_magnitude,
            student["c"], student["b"], student["d"], student["p"],
            teacher["c"], teacher["b"], teacher["d"], teacher["p"], source
        ))
        c.commit(); c.close()

class StudentTrainer:
    @staticmethod
    def train(turbo=False):
        if not turbo:
            print(f"\n  {C.MAGENTA}{C.BOLD}🧠 Acumulados {BATCH_TO_TRAIN} reprobados. Iniciando Reentrenamiento (Replay Buffer)...{C.RESET}")
        try:
            # Replay Buffer PER: 4K peores errores + 4K aleatorios + 2K recientes
            c = sqlite3.connect(GOLD_DB); c.row_factory = sqlite3.Row
            data = c.execute("SELECT * FROM gold_experiences ORDER BY error_magnitude DESC LIMIT 4000").fetchall()
            data += c.execute("SELECT * FROM gold_experiences ORDER BY RANDOM() LIMIT 4000").fetchall()
            data += c.execute("SELECT * FROM gold_experiences ORDER BY id DESC LIMIT 2000").fetchall()
            c.close()

            if len(data) < 50: 
                print(f"  {C.YELLOW}⚠️ Muy pocos datos en BD ({len(data)}), omitiendo entreno.{C.RESET}")
                return

            X, Y = [], []
            for r in data:
                try:
                    features = [
                        math.log1p(float(r["size_mb"] or 0)),
                        math.sin(2 * math.pi * float(r["hour"] or 0) / 24), math.cos(2 * math.pi * float(r["hour"] or 0) / 24),
                        math.sin(2 * math.pi * float(r["day_of_week"] or 0) / 7), math.cos(2 * math.pi * float(r["day_of_week"] or 0) / 7),
                        float(r["cpu"] or 0), float(r["ram"] or 0),
                        (float(r["cpu"] or 0) * float(r["ram"] or 0)) / 10000,
                        hash(r["url_domain"] or "") % 100,
                        math.sqrt(max(0, float(r["latency"] or 0))),
                        math.log1p(max(0, float(r["latency"] or 0))),
                        0.5
                    ]
                    X.append(features)
                    Y.append([
                        float(r["target_num_conn"] or 8), 
                        math.log(max(1, float(r["target_buffer"] or 1024))), 
                        float(r["target_delay"] or 0.1), 
                        float(r["target_prefetch"] or 4)
                    ])
                except Exception:
                    continue

            if len(X) < 50: return

            X, Y = np.array(X), np.array(Y)
            
            # IMPROVEMENT: Cargar Scalers existentes para usar partial_fit y no resetear la media/varianza (que invalidaría pesos)
            scaler_x = joblib.load(SCALER_X_PATH) if os.path.exists(SCALER_X_PATH) else StandardScaler()
            scaler_y = joblib.load(SCALER_Y_PATH) if os.path.exists(SCALER_Y_PATH) else StandardScaler()
            
            if not hasattr(scaler_x, 'mean_'):
                X_scaled = scaler_x.fit_transform(X)
                Y_scaled = scaler_y.fit_transform(Y)
            else:
                scaler_x.partial_fit(X)
                scaler_y.partial_fit(Y)
                X_scaled = scaler_x.transform(X)
                Y_scaled = scaler_y.transform(Y)

            # Cargar Modelo con warm_start (continuar donde dejamos)
            model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else MLPRegressor(
                hidden_layer_sizes=(32, 24, 16, 8), activation='relu', solver='adam', max_iter=800, warm_start=True)
            
            # Se fuerza el parametro warm_start siempre a True para no perder conocimiento
            model.warm_start = True
            model.fit(X_scaled, Y_scaled)
            
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler_x, SCALER_X_PATH)
            joblib.dump(scaler_y, SCALER_Y_PATH)
            
            if not turbo:
                print(f"  {C.GREEN}✅ Modelo reentrenado. Muestras: {len(X):,}. Loss: {model.loss_:.6f}{C.RESET}\n")
            return model.loss_
        except Exception as e:
            if not turbo:
                print(f"  {C.RED}❌ Error en entrenamiento: {e}{C.RESET}\n")
            return None

# ═══════════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════
def run_simulator(turbo=False):
    if not turbo:
        print(f"\n{C.CYAN}{C.BOLD}{'═'*75}{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}  🎓 AI TEACHER-STUDENT v3.0 — SIMULADOR DINÁMICO DE EXÁMENES{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}{'═'*75}{C.RESET}")
        print(f"  {C.DIM}Generando escenarios extremos aleatorios para evitar overfitting.{C.RESET}\n")
    else:
        print(f"\n{C.YELLOW}{C.BOLD}🚀 TURBO MODE ACTIVADO. Entrenando a máxima velocidad sin pausas visuales.{C.RESET}\n")

    student = AIStudent()
    gold_db = GoldDB()
    failed_count = 0
    total_reprobados = 0
    exam_id = 1
    last_loss = 0.0

    try:
        while True:
            # 1. Generar Examen
            rec = ScenarioGenerator.generate()
            mode_color = C.RED if rec['mode'] != 'normal' else C.GREEN
            
            if not turbo:
                print(f"{C.BLUE}▶ [EXAMEN #{exam_id:<5}]{C.RESET} {mode_color}[{rec['mode'].upper()}]{C.RESET} "
                      f"Speed: {rec['actual_speed']:>5.0f} KB/s | Lat: {rec['latency']:>4.0f} ms | CPU: {rec['cpu']:>2.0f}%")

            # 2. El Estudiante Responde
            pred_student = student.predict(rec)
            if not turbo:
                print(f"  {C.DIM}├─ Estudiante : conn={pred_student['c']:>2} | buf={pred_student['b']:>4} | delay={pred_student['d']:>4.2f}{C.RESET}")

            # 3. El Maestro Calcula la Verdad
            pred_teacher = RuleEngine.get_perfect_targets(rec)
            source = "RuleEngine"

            # Fallback a Ollama si el caso es extremadamente ambiguo o queremos inyectar IA
            if rec["mode"] == "system_stress" and random.random() < 0.2:
                ollama_pred = OllamaTeacher.ask(rec, pred_student)
                if ollama_pred:
                    pred_teacher = ollama_pred
                    source = f"Ollama ({OLLAMA_MODEL})"

            if not turbo:
                print(f"  {C.DIM}├─ Maestro    : conn={pred_teacher['c']:>2} | buf={pred_teacher['b']:>4} | delay={pred_teacher['d']:>4.2f} [{source}]{C.RESET}")

            # 4. Calificar
            passed, errors, error_magnitude = ExamEvaluator.grade(rec, pred_student, pred_teacher)

            if passed:
                if not turbo:
                    print(f"  {C.GREEN}└─ ✅ APROBADO{C.RESET}")
                    time.sleep(0.3)
            else:
                failed_count += 1
                total_reprobados += 1
                if not turbo:
                    err_str = " | ".join(errors)
                    print(f"  {C.RED}└─ ❌ REPROBADO: {err_str} -> Guardando Lección ({failed_count}/{BATCH_TO_TRAIN}){C.RESET}")
                
                gold_db.save_lesson(rec, pred_student, pred_teacher, source, error_magnitude)
                if not turbo: time.sleep(0.6)

            # 5. Reentrenar si acumulamos suficientes reprobados
            if failed_count >= BATCH_TO_TRAIN:
                loss = StudentTrainer.train(turbo=turbo)
                if loss is not None: last_loss = loss
                student.load_brain() # Recargar cerebro fresco
                failed_count = 0

            # 6. Resumen Estadístico Compacto en Turbo Mode
            if turbo and exam_id % 1000 == 0:
                print(f"⚡ Exámenes: {exam_id} | Reprobados: {total_reprobados} | Último Loss: {last_loss:.6f}")

            exam_id += 1
            if not turbo: time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}⚠️ Simulador detenido por el usuario.{C.RESET}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Teacher-Student v3.0")
    parser.add_argument("--turbo", action="store_true", help="Activa el modo turbo para máxima velocidad sin prints.")
    args = parser.parse_args()
    
    run_simulator(turbo=args.turbo)
