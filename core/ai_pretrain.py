#!/usr/bin/env python3
"""
ai_pretrain.py - Pre-Entrenamiento Masivo Avanzado v2.0
════════════════════════════════════════════════════════════════════════
Genera 30.000+ experiencias usando modelos matemáticos avanzados:

  📐 FOURIER    — Patrones de tráfico de red modelados como series de Fourier
  📊 LOG-NORMAL — Distribución real de tamaños de archivo y velocidades
  🌊 WEIBULL    — Tiempos entre fallos de red (distribución estándar en telecoms)
  📡 POISSON    — Llegada de eventos de error (procesos de cola M/M/1)
  🔢 COMPLEJO   — Fasor de red: magnitud=calidad, fase=fiabilidad
  📉 PARETO     — Distribución de popularidad de canales (Ley de Zipf)
  🧮 ENTROPÍA   — Shannon entropy para calcular buffer óptimo
  🌐 MARKOV     — Transiciones de estado de red para simular degradación gradual

Cantidad: 30.000 experiencias (≈1000% superior al entrenamiento inicial de 2700)
"""

import sys, io
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except: pass

import os, time, math, random, sqlite3, cmath
import numpy as np
from collections import deque

print("=" * 70)
print("  🧠 PRE-ENTRENAMIENTO MASIVO v2.0 — Matemáticas Avanzadas")
print("=" * 70)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
DB_PATH   = os.path.join(CACHE_DIR, "iptv_permanent_memory.db")
os.makedirs(CACHE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════
# A. MODELADO DE RED CON NÚMEROS COMPLEJOS (FASOR)
# ══════════════════════════════════════════════════

def network_phasor(bandwidth_kbps: float, latency_ms: float) -> complex:
    """
    Representa el estado de la red como un número complejo (fasor):
      - Parte real    = capacidad de ancho de banda normalizada
      - Parte imaginaria = latencia inversa (mayor latencia → menor Im)
    
    |z| = magnitud de calidad total de la red
    ∠z  = ángulo de confiabilidad (1er cuadrante = bueno)
    
    z = (bw / BW_MAX) + j*(1 / (1 + lat/LAT_REF))
    """
    BW_MAX  = 50000.0   # 50 Mbps máximo de referencia
    LAT_REF = 100.0     # 100ms = latencia de referencia neutra
    
    real_part = min(bandwidth_kbps / BW_MAX, 1.0)
    imag_part = 1.0 / (1.0 + latency_ms / LAT_REF)
    return complex(real_part, imag_part)


def phasor_quality_score(z: complex) -> float:
    """Magnitud del fasor [0, √2] — 1 = red perfecta en ambas dimensiones."""
    return abs(z)


def phasor_reliability_angle(z: complex) -> float:
    """
    Ángulo de fase del fasor en radianes.
    π/4 (45°) = balance perfecto entre velocidad y latencia.
    Desviación del π/4 = degradación en alguna dimensión.
    """
    return abs(cmath.phase(z) - math.pi / 4)


# ══════════════════════════════════════════════════
# B. SERIE DE FOURIER PARA TRÁFICO DE RED (24h)
# ══════════════════════════════════════════════════

# Coeficientes de Fourier ajustados empíricamente al tráfico IPTV real:
#   Componente DC (media)
#   1° armónico: ciclo diario (pico noche)
#   2° armónico: pico mediodía secundario (media tarde)
#   3° armónico: variación semanal día/noche
FOURIER_A = [0.55, 0.30, 0.10, 0.05]   # cosenos (amplitudes)
FOURIER_B = [0.00, 0.25, 0.08, 0.03]   # senos   (fases)
FOURIER_NOISE_STD = 0.06                 # Ruido gaussiano sobre la señal

def fourier_traffic_load(hour: float, day: int) -> float:
    """
    Calcula la carga de tráfico de red en [0, 1] usando serie de Fourier.
    
    f(t) = A₀ + Σₙ [Aₙ·cos(2πnt/T) + Bₙ·sin(2πnt/T)] + ε
    
    Donde T=24h, n=1,2,3 armónicos,  ε ~ N(0, σ²)
    """
    T = 24.0
    t = hour
    
    result = FOURIER_A[0]  # DC offset
    for n in range(1, len(FOURIER_A)):
        omega = 2 * math.pi * n / T
        result += FOURIER_A[n] * math.cos(omega * t)
        result += FOURIER_B[n] * math.sin(omega * t)
    
    # Ajuste semanal: fin de semana +20% de tráfico en horas pico
    if day >= 5:  # sábado, domingo
        result *= 1.20
    
    # Ruido gaussiano realista
    result += random.gauss(0, FOURIER_NOISE_STD)
    return max(0.05, min(1.0, result))


# ══════════════════════════════════════════════════
# C. DISTRIBUCIONES AVANZADAS
# ══════════════════════════════════════════════════

def lognormal_speed(mean_kbps: float, cv: float = 0.4) -> float:
    """
    Velocidad real sigue distribución log-normal (comprobado empíricamente en CDNs).
    cv = coeficiente de variación (std/mean). 0.4 es típico en redes residenciales.
    
    Si X ~ LN(μ, σ²):  E[X] = e^(μ+σ²/2),  σ² = log(1 + cv²)
    """
    sigma2 = math.log(1 + cv**2)
    mu     = math.log(mean_kbps) - sigma2 / 2
    return np.random.lognormal(mu, math.sqrt(sigma2))


def weibull_latency(lambda_ms: float, k: float = 1.5) -> float:
    """
    Latencia distribuida Weibull — estándar en telecomunicaciones para
    modelar colas y tiempos de transferencia.
    
    k < 1: Decreasing Failure Rate (infraestructura nueva)
    k = 1: Exponencial (red sin degradación)
    k > 1: Increasing Failure Rate (red congestionada) → k=1.5 típico IPTV
    
    F(x) = 1 - exp(-(x/λ)^k)
    """
    # numpy weibull genera con λ=1; escalamos por lambda_ms
    return lambda_ms * np.random.weibull(k)


def poisson_error_rate(base_rate: float, load: float) -> float:
    """
    Tasa de errores sigue proceso de Poisson cuya intensidad λ crece
    con la carga de red (modelo M/M/1 simplificado):
    
    λ_efectivo = λ₀ / (1 - ρ)  donde ρ = load ∈ [0,1)
    
    Retorna probabilidad de error ∈ [0,1].
    """
    rho = min(load, 0.95)  # Factor de utilización
    lambda_eff = base_rate / (1.0 - rho)
    # P(error) = 1 - e^(-λ_eff) (cdf Poisson para k=0 no error → invertido)
    return min(1.0, 1.0 - math.exp(-lambda_eff))


def zipf_channel_popularity(n_channels: int, rank: int, alpha: float = 1.2) -> float:
    """
    Ley de Zipf/Pareto para popularidad de canales:
    El canal de rango r tiene popularidad ∝ 1/r^α
    Canal #1 es el más popular (más cargado en el servidor).
    
    Retorna factor de carga adicional [1.0, 5.0].
    """
    normalization = sum(1.0 / (i**alpha) for i in range(1, n_channels+1))
    pop = (1.0 / (rank**alpha)) / normalization
    # Convertir popularidad a factor de carga del servidor [1, 5]
    return 1.0 + 4.0 * (pop * n_channels)


def shannon_buffer_entropy(packet_loss: float, jitter_ms: float, bandwidth_kbps: float) -> float:
    """
    Buffer óptimo basado en Entropía de Shannon de la canal de comunicación.
    
    H = -p·log₂(p) - (1-p)·log₂(1-p)   donde p = packet_loss
    
    Más entropía (canal más impredecible) → buffer mayor necesario.
    Capacidad de Shannon: C = B·log₂(1 + SNR)
    """
    p = max(1e-10, min(1 - 1e-10, packet_loss))
    entropy = -(p * math.log2(p) + (1-p) * math.log2(1-p))  # [0, 1]
    
    # SNR aproximado desde jitter (menor jitter = mejor SNR)
    snr = max(0.1, bandwidth_kbps / (1 + jitter_ms * 10))
    capacity = math.log2(1 + snr)
    
    # Buffer óptimo: proporcional a entropía e inversamente a la capacidad
    buffer_kb = 256 + 4096 * (entropy / max(0.1, math.log2(capacity + 1)))
    return float(np.clip(buffer_kb, 128, 8192))


def markov_network_state(prev_state: str, load: float) -> str:
    """
    Cadena de Markov de 3 estados para simular degradación gradual de red:
    
    Estados: GOOD → DEGRADED → CRITICAL → GOOD
    
    Matriz de transición (depende de la carga actual):
    
         GOOD    DEGRAD  CRITI
    GOOD [ 1-α    α       0   ]
    DEG  [ β      1-β-γ   γ   ]
    CRIT [ 0      δ       1-δ  ]
    
    α, β, γ, δ dependen de la carga de tráfico.
    """
    alpha = 0.05 * load           # GOOD→DEGRADED (más carga = más probable)
    beta  = 0.30 * (1 - load)     # DEGRADED→GOOD (menos carga = más probable)
    gamma = 0.15 * load           # DEGRADED→CRITICAL
    delta = 0.40 * (1 - load)     # CRITICAL→DEGRADED

    r = random.random()
    if prev_state == "GOOD":
        return "DEGRADED" if r < alpha else "GOOD"
    elif prev_state == "DEGRADED":
        if r < beta:        return "GOOD"
        elif r < beta+gamma: return "CRITICAL"
        else:               return "DEGRADED"
    else:  # CRITICAL
        return "DEGRADED" if r < delta else "CRITICAL"


# ══════════════════════════════════════════════════
# D. CÁLCULO DE TARGETS ÓPTIMOS (MATEMÁTICA DERIVADA)
# ══════════════════════════════════════════════════

def compute_optimal_targets(
    z: complex,          # Fasor de red
    cpu: float,          # Carga CPU [0,100]
    ram: float,          # Uso RAM [0,100]
    packet_loss: float,  # Probabilidad de pérdida de paquetes [0,1]
    jitter_ms: float,    # Jitter en ms
    bandwidth_kbps: float,
    network_state: str,
) -> tuple:
    """
    Calcula configuración óptima usando:
    
    1. CONEXIONES:
       optimal_conn = round( K_conn · |z|² · log₂(bw/1000+1) · (1 - cpu/200) )
       
       Justificación: |z|² escala cuadráticamente con calidad de red (más canal
       disponible = más conexiones rentables). Log₂ del bandwidth limita el
       crecimiento (rendimientos decrecientes). Factor CPU reduce agresividad
       en sistemas cargados.
    
    2. BUFFER:
       Calculado con Shannon entropy (ver arriba): más errores → más buffer.
    
    3. RETRY DELAY:
       τ = τ_min · e^(λ·jitter)  (backoff exponencial con constante λ=0.01)
       Inspirado en TCP CUBIC: el delay crece exponencialmente con el jitter.
    
    4. PREFETCH:
       prefetch = ceil( T_startup / T_segment )
       T_startup = expected startup latency = latency / |z|
       T_segment = típico 2s para HLS
    """
    quality = phasor_quality_score(z)          # [0, √2]
    reliability_dev = phasor_reliability_angle(z)  # desviación de π/4
    
    # Factor de estrés del sistema (combinación CPU+RAM via media geométrica)
    stress = math.sqrt((cpu/100) * (ram/100))  # [0, 1]
    
    # ── 1. CONEXIONES ÓPTIMAS ──
    K_conn = 32.0  # Constante de escala máxima
    bw_log = math.log2(bandwidth_kbps / 1000 + 1)   # [0, log₂(51)] ≈ [0, 5.67]
    q2 = quality ** 2                                  # cuadrático en calidad
    conn_raw = K_conn * q2 * bw_log * (1 - stress/2)
    
    # Penalización por estado de red o pérdida de paquetes
    if network_state == "CRITICAL":
        conn_raw *= 1.8   # Necesita más conexiones para compensar
    elif network_state == "DEGRADED":
        conn_raw *= 1.3
    
    conn_raw *= (1 + 2 * packet_loss)  # Más pérdida → más conexiones redundantes
    optimal_conn = int(np.clip(round(conn_raw), 2, 48))
    
    # ── 2. BUFFER ÓPTIMO (Shannon entropy) ──
    optimal_buffer = int(shannon_buffer_entropy(packet_loss, jitter_ms, bandwidth_kbps))
    
    # Ajuste por estado Markov
    if network_state == "CRITICAL":
        optimal_buffer = min(8192, int(optimal_buffer * 2.0))
    elif network_state == "DEGRADED":
        optimal_buffer = min(8192, int(optimal_buffer * 1.4))
    
    # ── 3. RETRY DELAY (backoff exponencial TCP-inspired) ──
    tau_min = 0.02     # 20ms mínimo
    tau_max = 0.80     # 800ms máximo
    lambda_j = 0.008   # constante de crecimiento con jitter
    tau = tau_min * math.exp(lambda_j * jitter_ms)
    
    # Escalar con fiabilidad del fasor y estado Markov
    tau *= (1 + reliability_dev)  # más desviación del ideal → más delay
    if network_state == "CRITICAL":  tau *= 2.5
    elif network_state == "DEGRADED": tau *= 1.5
    optimal_delay = float(np.clip(tau, tau_min, tau_max))
    
    # ── 4. PREFETCH_COUNT ──
    # Número de segmentos a precargar = startup_time_estimado / 2s_por_segmento
    # startup_time ≈ latency_inherente / calidad_del_fasor
    imag_part = z.imag  # Componente de latencia del fasor
    startup_est = (1.0 / max(0.1, imag_part)) * (1 + packet_loss * 3)
    prefetch_raw = math.ceil(startup_est / 2.0)
    if network_state == "CRITICAL":  prefetch_raw += 5
    elif network_state == "DEGRADED": prefetch_raw += 2
    optimal_prefetch = int(np.clip(prefetch_raw, 1, 15))
    
    return optimal_conn, optimal_buffer, round(optimal_delay, 3), optimal_prefetch


# ══════════════════════════════════════════════════
# E. DOMINIOS Y CANALES
# ══════════════════════════════════════════════════

DOMAINS = [
    # (nombre, bw_base_kbps, latency_lambda_ms, reliability)
    ("cdn.iptv-pro.com",       45000, 15,  0.98),
    ("stream.hdtv-plus.net",   38000, 20,  0.96),
    ("edge.ultrahdtv.net",     50000, 12,  0.99),
    ("live.tvcenter.io",       30000, 25,  0.95),
    ("srv1.megastream.tv",     35000, 18,  0.97),
    ("free-iptv.org",           8000, 180, 0.72),
    ("iptv-community.net",      6000, 220, 0.68),
    ("openstreams.xyz",         5000, 280, 0.61),
    ("public.streamhub.cc",     4000, 350, 0.55),
    ("p2p.iptv-share.net",     12000, 120, 0.78),
    ("backup.tvserver.org",    15000, 90,  0.82),
    ("iptv-latam.tv",          20000, 60,  0.88),
]

CHANNELS = [
    ("CNN en Español",    "live",  0.5,   3),
    ("ESPN HD",           "live",  1.2,   2),
    ("HBO Max Live",      "live",  2.5,   1),
    ("Fox Sports 4K",     "live",  4.0,   1),
    ("Pelicula 4K UHD",   "vod",   8000,  5),
    ("Serie Netflix HD",  "vod",   900,   4),
    ("Documental HD",     "vod",   600,   7),
    ("Deportes SD",       "live",  0.4,   6),
    ("Kids Channel",      "live",  0.6,   8),
    ("Noticias 24h",      "live",  0.5,   3),
    ("PlayStation Vue",   "live",  1.8,   4),
    ("Amazon Prime 4K",   "vod",   12000, 2),
    ("Disney+ HD",        "vod",   2000,  3),
    ("Musica HD",         "live",  0.9,   9),
    ("Discovery 4K",      "live",  3.5,   5),
    ("Anime HD",          "live",  1.1,   6),
    ("Reality Show SD",   "live",  0.5,   7),
    ("Cine Clasico",      "vod",   400,   10),
    ("UFC PPV 4K",        "live",  5.0,   1),
    ("Telenovela HD",     "live",  1.0,   4),
]

INSERT_SQL = '''
    INSERT INTO experiences
    (timestamp, url_domain, full_url, channel_name, size_mb, hour, day_of_week,
     cpu, ram, latency, target_num_conn, target_buffer, target_delay, target_prefetch,
     actual_speed, success, is_critical)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
'''

# ══════════════════════════════════════════════════
# F. GENERACIÓN CON CADENA DE MARKOV
# ══════════════════════════════════════════════════

def generate_markov_sequence(n: int, start_hour: int = 0) -> list:
    """
    Genera una secuencia de n experiencias usando la cadena de Markov para
    simular sesiones reales donde la red se degrada/recupera gradualmente.
    """
    rows = []
    state      = "GOOD"
    base_time  = time.time() - random.randint(0, 86400 * 45)
    n_channels = len(CHANNELS)
    
    # Elegir dominio para esta sesión (una sesión = un servidor)
    dom_info = random.choice(DOMAINS)
    domain, bw_base, lat_lambda, reliability = dom_info
    
    for i in range(n):
        hour    = (start_hour + i // 20) % 24
        dow     = random.randint(0, 6)
        now     = base_time + i * random.uniform(30, 180)  # 30s-3min entre eventos
        
        # ── Carga de red (Fourier)
        load    = fourier_traffic_load(hour, dow)
        
        # ── Transición de estado Markov
        state = markov_network_state(state, load)
        
        # ── Velocidad log-normal (degradada según estado)
        state_bw_factor = {"GOOD": 1.0, "DEGRADED": 0.55, "CRITICAL": 0.20}[state]
        bandwidth_kbps  = lognormal_speed(bw_base * state_bw_factor, cv=0.35)
        
        # ── Latencia Weibull (peor en estado crítico)
        state_lat_factor = {"GOOD": 1.0, "DEGRADED": 2.5, "CRITICAL": 5.0}[state]
        latency_ms = weibull_latency(lat_lambda * state_lat_factor, k=1.5)
        
        # ── Jitter (proporcional a la carga y estado)
        jitter_ms  = load * 50 * state_lat_factor + random.gauss(0, 5)
        jitter_ms  = max(0, jitter_ms)
        
        # ── Tasa de errores Poisson
        base_err   = 1.0 - reliability
        packet_loss = poisson_error_rate(base_err * state_lat_factor, load)
        
        # ── CPU y RAM (correlacionados con carga via Fourier + ruido)
        cpu = np.clip(load * 60 + random.gauss(0, 10), 2, 98)
        ram = np.clip(load * 50 + 25 + random.gauss(0, 8), 15, 95)
        stress_ratio = (cpu * ram) / 10000
        
        # ── Canal aleatorio con popularidad Zipf
        ch_name, ch_type, size_base, ch_rank = random.choice(CHANNELS)
        zipf_factor = zipf_channel_popularity(n_channels, ch_rank)
        
        if ch_type == "live":
            size_mb = float(np.random.lognormal(math.log(size_base), 0.3))
        else:
            size_mb = float(np.random.lognormal(math.log(size_base + 1), 0.5))
        
        # ── Fasor de red
        z = network_phasor(bandwidth_kbps, latency_ms)
        
        # ── Targets óptimos (matemáticamente derivados)
        t_conn, t_buf, t_delay, t_prefetch = compute_optimal_targets(
            z, cpu, ram, packet_loss, jitter_ms, bandwidth_kbps, state
        )
        
        # ── Éxito: determinístico por estado Markov + probabilidad
        if state == "GOOD":
            success = 1 if random.random() < reliability else 0
        elif state == "DEGRADED":
            success = 1 if random.random() < reliability * 0.6 else 0
        else:  # CRITICAL
            success = 1 if random.random() < reliability * 0.2 else 0
        
        is_critical = 1 if (not success or bandwidth_kbps < 1000) else 0
        
        full_url = f"http://{domain}/{'live' if ch_type=='live' else 'vod'}/{random.randint(10000,99999)}.{'m3u8' if ch_type=='live' else 'ts'}"
        
        rows.append((
            now, domain, full_url, ch_name,
            round(size_mb, 3), hour, dow,
            round(float(cpu), 1), round(float(ram), 1),
            round(float(latency_ms), 1),
            t_conn, t_buf, t_delay, t_prefetch,
            round(float(bandwidth_kbps), 1),
            success, is_critical
        ))
    
    return rows


# ══════════════════════════════════════════════════
# G. SETUP BD Y LIMPIEZA
# ══════════════════════════════════════════════════

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL, url_domain TEXT, full_url TEXT, channel_name TEXT,
        size_mb REAL, hour INTEGER, day_of_week INTEGER, cpu REAL, ram REAL,
        latency REAL, target_num_conn INTEGER, target_buffer INTEGER,
        target_delay REAL, target_prefetch INTEGER, actual_speed REAL,
        success INTEGER, is_critical INTEGER
    )
''')
for col, ctype in [('channel_name','TEXT'), ('full_url','TEXT')]:
    try: cursor.execute(f"ALTER TABLE experiences ADD COLUMN {col} {ctype}")
    except: pass
conn.commit()

existing = cursor.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
print(f"\n  📊 Experiencias previas en BD: {existing:,}")
if existing > 5000:
    print(f"  ⚠️  BD tiene {existing:,} experiencias (posiblemente de Kaggle). NO se borra.")
    print(f"  ℹ️  Para forzar reset, borra manualmente el archivo: {DB_PATH}")
else:
    print(f"  🗑️  Limpiando datos anteriores para re-entrenamiento limpio...")
    cursor.execute("DELETE FROM experiences")
    conn.commit()
    print(f"  ✅ BD limpia. Generando experiencias sintéticas...")

# ══════════════════════════════════════════════════
# H. GENERACIÓN POR LOTES
# ══════════════════════════════════════════════════

BATCHES = [
    ("📡 Sesiones diurnas (8-18h) - Oficinas y telecos",      25000, 8),
    ("🌙 Sesiones nocturnas (20-23h) - Pico IPTV residencial",25000, 20),
    ("🌅 Sesiones madrugada (0-6h) - Tráfico mínimo",         15000, 0),
    ("💥 Eventos críticos - Fallos en cascada (Markov)",       17500, 20),
    ("🏆 Eventos premium - ESPN/PPV pico simultáneo",          15000, 21),
    ("📺 VOD masivo - Netflix/Prime fin de semana",            17500, 19),
    ("🌐 Servidores internacionales - Latencias altas",        15000, 14),
    ("⚡ Recuperación post-fallo - Degraded→Good (Markov)",   12500, 3),
    ("🎮 Competencia de red - Gaming + IPTV simultáneo",       12500, 19),
]

total_inserted = 0
t0_global = time.time()

for desc, count, start_hour in BATCHES:
    t0 = time.time()
    rows = generate_markov_sequence(count, start_hour=start_hour)
    cursor.executemany(INSERT_SQL, rows)
    conn.commit()
    elapsed = time.time() - t0
    ok   = sum(1 for r in rows if r[15] == 1)   # índice 15 = success
    crit = sum(1 for r in rows if r[16] == 1)   # índice 16 = is_critical
    total_inserted += count
    print(f"  ✅ {desc[:55]:55s}  +{count:5,}  OK:{ok:4d}  CRIT:{crit:4d}  ({elapsed:.1f}s)")

conn.close()

# ══════════════════════════════════════════════════
# I. ESTADÍSTICAS FINALES Y ENTRENAMIENTO
# ══════════════════════════════════════════════════
conn2 = sqlite3.connect(DB_PATH)
total_bd    = conn2.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
avg_spd     = conn2.execute("SELECT AVG(actual_speed) FROM experiences").fetchone()[0] or 0
avg_lat     = conn2.execute("SELECT AVG(latency) FROM experiences").fetchone()[0] or 0
pct_success = conn2.execute("SELECT AVG(success)*100 FROM experiences").fetchone()[0] or 0
pct_crit    = conn2.execute("SELECT AVG(is_critical)*100 FROM experiences").fetchone()[0] or 0
domains_bd  = conn2.execute("SELECT COUNT(DISTINCT url_domain) FROM experiences").fetchone()[0]
conn2.close()

print(f"\n{'=' * 70}")
print(f"  📊 ESTADÍSTICAS DE LA BASE DE DATOS")
print(f"{'=' * 70}")
print(f"  Total experiencias : {total_bd:,}")
print(f"  Velocidad promedio : {avg_spd:,.0f} KB/s")
print(f"  Latencia promedio  : {avg_lat:.1f} ms")
print(f"  Tasa de éxito      : {pct_success:.1f}%")
print(f"  Experiencias críti.: {pct_crit:.1f}%")
print(f"  Dominios distintos : {domains_bd}")

print(f"\n{'=' * 70}")
print(f"  🔥 ENTRENANDO RED NEURONAL con {total_bd:,} experiencias...")
print(f"{'=' * 70}\n")

try:
    from iptv_ai_core import ai_optimizer
    # Resetear modelo para reentrenar con arquitectura nueva y más features
    import os
    # Solo resetear si no existen pkl de Kaggle o son inválidos
    pkl_ok = all(
        os.path.exists(os.path.join(CACHE_DIR, f))
        for f in ["iptv_dynamic_brain.pkl", "iptv_scaler_x_v4.pkl", "iptv_scaler_y_v4.pkl"]
    )
    if not pkl_ok:
        print("  🔄 No se encontraron pkl de Kaggle. Reseteando para entrenar desde cero...")
        for f in ["iptv_dynamic_brain.pkl", "iptv_scaler_x_v4.pkl", "iptv_scaler_y_v4.pkl", "brain_config.json"]:
            path = os.path.join(CACHE_DIR, f)
            if os.path.exists(path): os.remove(path)
        ai_optimizer.model = None
    else:
        print("  ✅ PKL de Kaggle detectados. Se usarán como base para evolve_brain().")
    ai_optimizer.experience_count = total_bd
    
    ai_optimizer.evolve_brain()
    
    cfg = ai_optimizer.config
    model_ok = ai_optimizer.model is not None
    
    print(f"\n{'=' * 70}")
    print(f"  🎉 ENTRENAMIENTO COMPLETADO")
    print(f"{'=' * 70}")
    print(f"  Modelo activo     : {'✅ SÍ' if model_ok else '❌ No'}")
    if model_ok:
        print(f"  Error de la red   : {cfg.get('avg_error', 0):.6f}")
        print(f"  Arquitectura      : {cfg.get('layers', [])}")
        print(f"  Complejidad       : Nivel {cfg.get('complexity_level', 1)}")
        print(f"  Entrenamientos    : {cfg.get('total_trainings', 0)}")
        
        print(f"\n  🧪 Predicciones con matemáticas complejas:")
        tests = [
            ("http://cdn.iptv-pro.com/live/1.m3u8",    1.2,   15,  "ESPN HD - servidor CDN rápido"),
            ("http://free-iptv.org/vod/big.ts",         8000,  280, "VOD pesado - servidor lento"),
            ("http://edge.ultrahdtv.net/live/4k.m3u8",  4.0,   12,  "Canal 4K - servidor premium"),
            ("http://p2p.iptv-share.net/live/sd.ts",    0.5,   120, "Canal SD - red P2P"),
        ]
        for url, size, lat, label in tests:
            pred = ai_optimizer.predict_optimal_config(url, size, lat)
            print(f"\n    [{label}]")
            print(f"      conex={pred['num_conn']:>3} | buffer={pred['buffer_size']:>5}KB | delay={pred['retry_delay']:.3f}s | prefetch={pred['prefetch_count']}")

except Exception as e:
    import traceback
    print(f"\n  ❌ Error en entrenamiento:")
    traceback.print_exc()

total_elapsed = time.time() - t0_global
print(f"\n  ⏱️  Tiempo total: {total_elapsed:.1f}s")
print(f"  📁 Archivos del cerebro IA:")
for f in sorted(os.listdir(CACHE_DIR)):
    kb = os.path.getsize(os.path.join(CACHE_DIR, f)) / 1024
    print(f"     {f:45s}  {kb:8.1f} KB")
print()
