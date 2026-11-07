#!/usr/bin/env python3
"""Test de validacion v6.1 — verifica que evolve_brain() produce labels continuas
y que las predicciones tienen varianza real (no son constantes)."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

print("=== Cargando modulo iptv_ai_core v6.1 ===")
from iptv_ai_core import ai_optimizer

print(f"Modelo cargado: {ai_optimizer.model}")
print(f"Experience count: {ai_optimizer.experience_count:,}")
print()
print("=== Ejecutando evolve_brain() v6.1 ===")
ai_optimizer.evolve_brain()
print()

print("=== Verificacion de varianza de predicciones ===")
test_scenarios = [
    ("http://flexlive.lat:8080/live/1.ts",  1.2,   10,   5000,  2500,  "RED BUENA, madrugada"),
    ("http://flexlive.lat:8080/live/2.ts",  5.0,  400,   1000,  9600,  "ALTA LATENCIA 400ms"),
    ("http://flexlive.lat:8080/live/3.ts", 15.0,  100,  20000,  9600,  "VOD pesado rapido"),
    ("http://flexlive.lat:8080/live/4.ts",  0.5,   50,    200,  2500,  "LENTO asfixiado"),
    ("http://flexlive.lat:8080/live/5.ts",  2.0,  800,    500,  5000,  "LATENCIA SATELITAL"),
]

results = []
for url, size, lat, speed, brate, label in test_scenarios:
    pred = ai_optimizer.predict_optimal_config(url, size, lat, speed, brate)
    results.append(pred)
    print(f"  [{label}]")
    print(f"    conn={pred['num_conn']} | buf={pred['buffer_kb']}KB | delay={pred['retry_delay']}s"
          f" | stall={pred['stall_prob']:.3f} | preset={pred['preset']}")

conn_vals  = [r["num_conn"]   for r in results]
buf_vals   = [r["buffer_kb"]  for r in results]
stall_vals = [r["stall_prob"] for r in results]
print()
print(f"  Varianza conn:   min={min(conn_vals)} max={max(conn_vals)} range={max(conn_vals)-min(conn_vals)}")
print(f"  Varianza buffer: min={min(buf_vals)} max={max(buf_vals)} range={max(buf_vals)-min(buf_vals)}")
print(f"  Varianza stall:  min={min(stall_vals):.3f} max={max(stall_vals):.3f} range={max(stall_vals)-min(stall_vals):.3f}")
print()

print("=== brain_config.json actualizado ===")
import json
with open(os.path.join(os.path.dirname(__file__), '..', 'cache', 'brain_config.json')) as f:
    print(json.dumps(json.load(f), indent=2))

print()
print("=== Muestra de labels continuas generadas (5 escenarios) ===")
from iptv_ai_core import IPTVEvolutionaryBrain
import math, numpy as np

sample_cases = [
    (1.0,  5, 40,   10,  25000, 12, True),
    (5.0, 70, 80,  400,   1000, 20, False),
    (0.5, 20, 50,   50,  15000,  8, True),
    (8.0, 90, 85,  100,    500,  2, False),
    (3.0, 50, 60,  200,   5000, 14, True),
]
for i, (sz, cpu, ram, lat, spd, hr, suc) in enumerate(sample_cases):
    labels = IPTVEvolutionaryBrain._compute_continuous_labels(sz, cpu, ram, lat, spd, hr, suc)
    print(f"  Caso {i+1} (speed={spd:>6}KB/s lat={lat:>4}ms cpu={cpu}%):")
    print(f"    conn={labels[0]:.1f} buf={math.exp(labels[1]):.0f}KB"
          f" delay={labels[2]:.3f}s pref={labels[3]:.0f}"
          f" threads={labels[4]:.0f} preset={labels[5]:.0f}"
          f" reconn={labels[6]:.1f} stall={labels[7]:.3f}")
