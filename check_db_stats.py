import sqlite3, os, sys, io
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except: pass

db = os.path.join(os.path.dirname(__file__), "cache", "iptv_permanent_memory.db")
conn = sqlite3.connect(db)

print("=" * 60)
print("1. DISTRIBUCIÓN ÉXITO / FALLO")
print("=" * 60)
for row in conn.execute("SELECT success, COUNT(*) as total FROM experiences GROUP BY success").fetchall():
    label = "ÉXITO" if row[0] == 1 else "FALLO"
    print(f"  success={row[0]} ({label}): {row[1]:,}")

print()
print("=" * 60)
print("2. ÚLTIMAS 10 EXPERIENCIAS DE ai_teacher (url_domain='teacher.local')")
print("=" * 60)
rows = conn.execute("""
    SELECT cpu, ram, latency, actual_speed, size_mb,
           target_num_conn, target_buffer, target_delay,
           success, url_domain
    FROM experiences
    WHERE url_domain = 'teacher.local'
    ORDER BY id DESC LIMIT 10
""").fetchall()
if rows:
    print(f"  {'CPU':>5} {'RAM':>5} {'LAT':>7} {'SPD':>9} {'MB':>6} {'CONN':>5} {'BUF':>5} {'DLY':>5} {'OK':>3}  DOMAIN")
    print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*3}  {'------'}")
    for r in rows:
        print(f"  {r[0]:>5.1f} {r[1]:>5.1f} {r[2]:>7.1f} {r[3]:>9.1f} {r[4]:>6.2f} {r[5]:>5} {r[6]:>5} {r[7]:>5.3f} {r[8]:>3}  {r[9]}")
else:
    print("  ⚠️  Sin filas con url_domain='teacher.local'")
    print("  Dominios distintos disponibles:")
    for d in conn.execute("SELECT DISTINCT url_domain FROM experiences LIMIT 15").fetchall():
        print(f"    - {d[0]}")

print()
print("=" * 60)
print("3. RANGO DE VALORES REALES")
print("=" * 60)
r = conn.execute("""
    SELECT 
        MIN(actual_speed) as spd_min,
        MAX(actual_speed) as spd_max,
        AVG(actual_speed) as spd_avg,
        MIN(latency)      as lat_min,
        MAX(latency)      as lat_max,
        AVG(latency)      as lat_avg
    FROM experiences
""").fetchone()
print(f"  actual_speed → min={r[0]:,.1f}  max={r[1]:,.1f}  avg={r[2]:,.1f}  KB/s")
print(f"  latency      → min={r[3]:,.1f}  max={r[4]:,.1f}  avg={r[5]:,.1f}  ms")

conn.close()
