#!/usr/bin/env python3
"""
===============================================================================
 Verificador de Listas IPTV M3U / Xtream Codes
===============================================================================
 Este script analiza un archivo .txt con URLs de listas IPTV, prueba su validez,
 mide la latencia de respuesta, verifica la presencia de canales válidos (#EXTM3U)
 y guarda únicamente las listas activas y funcionales en un archivo de salida.

 Uso:
    python check_iptv_playlists.py [archivo_entrada.txt] [archivo_salida.txt]

 Ejemplo:
    python check_iptv_playlists.py mis_listas.txt listas_validas.txt
===============================================================================
"""

import os
import sys
import re
import time
import urllib.request
import urllib.parse
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración por defecto
DEFAULT_INPUT_FILE = "listas.txt"
DEFAULT_OUTPUT_FILE = "listas_validas.txt"
DEFAULT_REPORT_FILE = "listas_reporte.txt"
NUM_WORKERS = 20
TIMEOUT_SECONDS = 6

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Contexto SSL flexible para permitir certificados autofirmados o caducados de servidores IPTV
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def extract_urls_from_file(file_path):
    """Extrae todas las URLs HTTP/HTTPS válidas de un archivo de texto"""
    urls = []
    if not os.path.exists(file_path):
        return urls

    url_pattern = re.compile(r'https?://[^\s\'"<>\(\)]+', re.IGNORECASE)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            found_urls = url_pattern.findall(line_str)
            for url in found_urls:
                # Limpiar caracteres sobrantes al final de la URL
                cleaned_url = url.rstrip(",;\"')]>")
                if cleaned_url and cleaned_url not in urls:
                    urls.append(cleaned_url)

    return urls


def test_iptv_url(url):
    """
    Comprueba si una URL de lista IPTV responde correctamente y contiene contenido M3U/Xtream válido.
    Retorna (is_valid, latency_ms, channel_count, status_message)
    """
    start_time = time.time()
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                "Connection": "keep-alive"
            }
        )

        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS, context=ssl_context) as resp:
            latency_ms = int((time.time() - start_time) * 1000)
            status_code = resp.getcode()

            if status_code not in (200, 206):
                return False, latency_ms, 0, f"HTTP {status_code}"

            # Leer los primeros 64KB para verificar la estructura M3U/playlist
            chunk = resp.read(65536)

            # Intentar decodificar el contenido
            try:
                content_text = chunk.decode("utf-8", errors="ignore")
            except Exception:
                content_text = chunk.decode("latin-1", errors="ignore")

            # Verificar si contiene etiquetas de listas IPTV
            is_m3u = "#EXTM3U" in content_text or "#EXTINF" in content_text
            channel_count = content_text.count("#EXTINF:")

            # Si es un servidor Xtream Codes que devuelve un stream o JSON directo
            is_xtream_json = "user_info" in content_text or "available_channels" in content_text

            if is_m3u or is_xtream_json or (len(chunk) > 500 and "http" in content_text):
                msg = f"OK ({channel_count}+ canales)" if channel_count > 0 else "OK (Válida)"
                return True, latency_ms, channel_count, msg
            else:
                return False, latency_ms, 0, "No es una lista M3U válida"

    except urllib.error.HTTPError as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return False, latency_ms, 0, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        latency_ms = int((time.time() - start_time) * 1000)
        reason = str(e.reason)
        if "timed out" in reason.lower():
            return False, latency_ms, 0, "Timeout"
        return False, latency_ms, 0, "Servidor inalcanzable"
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        err_name = type(e).__name__
        if "timeout" in str(e).lower():
            return False, latency_ms, 0, "Timeout"
        return False, latency_ms, 0, f"Error: {err_name}"


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FILE

    print("=" * 70)
    print("      VERIFICADOR DE LISTAS IPTV — AUDITORÍA DE SERVIDORES")
    print("=" * 70)

    if not os.path.exists(input_file):
        print(f" ERROR: No se encontró el archivo de entrada '{input_file}'.")
        print(f"\nUso:")
        print(f"  python check_iptv_playlists.py <archivo_listas.txt> [archivo_salida.txt]")
        print(f"\nCreando archivo de plantilla '{input_file}' con ejemplo...")
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("# Pega aquí tus URLs de listas IPTV (una por línea):\n")
            f.write("http://ejemplo-iptv.net:8080/get.php?username=test&password=test&type=m3u_plus\n")
        print(f"✓ Plantilla creada en '{input_file}'. Agrega tus URLs y vuelve a ejecutar.")
        return

    print(f" Leyendo URLs desde: {input_file}")
    urls = extract_urls_from_file(input_file)

    if not urls:
        print(" No se encontraron URLs válidas HTTP/HTTPS en el archivo.")
        return

    print(f" URLs encontradas para analizar: {len(urls)}")
    print(f" Hilos concurrentes: {NUM_WORKERS} | Timeout por servidor: {TIMEOUT_SECONDS}s")
    print("-" * 70)

    valid_results = []
    invalid_results = []
    completed_count = 0
    total_urls = len(urls)

    start_audit_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_url = {executor.submit(test_iptv_url, url): url for url in urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            completed_count += 1
            try:
                is_valid, latency_ms, channel_count, msg = future.result()
            except Exception as ex:
                is_valid, latency_ms, channel_count, msg = False, 0, 0, f"Error interno: {ex}"

            status_symbol = "[✓ VÁLIDA ]" if is_valid else "[✗ INVÁLIDA]"
            domain = urllib.parse.urlparse(url).netloc or url[:30]

            print(f"[{completed_count:3d}/{total_urls:3d}] {status_symbol} {domain:<35} | {latency_ms:4d}ms | {msg}")

            result_entry = {
                "url": url,
                "domain": domain,
                "latency_ms": latency_ms,
                "channels": channel_count,
                "msg": msg
            }

            if is_valid:
                valid_results.append(result_entry)
            else:
                invalid_results.append(result_entry)

    total_duration = time.time() - start_audit_time

    # Ordenar válidas por latencia (más rápidas primero)
    valid_results.sort(key=lambda x: x["latency_ms"])

    # Guardar solo las URLs válidas en el archivo de salida
    with open(output_file, "w", encoding="utf-8") as f:
        for item in valid_results:
            f.write(f"{item['url']}\n")

    # Guardar reporte detallado
    with open(DEFAULT_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== REPORTE DE AUDITORÍA DE LISTAS IPTV ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")
        f.write(f"Total analizadas: {total_urls}\n")
        f.write(f"Válidas: {len(valid_results)}\n")
        f.write(f"Inválidas: {len(invalid_results)}\n")
        f.write(f"Tiempo de auditoría: {total_duration:.2f} segundos\n\n")
        f.write("--- LISTAS VÁLIDAS (ORDENADAS POR LATENCIA) ---\n")
        for item in valid_results:
            f.write(f"[{item['latency_ms']}ms] [{item['msg']}] {item['url']}\n")
        f.write("\n--- LISTAS INVÁLIDAS / CAÍDAS ---\n")
        for item in invalid_results:
            f.write(f"[{item['msg']}] {item['url']}\n")

    print("=" * 70)
    print("                     RESUMEN DE AUDITORÍA IPTV")
    print("=" * 70)
    print(f" Total listas analizadas: {total_urls}")
    print(f"  Listas VÁLIDAS activas: {len(valid_results)}")
    print(f" ✗ Listas INVÁLIDAS / caídas: {len(invalid_results)}")
    print(f" ⏱ Tiempo de escaneo: {total_duration:.2f}s")
    print("-" * 70)
    print(f" Listas válidas guardadas en: {os.path.abspath(output_file)}")
    print(f" Reporte detallado guardado en: {os.path.abspath(DEFAULT_REPORT_FILE)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
