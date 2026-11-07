#!/usr/bin/env python3
"""
async_vod_downloader.py — Motor VOD Asíncrono Paralelo de Alto Rendimiento (v7.3)
══════════════════════════════════════════════════════════════════════════════════
Características Clave:
  1. Soporte Nativo de Range Requests: permite seeking instantáneo en cualquier posición.
  2. Bounded Memory Buffer (Ventana Deslizante): streaming secuencial acotado a < 16MB RAM.
  3. Chunk Pipelining: descarga fragmentos concurrentes con aiohttp y connection pooling.
  4. Cancelación Inmediata: si el cliente desconecta o hace seek, cancela tareas sin zombies.
  5. Camuflaje Anti-Bot: integración con ThrottleDetector (User-Agent rotativo y Jitter).
"""

import asyncio
import threading
import queue
import time
import os
import sys
import aiohttp

# Configuración por defecto de streaming
DEFAULT_CHUNK_SIZE = 512 * 1024  # 512 KB por micro-trozo
MAX_BUFFER_CHUNKS  = 16          # Máximo 16 chunks en cola RAM (~8 MB)
SOCKET_WRITE_CHUNK = 64 * 1024   # 64 KB por write al socket


async def _fetch_chunk(session, url, chunk_idx, start_b, end_b, ai_headers_func, ai_jitter_func, stop_event):
    """Descarga un micro-chunk específico usando aiohttp con cabeceras y jitter."""
    if stop_event.is_set():
        return None

    headers = {}
    if ai_headers_func:
        try:
            headers = ai_headers_func(chunk_idx, url)
        except Exception:
            pass

    if not headers:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": url,
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "Connection": "keep-alive"
        }

    headers["Range"] = f"bytes={start_b}-{end_b}"

    if ai_jitter_func:
        try:
            delay = ai_jitter_func(chunk_idx)
            if delay > 0:
                await asyncio.sleep(delay)
        except Exception:
            pass

    timeout = aiohttp.ClientTimeout(total=45, connect=10, sock_read=30)
    for retry in range(3):
        if stop_event.is_set():
            return None
        try:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                if resp.status in (200, 206):
                    data = await resp.read()
                    return data
                elif resp.status in (404, 403, 410):
                    # Error no recuperable inmediatamente
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status, message=f"HTTP {resp.status}"
                    )
                else:
                    await asyncio.sleep(0.5 * (retry + 1))
        except (asyncio.CancelledError, Exception) as e:
            if isinstance(e, asyncio.CancelledError) or stop_event.is_set():
                return None
            if retry == 2:
                raise e
            await asyncio.sleep(0.4 * (retry + 1))
    return None


async def _async_producer(url, ranges, num_conn, chunk_queue, stop_event, loop, ai_headers_func, ai_jitter_func):
    """
    Produce los fragmentos en paralelo y los coloca ordenados en la cola acotada chunk_queue.
    Garantiza que la memoria RAM permanezca acotada limitando la ventana de prefetch.
    """
    connector = aiohttp.TCPConnector(
        limit=num_conn + 4,
        limit_per_host=num_conn + 2,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        ssl=False
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        total_chunks = len(ranges)
        next_chunk_to_queue = 0
        in_flight = {}  # {chunk_idx: task}
        completed_buffer = {}  # {chunk_idx: bytes}
        concurrency_limit = max(2, min(num_conn, 16))

        try:
            while (next_chunk_to_queue < total_chunks or in_flight or completed_buffer) and not stop_event.is_set():
                # 1. Despachar nuevas tareas hasta el límite de concurrencia
                while len(in_flight) < concurrency_limit and (next_chunk_to_queue + len(in_flight) + len(completed_buffer)) < total_chunks:
                    idx = next_chunk_to_queue + len(in_flight) + len(completed_buffer)
                    if idx >= total_chunks:
                        break
                    start_b, end_b = ranges[idx]
                    task = asyncio.create_task(
                        _fetch_chunk(session, url, idx, start_b, end_b, ai_headers_func, ai_jitter_func, stop_event)
                    )
                    in_flight[idx] = task

                if not in_flight and not completed_buffer:
                    break

                # 2. Esperar al menos una tarea completada
                if in_flight:
                    done, _ = await asyncio.wait(in_flight.values(), return_when=asyncio.FIRST_COMPLETED, timeout=1.0)
                    for task in done:
                        # Identificar el índice de la tarea terminada
                        done_idx = None
                        for idx, t in list(in_flight.items()):
                            if t is task:
                                done_idx = idx
                                del in_flight[idx]
                                break

                        if done_idx is not None and not stop_event.is_set():
                            try:
                                chunk_data = task.result()
                                if chunk_data is not None:
                                    completed_buffer[done_idx] = chunk_data
                            except Exception as err:
                                chunk_queue.put(err)
                                return

                # 3. Entregar los chunks estrictamente secuenciales a la cola del socket
                while next_chunk_to_queue in completed_buffer and not stop_event.is_set():
                    data = completed_buffer.pop(next_chunk_to_queue)
                    
                    # Colocar en chunk_queue bloqueando si la cola está llena (protección RAM)
                    while not stop_event.is_set():
                        try:
                            chunk_queue.put(data, timeout=0.2)
                            next_chunk_to_queue += 1
                            break
                        except queue.Full:
                            await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not stop_event.is_set():
                chunk_queue.put(e)
        finally:
            # Cancelar tareas pendientes al salir
            for t in in_flight.values():
                if not t.done():
                    t.cancel()
            if in_flight:
                await asyncio.gather(*in_flight.values(), return_exceptions=True)
            # Indicar fin de transmisión
            chunk_queue.put(None)


def download_and_stream(url, wfile, start_byte=0, end_byte=None, content_length=None,
                        num_conn=8, ai_headers_func=None, ai_jitter_func=None,
                        chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Coordina la descarga asíncrona paralela y el streaming síncrono al cliente (wfile).
    
    Parámetros:
      url: URL directa del video VOD.
      wfile: socket sincrónico de escritura HTTP del cliente.
      start_byte: inicio del rango solicitado (default: 0).
      end_byte: fin del rango solicitado (default: content_length - 1).
      content_length: tamaño total del archivo en bytes (de HEAD previo).
      num_conn: número de conexiones paralelas sugeridas por IA.
      chunk_size: tamaño en bytes de cada micro-trozo.
    """
    if content_length is None and end_byte is None:
        raise ValueError("Se requiere content_length o end_byte")

    if end_byte is None:
        end_byte = content_length - 1

    if start_byte > end_byte or start_byte < 0:
        return 0

    total_bytes = end_byte - start_byte + 1
    if total_bytes <= 0:
        return 0

    # Construir lista de rangos (chunks)
    ranges = []
    curr = start_byte
    while curr <= end_byte:
        next_b = min(curr + chunk_size - 1, end_byte)
        ranges.append((curr, next_b))
        curr = next_b + 1

    # Cola acotada para que la memoria RAM máxima no exceda MAX_BUFFER_CHUNKS * chunk_size (~8MB)
    chunk_queue = queue.Queue(maxsize=MAX_BUFFER_CHUNKS)
    stop_event = threading.Event()
    loop = asyncio.new_event_loop()

    def _loop_worker():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _async_producer(url, ranges, num_conn, chunk_queue, stop_event, loop, ai_headers_func, ai_jitter_func)
            )
        finally:
            try:
                # Cancelar tareas huérfanas
                pending = asyncio.all_tasks(loop)
                for p in pending:
                    p.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()
            except Exception:
                pass

    producer_thread = threading.Thread(target=_loop_worker, daemon=True, name="AsyncVODProducer")
    producer_thread.start()

    bytes_streamed = 0
    t0 = time.time()

    try:
        while True:
            try:
                item = chunk_queue.get(timeout=25.0)
            except queue.Empty:
                if not producer_thread.is_alive():
                    break
                continue

            if item is None:
                # Centinela de fin de stream
                chunk_queue.task_done()
                break

            if isinstance(item, Exception):
                chunk_queue.task_done()
                raise item

            # Escribir el fragmento al socket del cliente en bloques eficientes
            view = memoryview(item)
            offset = 0
            while offset < len(view):
                sub = view[offset:offset + SOCKET_WRITE_CHUNK]
                wfile.write(sub)
                offset += len(sub)
                bytes_streamed += len(sub)

            try:
                wfile.flush()
            except Exception:
                pass

            chunk_queue.task_done()

    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        # Desconexión normal del reproductor (por ejemplo, el usuario cerró o hizo seek)
        pass
    except Exception as e:
        raise e
    finally:
        # Señalizar al productor para que libere todos los recursos inmediatamente
        stop_event.set()
        # Drenar la cola para no bloquear el hilo de loop si estaba en put()
        try:
            while not chunk_queue.empty():
                chunk_queue.get_nowait()
                chunk_queue.task_done()
        except Exception:
            pass
        producer_thread.join(timeout=2.0)

    return bytes_streamed
