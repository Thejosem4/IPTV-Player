import asyncio
import aiohttp
import time
import threading

class PieceBuffer:
    def __init__(self, expected_size):
        self.data = bytearray()
        self.expected_size = expected_size
        self.done = False
        self.error = None
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.read_pos = 0

    def append(self, chunk):
        with self.cond:
            self.data.extend(chunk)
            self.cond.notify_all()

    def set_done(self, error=None):
        with self.cond:
            self.done = True
            self.error = error
            self.cond.notify_all()

    def stream_to(self, wfile):
        """Lee el buffer progresivamente y lo escribe en el socket sincrónico del cliente"""
        while True:
            with self.cond:
                while self.read_pos == len(self.data) and not self.done:
                    self.cond.wait(timeout=1.0)
                
                if self.read_pos < len(self.data):
                    chunk = self.data[self.read_pos:]
                    self.read_pos = len(self.data)
                else:
                    if self.error:
                        raise Exception(self.error)
                    break
            
            try:
                wfile.write(chunk)
            except Exception as e:
                # Client disconnected
                raise e

async def _download_part(session, url, idx, start, end, piece, ai_headers_func, ai_jitter_func):
    headers = {}
    if ai_headers_func:
        try:
            headers = ai_headers_func(idx, url)
        except Exception:
            pass
            
    if not headers:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": url
        }
    headers["Range"] = f"bytes={start}-{end}"
    
    if ai_jitter_func:
        try:
            delay = ai_jitter_func(idx)
            await asyncio.sleep(delay)
        except Exception:
            pass

    try:
        async with session.get(url, headers=headers, timeout=120) as resp:
            resp.raise_for_status()
            while True:
                chunk = await resp.content.read(64 * 1024)
                if not chunk:
                    break
                piece.append(chunk)
        piece.set_done()
    except Exception as e:
        piece.set_done(error=str(e))

async def _run_parallel_downloads(url, num_conn, content_length, pieces, ai_headers_func, ai_jitter_func):
    chunk_size = content_length // num_conn
    ranges = []
    for i in range(num_conn):
        start = i * chunk_size
        end = (i + 1) * chunk_size - 1 if i < num_conn - 1 else content_length - 1
        ranges.append((start, end))

    # Connection pooling with high limits for extreme parallel pieces
    connector = aiohttp.TCPConnector(limit=num_conn + 5, keepalive_timeout=60)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for idx, (start, end) in enumerate(ranges):
            task = asyncio.create_task(_download_part(
                session, url, idx, start, end, pieces[idx], ai_headers_func, ai_jitter_func
            ))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

def download_and_stream(url, num_conn, content_length, wfile, ai_headers_func=None, ai_jitter_func=None):
    """
    Ejecuta la descarga asíncrona en un hilo separado y emite los bytes
    de manera secuencial al cliente en tiempo real.
    """
    chunk_size = content_length // num_conn
    pieces = []
    for i in range(num_conn):
        start = i * chunk_size
        end = (i + 1) * chunk_size - 1 if i < num_conn - 1 else content_length - 1
        pieces.append(PieceBuffer(end - start + 1))

    # Iniciar el motor asyncio en un hilo separado
    loop = asyncio.new_event_loop()
    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_parallel_downloads(
            url, num_conn, content_length, pieces, ai_headers_func, ai_jitter_func
        ))
        loop.close()
        
    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()

    # Transmitir progresivamente al socket sincrónico del cliente
    try:
        for idx, piece in enumerate(pieces):
            piece.stream_to(wfile)
            # print(f"   ✅ Pieza {idx+1}/{num_conn} enviada al reproductor.")
    except Exception as e:
        # Si el cliente se desconecta (ej. el usuario hace seek), cerramos el loop
        loop.call_soon_threadsafe(loop.stop)
        raise e
