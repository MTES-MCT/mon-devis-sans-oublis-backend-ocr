import os
import multiprocessing

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '80')}"

# Backlog - number of pending connections. Should be high for handling bursts
backlog = int(os.getenv('BACKLOG', '2048'))

# Worker processes
workers = int(os.getenv('WORKERS', '4'))
worker_class = os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')

# Worker connections - for async workers, this is the max concurrent requests per worker
# Set higher to allow queueing within each worker
worker_connections = int(os.getenv('WORKER_CONNECTIONS', '100'))

# Timeouts - increased for PDF processing
timeout = int(os.getenv('TIMEOUT', '300'))  # 5 minutes for large PDFs
graceful_timeout = int(os.getenv('GRACEFUL_TIMEOUT', '60'))  # More time for graceful shutdown
keepalive = int(os.getenv('KEEPALIVE', '5'))

# Thread pool - keep at 1 for thread safety with marker/pypdfium2
threads = int(os.getenv('THREADS', '1'))

# Don't preload the app - each worker needs its own model instance
preload_app = False
# Restart workers after this many requests, with some variability
max_requests = int(os.getenv('MAX_REQUESTS', '500'))
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', '50'))

# Logging
loglevel = os.getenv('LOG_LEVEL', 'info')
accesslog = "-" if os.getenv('ACCESS_LOG', 'true').lower() == 'true' else None
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'ocr-backend'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (optional, can be configured via environment variables)
keyfile = os.getenv("SSL_KEYFILE")
certfile = os.getenv("SSL_CERTFILE")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def on_starting(server):
    server.log.info("Starting Gunicorn server")
    enabled_services = os.getenv('ENABLED_SERVICES', 'marker,nanonets,olmocr')
    server.log.info(f"Enabled services: {enabled_services}")
    server.log.info(f"Configuration:")
    server.log.info(f"  Workers: {workers}")
    server.log.info(f"  Worker connections: {worker_connections}")
    server.log.info(f"  Backlog: {backlog}")
    server.log.info(f"  Timeout: {timeout}s")
    server.log.info(f"  Max requests per worker: {max_requests}")
    server.log.info(f"  Total capacity: {workers} workers × {worker_connections} connections = {workers * worker_connections} concurrent requests")

def worker_abort(worker):
    worker.log.info(f"Worker {worker.pid} aborted!")
    # Force cleanup on worker abort
    import gc
    gc.collect()

def on_reload(server):
    server.log.info("Reloading Gunicorn...")

def child_exit(server, worker):
    server.log.info(f"Worker {worker.pid} exited")