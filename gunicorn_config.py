import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '80')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', '1'))
worker_class = os.getenv('WORKER_CLASS', 'uvicorn.workers.UvicornWorker')
worker_connections = int(os.getenv('WORKER_CONNECTIONS', '1000'))
timeout = int(os.getenv('TIMEOUT', '120'))
graceful_timeout = int(os.getenv('GRACEFUL_TIMEOUT', '30'))
keepalive = int(os.getenv('KEEPALIVE', '5'))

# Restart workers after this many requests, with some variability
max_requests = int(os.getenv('MAX_REQUESTS', '1000'))
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
    server.log.info(f"Number of workers: {workers}")