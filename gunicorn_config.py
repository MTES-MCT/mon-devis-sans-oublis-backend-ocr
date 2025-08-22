import os

# Import config values
from app.config import config

# Server socket
bind = f"{config.HOST}:{config.PORT}"
backlog = 2048

# Worker processes
workers = config.WORKERS
worker_class = config.WORKER_CLASS
worker_connections = config.WORKER_CONNECTIONS
timeout = config.TIMEOUT
graceful_timeout = config.GRACEFUL_TIMEOUT
keepalive = config.KEEPALIVE

# Restart workers after this many requests, with some variability
max_requests = config.MAX_REQUESTS
max_requests_jitter = config.MAX_REQUESTS_JITTER

# Logging
loglevel = config.LOG_LEVEL
accesslog = "-" if config.ACCESS_LOG else None
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
    server.log.info(f"Enabled services: {config.get_enabled_services()}")
    server.log.info(f"Number of workers: {config.WORKERS}")