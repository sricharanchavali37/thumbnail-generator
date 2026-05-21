from celery import Celery
from backend.config import settings


# ─────────────────────────────────────────────────────────
# WHAT IS HAPPENING HERE
#
# We create one Celery application object.
# This object is the brain of the worker system.
#
# It needs two things to work:
#
# BROKER  = where jobs are sent and stored waiting
#           This is Redis.
#           FastAPI pushes jobs here.
#           Workers pull jobs from here.
#
# BACKEND = where job results are stored after finishing
#           This is also Redis.
#           After a worker finishes a job
#           Celery stores the result here temporarily.
#           We also update MongoDB directly from the worker
#           so this is a secondary record.
# ─────────────────────────────────────────────────────────

celery_app = Celery(
    # Name of this Celery application
    # Used internally by Celery for identification
    "thumbnail_generator",

    # BROKER: where jobs wait to be picked up
    # Redis URL from our .env file
    # Example: redis://localhost:6379/0
    # The /0 means database 0 inside Redis
    # Redis supports multiple databases numbered 0 to 15
    broker=settings.REDIS_URL,

    # BACKEND: where Celery stores task results
    # Using the same Redis instance
    # but database 1 to keep broker and results separate
    backend=settings.REDIS_URL.replace("/0", "/1"),

    # Tell Celery where our tasks live
    # Our video processing task is in this file
    include=["backend.workers.video_processor"],
)


# ─────────────────────────────────────────────────────────
# CELERY CONFIGURATION
#
# These settings control how Celery behaves.
# Each setting is explained below.
# ─────────────────────────────────────────────────────────

celery_app.conf.update(

    # ── Serialization ────────────────────────────────────
    # How data is converted when sending jobs to Redis
    # and receiving results back.
    # JSON is human readable and works everywhere.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Timezone ─────────────────────────────────────────
    # Always use UTC for timestamps.
    # Never use local timezone in servers.
    # UTC is consistent everywhere in the world.
    # Your MongoDB timestamps will match your Celery timestamps.
    timezone="UTC",
    enable_utc=True,

    # ── Task Results ─────────────────────────────────────
    # How long to keep task results in Redis.
    # After this time results are automatically deleted.
    # We keep them for 1 hour.
    # We do not need them longer because
    # all important data is in MongoDB anyway.
    # Redis is just a secondary record.
    result_expires=3600,  # 1 hour in seconds

    # ── Worker Settings ───────────────────────────────────
    # How many tasks one worker can run at the same time.
    # We set this to 1 because video processing is heavy.
    # One video at a time per worker process.
    # If you want more parallel processing
    # you add more worker machines, not more concurrency.
    # Running 4 videos on one machine simultaneously
    # would fight over the same CPU and slow everything down.
    worker_concurrency=1,

    # ── Retries ───────────────────────────────────────────
    # If a task fails should Celery retry it automatically?
    # We set this to False because:
    # We handle retries manually from the frontend
    # via the re-generate button.
    # Automatic retries could cause duplicate processing
    # which wastes resources and confuses users.
    task_acks_late=True,
    # task_acks_late means:
    # Only mark the job as done in Redis
    # AFTER the worker finishes processing.
    # If the worker crashes mid-job
    # the job goes back to the queue automatically.
    # Another worker picks it up.
    # This prevents jobs from disappearing if a worker dies.

    # ── Queue Settings ────────────────────────────────────
    # Which queue to use by default.
    # We use one queue called "video_processing".
    # All jobs go into this queue.
    # All workers pull from this queue.
    task_default_queue="video_processing",

    # ── Prefetch ──────────────────────────────────────────
    # How many jobs a worker grabs from Redis at once.
    # We set this to 1.
    # Worker grabs 1 job, processes it completely,
    # then grabs the next one.
    # If we set this to 4 a worker would grab 4 jobs
    # even if it can only work on 1 at a time.
    # The other 3 would sit idle in that worker
    # instead of being available to other workers.
    worker_prefetch_multiplier=1,
)