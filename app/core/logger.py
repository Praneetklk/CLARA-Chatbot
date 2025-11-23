# app/core/logger.py
import logging
from core.config import settings

logger = logging.getLogger("chatbot-logger")
logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
logger.propagate = False  

# Always add a console handler with a simple, structured-ish format
_console = logging.StreamHandler()
_console.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
_console.setFormatter(logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
))
logger.addHandler(_console)

# CloudWatch logging disabled per request; console logging remains
# if not settings.DEBUG:
#     try:
#         import watchtower
#         cw = watchtower.CloudWatchLogHandler(
#             log_group=getattr(settings, "CLOUDWATCH_LOG_GROUP", "medlaunch-chatbot-log-group"),
#         )
#         cw.setLevel(logging.INFO)
#         cw.setFormatter(logging.Formatter(
#             fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#             datefmt="%Y-%m-%dT%H:%M:%S%z",
#         ))
#         logger.addHandler(cw)
#         logger.info("CloudWatch logging enabled")
#     except Exception as e:
#         logger.warning(f"CloudWatch logging not enabled: {e}")
