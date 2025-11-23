from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import settings

limiter = Limiter(key_func=get_remote_address)
limit_param = f"{settings.RATE_LIMIT_MIN}/minute"
