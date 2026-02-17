from core.config import LOG_LEVEL, LogLevel

def log(message: str, level: LogLevel = LogLevel.INFO):
    if LOG_LEVEL.value >= level.value:
        print(message)
