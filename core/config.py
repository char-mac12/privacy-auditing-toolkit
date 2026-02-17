from enum import Enum

class LogLevel(Enum):
    NONE = 0
    INFO = 1
    VERBOSE = 2

LOG_LEVEL = LogLevel.INFO