from loguru import logger

logger.add("../logs/file.log", rotation="500 MB", enqueue=True)

