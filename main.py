"""
CLI Entry Point
"""
import logging

import uvicorn

from config import API_CONFIG, CHAT_CONFIG, DATA_FETCH_CONFIG, LOG_FILE, LOG_LEVEL

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    logger.info("=" * 70)
    logger.info("  🎯 LOCAL FINANCE INTELLIGENCE SYSTEM v2.0")
    logger.info("  TimesFM 2.5 + %s", CHAT_CONFIG["model"])
    logger.info("=" * 70)

    logger.info("Loading models — first run downloads TimesFM weights...")

    logger.info("🌐 Open your browser: http://localhost:%s", API_CONFIG["port"])
    logger.info("📝 Indian: %s", ", ".join(DATA_FETCH_CONFIG["default_stocks"]))
    logger.info("📝 US: %s", ", ".join(DATA_FETCH_CONFIG["us_stocks"]))
    logger.info("⚠️  Chat runs Gemma 4 12B on CPU: several minutes per reply.")

    uvicorn.run(
        "api_server:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
        log_level="info",
    )


if __name__ == "__main__":
    main()
