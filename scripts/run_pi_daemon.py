from __future__ import annotations

import logging

from agents.pi_daemon import run_daemon

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        run_daemon()
    except KeyboardInterrupt:
        logger.info("Pi daemon interrupted, exiting")


if __name__ == "__main__":
    main()
