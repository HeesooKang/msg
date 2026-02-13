import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """로깅을 설정하고 루트 로거를 반환한다."""
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 루트 로거
    root = logging.getLogger("kis_trader")
    root.setLevel(level)

    if root.handlers:
        return root

    # 콘솔
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 메인 로그 파일
    main_file = RotatingFileHandler(
        os.path.join(log_dir, "trading.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    main_file.setLevel(level)
    main_file.setFormatter(fmt)
    root.addHandler(main_file)

    # 주문 전용 로그
    order_logger = logging.getLogger("kis_trader.orders")
    order_file = RotatingFileHandler(
        os.path.join(log_dir, "orders.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    order_file.setLevel(logging.INFO)
    order_file.setFormatter(fmt)
    order_logger.addHandler(order_file)

    return root
