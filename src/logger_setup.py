import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler

_DATE_SUFFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class OrganizedTimedRotatingFileHandler(TimedRotatingFileHandler):
    """일별 회전 로그를 연/월 하위 폴더로 정리하는 핸들러."""

    def __init__(self, filename, *args, archive_root_dir=None, **kwargs):
        self.archive_root_dir = os.path.abspath(
            archive_root_dir or os.path.dirname(os.path.abspath(filename))
        )
        super().__init__(filename, *args, **kwargs)

    def _extract_suffix(self, path: str):
        filename = os.path.basename(path)
        prefix = f"{os.path.basename(self.baseFilename)}."
        if not filename.startswith(prefix):
            return None
        suffix = filename[len(prefix):]
        if _DATE_SUFFIX_RE.fullmatch(suffix):
            return suffix
        return None

    def rotation_filename(self, default_name: str) -> str:
        suffix = self._extract_suffix(default_name)
        if not suffix:
            return default_name
        year, month, _ = suffix.split("-")
        return os.path.join(self.archive_root_dir, year, month, os.path.basename(default_name))

    def rotate(self, source: str, dest: str):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            os.remove(dest)
        os.replace(source, dest)

    def getFilesToDelete(self):
        if self.backupCount <= 0:
            return []

        prefix = f"{os.path.basename(self.baseFilename)}."
        candidates = []
        for root_dir, _, files in os.walk(self.archive_root_dir):
            for filename in files:
                if not filename.startswith(prefix):
                    continue
                suffix = filename[len(prefix):]
                if _DATE_SUFFIX_RE.fullmatch(suffix):
                    candidates.append((suffix, os.path.join(root_dir, filename)))

        candidates.sort(key=lambda item: item[0])
        if len(candidates) <= self.backupCount:
            return []
        return [path for _, path in candidates[:-self.backupCount]]


def _organize_legacy_log_archives(log_dir: str, base_names=None):
    """루트에 쌓인 과거 일별 로그를 연/월 폴더로 이동한다."""
    if base_names is None:
        base_names = ("trading.log", "orders.log")

    os.makedirs(log_dir, exist_ok=True)

    for filename in os.listdir(log_dir):
        src = os.path.join(log_dir, filename)
        if not os.path.isfile(src):
            continue

        for base_name in base_names:
            prefix = f"{base_name}."
            if not filename.startswith(prefix):
                continue

            suffix = filename[len(prefix):]
            if not _DATE_SUFFIX_RE.fullmatch(suffix):
                continue

            year, month, _ = suffix.split("-")
            dest_dir = os.path.join(log_dir, year, month)
            dest = os.path.join(dest_dir, filename)
            os.makedirs(dest_dir, exist_ok=True)

            if os.path.exists(dest):
                os.remove(src)
            else:
                os.replace(src, dest)
            break


def setup_logger(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """로깅을 설정하고 루트 로거를 반환한다."""
    os.makedirs(log_dir, exist_ok=True)
    _organize_legacy_log_archives(log_dir)

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

    # 메인 로그 파일 (오늘 로그는 trading.log, 이전 로그는 logs/YYYY/MM/ 아래로 이동)
    main_file = OrganizedTimedRotatingFileHandler(
        os.path.join(log_dir, "trading.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    main_file.suffix = "%Y-%m-%d"
    main_file.setLevel(level)
    main_file.setFormatter(fmt)
    root.addHandler(main_file)

    # 주문 전용 로그
    order_logger = logging.getLogger("kis_trader.orders")
    order_file = OrganizedTimedRotatingFileHandler(
        os.path.join(log_dir, "orders.log"),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    order_file.suffix = "%Y-%m-%d"
    order_file.setLevel(logging.INFO)
    order_file.setFormatter(fmt)
    order_logger.addHandler(order_file)

    return root
