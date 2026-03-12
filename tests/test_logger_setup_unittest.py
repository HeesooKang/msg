import os
import tempfile
import unittest

from src.logger_setup import (
    OrganizedTimedRotatingFileHandler,
    _organize_legacy_log_archives,
)


class LoggerSetupTests(unittest.TestCase):
    def test_rotation_filename_uses_year_month_archive_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = OrganizedTimedRotatingFileHandler(
                os.path.join(temp_dir, "trading.log"),
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8",
            )
            self.addCleanup(handler.close)

            rotated = handler.rotation_filename(
                os.path.join(temp_dir, "trading.log.2026-03-11")
            )

            self.assertEqual(
                rotated,
                os.path.join(temp_dir, "2026", "03", "trading.log.2026-03-11"),
            )

    def test_organize_legacy_log_archives_moves_dated_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            active_log = os.path.join(temp_dir, "trading.log")
            archived_log = os.path.join(temp_dir, "trading.log.2026-03-10")
            with open(active_log, "w", encoding="utf-8") as fp:
                fp.write("active\n")
            with open(archived_log, "w", encoding="utf-8") as fp:
                fp.write("archived\n")

            _organize_legacy_log_archives(temp_dir)

            self.assertTrue(os.path.exists(active_log))
            self.assertFalse(os.path.exists(archived_log))
            self.assertTrue(
                os.path.exists(
                    os.path.join(temp_dir, "2026", "03", "trading.log.2026-03-10")
                )
            )

    def test_get_files_to_delete_tracks_archives_across_subdirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = OrganizedTimedRotatingFileHandler(
                os.path.join(temp_dir, "orders.log"),
                when="midnight",
                interval=1,
                backupCount=2,
                encoding="utf-8",
            )
            self.addCleanup(handler.close)

            for day in ("2026-03-09", "2026-03-10", "2026-03-11"):
                year, month, _ = day.split("-")
                archive_dir = os.path.join(temp_dir, year, month)
                os.makedirs(archive_dir, exist_ok=True)
                with open(
                    os.path.join(archive_dir, f"orders.log.{day}"),
                    "w",
                    encoding="utf-8",
                ) as fp:
                    fp.write(day)

            self.assertEqual(
                handler.getFilesToDelete(),
                [os.path.join(temp_dir, "2026", "03", "orders.log.2026-03-09")],
            )


if __name__ == "__main__":
    unittest.main()
