import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


DATA_MODULE_PATH = Path(__file__).with_name("data.py")
SPEC = importlib.util.spec_from_file_location("dataset_helper", DATA_MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load dataset helper from {DATA_MODULE_PATH}")
dataset_helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset_helper)


class DatasetHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "kaggle-cache"
        self.source.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def set_kaggle_source(self, source):
        fake_kagglehub = types.ModuleType("kagglehub")
        fake_kagglehub.dataset_download = lambda _: str(source)
        return patch.dict("sys.modules", {"kagglehub": fake_kagglehub})

    def test_copies_dataset_structure(self):
        nested = self.source / "csv" / "prices.csv"
        nested.parent.mkdir()
        nested.write_text("date,close\n2024-01-01,100\n", encoding="utf-8")
        destination = self.root / "organized" / "google-stock-prediction"

        with self.set_kaggle_source(self.source):
            result = dataset_helper.download_dataset(destination)

        self.assertEqual(result, destination.resolve())
        self.assertEqual(
            (result / "csv" / "prices.csv").read_text(encoding="utf-8"),
            "date,close\n2024-01-01,100\n",
        )

    def test_stages_updated_files_into_existing_destination(self):
        destination = self.root / "organized"
        destination.mkdir()
        (destination / "old.txt").write_text("keep", encoding="utf-8")
        (self.source / "new.txt").write_text("dataset", encoding="utf-8")

        with self.set_kaggle_source(self.source):
            result = dataset_helper.download_dataset(destination)

        self.assertTrue((result / "old.txt").is_file())
        self.assertEqual((result / "new.txt").read_text(encoding="utf-8"), "dataset")

    def test_rejects_non_directory_download(self):
        not_a_directory = self.root / "not-a-directory.txt"
        not_a_directory.write_text("invalid", encoding="utf-8")
        with self.set_kaggle_source(not_a_directory), self.assertRaises(
            NotADirectoryError
        ):
            dataset_helper.download_dataset(self.root / "organized")


if __name__ == "__main__":
    unittest.main(verbosity=2)
