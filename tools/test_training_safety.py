import tempfile
import unittest
from pathlib import Path
from training_safety import backup_before_training, verify_backup


class BackupTests(unittest.TestCase):
    def test_physical_copy_verified_and_reuse_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'original.pt'; source.write_bytes(b'weights-before-fit')
            dest = backup_before_training(root / 'run', [source])
            record = verify_backup(dest)['files'][0]
            copy = dest / record['backup']
            self.assertEqual(source.read_bytes(), copy.read_bytes())
            with self.assertRaises(FileExistsError):
                backup_before_training(root / 'run', [source])
            source.write_bytes(b'changed')
            self.assertEqual(copy.read_bytes(), b'weights-before-fit')
            with self.assertRaises(RuntimeError):
                verify_backup(dest)
            verify_backup(dest, verify_sources=False)

    def test_missing_or_corrupted_copy_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                backup_before_training(root / 'missing-run', [root / 'missing.pt'])
            source = root / 'original.pt'; source.write_bytes(b'checkpoint')
            dest = backup_before_training(root / 'run', [source])
            copy = dest / verify_backup(dest)['files'][0]['backup']; copy.write_bytes(b'bad-copy')
            with self.assertRaises(RuntimeError):
                verify_backup(dest, verify_sources=False)


if __name__ == '__main__':
    unittest.main()
