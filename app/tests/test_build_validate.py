"""Cross-platform checks for the frozen-output release gate."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from causal_shap.build.validate import _content_hash, _hash_tree


class FrozenOutputHashTests(unittest.TestCase):
    def test_hash_tree_uses_manifest_style_posix_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            artifact = nested / "artifact.txt"
            artifact.write_bytes(b"first\r\nsecond\r\n")

            hashes = _hash_tree(root)

        self.assertEqual(list(hashes), ["nested/artifact.txt"])
        self.assertEqual(
            hashes["nested/artifact.txt"],
            hashlib.sha256(b"first\nsecond\n").hexdigest(),
        )

    def test_binary_hash_preserves_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "artifact.png"
            content = b"binary\r\npayload\x00"
            artifact.write_bytes(content)

            digest = _content_hash(artifact)

        self.assertEqual(digest, hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
