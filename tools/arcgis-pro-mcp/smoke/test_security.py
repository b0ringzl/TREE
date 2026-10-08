"""Unit checks for ArcGIS MCP path restrictions."""

from __future__ import annotations

import unittest

from security import load_config, validate_input_path, validate_output_path


class PathPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_allowed_input_under_tree(self) -> None:
        value = validate_input_path(
            r"D:\TREE\tree20250604_converted.csv", self.config
        )
        self.assertEqual(value, r"D:\TREE\tree20250604_converted.csv")

    def test_allowed_output_under_gis_output(self) -> None:
        value = validate_output_path(
            r"D:\TREE\gis_output\safe.gdb", self.config
        )
        self.assertEqual(value, r"D:\TREE\gis_output\safe.gdb")

    def test_output_outside_designated_root_is_rejected(self) -> None:
        with self.assertRaises(PermissionError):
            validate_output_path(
                r"D:\TREE\dataset\unsafe.gdb", self.config
            )

    def test_relative_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_input_path("relative.shp", self.config)


if __name__ == "__main__":
    unittest.main()
