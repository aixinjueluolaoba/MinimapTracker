#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from minimap_tracker import NavigationEngine  # noqa: E402


class NavigationEngineIntegrationTest(unittest.TestCase):
    def test_processes_bgr_frame_through_shared_library(self) -> None:
        with NavigationEngine(
            model_root=REPO / "models",
            map_path=REPO.parent / "app/src/main/assets/maps/big_map.png",
            cache_path=REPO / "build/simple_api_test_cache.bin",
        ) as engine:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            result = engine.process(frame)

        self.assertIsInstance(result.located, bool)
        self.assertIsInstance(result.pointer_detected, bool)
        self.assertGreaterEqual(result.angle_deg, 0.0)
        self.assertLess(result.angle_deg, 360.0)

    def test_rejects_invalid_frame_without_calling_algorithm(self) -> None:
        with NavigationEngine(
            model_root=REPO / "models",
            map_path=REPO.parent / "app/src/main/assets/maps/big_map.png",
            cache_path=REPO / "build/simple_api_test_cache.bin",
        ) as engine:
            with self.assertRaisesRegex(ValueError, "shape"):
                engine.process(np.zeros((720, 1280), dtype=np.uint8))
            with self.assertRaisesRegex(RuntimeError, "smaller than"):
                engine.process(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_reports_resource_error_from_shared_library(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "pointer param not found"):
            NavigationEngine(
                model_root=REPO / "missing-models",
                map_path=REPO.parent / "app/src/main/assets/maps/big_map.png",
            )


if __name__ == "__main__":
    unittest.main()
