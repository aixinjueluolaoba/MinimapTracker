"""Thin ctypes binding for libfishing_native.so.

All localization and pointer inference stays in the native library.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, os.PathLike[str]]


class _NativeResult(ctypes.Structure):
    _fields_ = [
        ("located", ctypes.c_int32),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("locate_cost_ms", ctypes.c_int32),
        ("pointer_detected", ctypes.c_int32),
        ("angle_deg", ctypes.c_float),
    ]


@dataclass(frozen=True)
class NavigationResult:
    located: bool
    x: float
    y: float
    locate_cost_ms: int
    pointer_detected: bool
    angle_deg: float


class NavigationEngine:
    """High-level native navigation engine accepting OpenCV BGR frames."""

    def __init__(
        self,
        model_root: PathLike,
        map_path: PathLike,
        cache_path: Optional[PathLike] = None,
        library_path: Optional[PathLike] = None,
    ) -> None:
        self._handle: Optional[int] = None
        if library_path is None:
            library_path = Path(__file__).resolve().parents[1] / "build" / "libfishing_native.so"

        self._lib = self._load_library(Path(library_path))
        self._configure_api()

        if cache_path is not None:
            cache = Path(cache_path)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache_bytes: Optional[bytes] = os.fsencode(cache)
        else:
            cache_bytes = None

        handle = self._lib.navigation_engine_create(
            os.fsencode(model_root),
            os.fsencode(map_path),
            cache_bytes,
        )
        if not handle:
            raise RuntimeError(self._last_error(None) or "failed to create navigation engine")
        self._handle = handle

    @staticmethod
    def _load_library(path: Path) -> ctypes.CDLL:
        try:
            return ctypes.CDLL(os.fspath(path))
        except OSError as error:
            raise RuntimeError(f"failed to load native library {path}: {error}") from error

    def _configure_api(self) -> None:
        self._lib.navigation_engine_create.restype = ctypes.c_void_p
        self._lib.navigation_engine_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._lib.navigation_engine_process_bgr.restype = ctypes.c_int
        self._lib.navigation_engine_process_bgr.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_NativeResult),
        ]
        self._lib.navigation_engine_last_error.restype = ctypes.c_char_p
        self._lib.navigation_engine_last_error.argtypes = [ctypes.c_void_p]
        self._lib.navigation_engine_release.restype = None
        self._lib.navigation_engine_release.argtypes = [ctypes.c_void_p]

    def _last_error(self, handle: Optional[int]) -> str:
        raw = self._lib.navigation_engine_last_error(handle)
        return raw.decode("utf-8", errors="replace") if raw else ""

    def process(self, frame_bgr: np.ndarray) -> NavigationResult:
        if self._handle is None:
            raise RuntimeError("navigation engine is closed")
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a numpy.ndarray")
        if frame_bgr.dtype != np.uint8:
            raise TypeError("frame_bgr dtype must be numpy.uint8")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr shape must be (height, width, 3)")

        frame = np.ascontiguousarray(frame_bgr)
        height, width = frame.shape[:2]
        result = _NativeResult()
        data = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ok = self._lib.navigation_engine_process_bgr(
            self._handle,
            data,
            width,
            height,
            frame.strides[0],
            ctypes.byref(result),
        )
        if not ok:
            raise RuntimeError(self._last_error(self._handle) or "native frame processing failed")

        return NavigationResult(
            located=bool(result.located),
            x=float(result.x),
            y=float(result.y),
            locate_cost_ms=int(result.locate_cost_ms),
            pointer_detected=bool(result.pointer_detected),
            angle_deg=float(result.angle_deg),
        )

    def close(self) -> None:
        if self._handle is not None:
            self._lib.navigation_engine_release(self._handle)
            self._handle = None

    def __enter__(self) -> "NavigationEngine":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        library = getattr(self, "_lib", None)
        if handle is not None and library is not None:
            library.navigation_engine_release(handle)
            self._handle = None
