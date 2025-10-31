"""
Lightweight adapter for the external repository:
  https://github.com/ItsMeRanajit/Math-Expression-Recognizer-and-Solver

Usage
-----
1) Copy that repository into your project at one of these paths:
     - external/Math-Expression-Recognizer-and-Solver
     - src/external/Math-Expression-Recognizer-and-Solver
     - vendor/Math-Expression-Recognizer-and-Solver
   Make sure the file `math-recog-model.h5` from that repo is present.

2) The GUI will expose a model choice "MER (External)" under Arithmetic.
   Selecting it routes recognition through this adapter. If the repo cannot
   be found, the adapter raises an informative error in the GUI area.

The adapter keeps the external code unmodified and interacts by:
 - Writing the GUI-drawn image to the repo's expected `temp/` folder
 - Calling `image_preprocessing.image_preprocessing(file_path)`
 - Calling `symbol_recognition.symbol_recognition()` to retrieve the equation
 - Optionally calling `equation_building_solving.build_and_solve(eqn)`

No probabilities are produced by the external model, so the adapter returns
uniform confidences (1.0) per character to satisfy the GUI display.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


POSSIBLE_REPO_DIRS = [
    Path("external") / "Math-Expression-Recognizer-and-Solver",
    Path("src") / "external" / "Math-Expression-Recognizer-and-Solver",
    Path("vendor") / "Math-Expression-Recognizer-and-Solver",
    Path("third_party") / "Math-Expression-Recognizer-and-Solver",
]


@dataclass
class _RepoHandles:
    root: Path
    preproc: object
    recogniser: object
    solver: object | None


class MERSolverAdapter:
    def __init__(self) -> None:
        self._handles: _RepoHandles | None = None

    def _find_repo(self) -> Optional[Path]:
        for candidate in POSSIBLE_REPO_DIRS:
            if (candidate / "symbol_recognition.py").exists() and (
                candidate / "image_preprocessing.py"
            ).exists():
                return candidate.resolve()
        return None

    def _ensure_loaded(self) -> _RepoHandles:
        if self._handles is not None:
            return self._handles

        root = self._find_repo()
        if root is None:
            raise RuntimeError(
                "MER repo not found. Copy 'Math-Expression-Recognizer-and-Solver' into 'external/' or 'src/external/'."
            )

        # Add repo to sys.path for direct import without modifying upstream
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            preproc = __import__("image_preprocessing")
            recogniser = __import__("symbol_recognition")
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError(f"Failed to import MER modules: {exc}")

        solver_mod = None
        try:
            solver_mod = __import__("equation_building_solving")
        except Exception:
            solver_mod = None

        self._handles = _RepoHandles(root=root, preproc=preproc, recogniser=recogniser, solver=solver_mod)
        return self._handles

    def _write_temp_image(self, image: np.ndarray, repo_root: Path) -> Path:
        """Write the given image into the repo's expected 'temp/' location."""
        temp_dir = repo_root / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # The external code expects RGB/BGR 8-bit images on disk
        out_path = temp_dir / "gui_input.png"
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image
        cv2.imwrite(str(out_path), img_bgr)
        return out_path

    def recognize(self, image: np.ndarray) -> Tuple[str, List[Tuple[str, float]], List[np.ndarray]]:
        """Return recognised expression, per-char predictions, and glyphs.

        If the MER repo or model file is missing, raises a RuntimeError with a
        message the GUI will surface to the user.
        """
        handles = self._ensure_loaded()
        input_path = self._write_temp_image(image, handles.root)

        # Run the external pipeline end-to-end
        # 1) Preprocess -> writes 'temp/image_with_border.png'
        try:
            handles.preproc.image_preprocessing(str(input_path))
        except Exception as exc:
            raise RuntimeError(f"MER preprocessing failed: {exc}")

        # 2) Predict -> returns (eqn, final_eqn) where final uses '^' etc.
        try:
            eqn, final_eqn = handles.recogniser.symbol_recognition()
        except Exception as exc:
            raise RuntimeError(f"MER recognition failed: {exc}")

        recognised = final_eqn or eqn or ""

        # Build GUI-friendly outputs (no confidences available -> use 1.0)
        predictions = [(ch, 1.0) for ch in recognised]
        glyphs: List[np.ndarray] = []  # not available from upstream; leave empty
        return recognised, predictions, glyphs

    def solve(self, expression: str) -> Optional[str]:
        """Use MER's solver when available; otherwise return None to fallback."""
        handles = self._ensure_loaded()
        if not expression:
            return None
        if handles.solver is None:
            return None
        try:
            return str(handles.solver.build_and_solve(expression))
        except Exception:
            return None

