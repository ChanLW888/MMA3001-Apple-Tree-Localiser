"""Shared imports: only the three audited, inference-free modules are loaded."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import apple_predictor_v1 as detector
import compare_apple_detectors as comparison
import full_grid_search_apple as grid


def detection(x=5, y=5, apple_id=1, image='frame.png'):
    return dict(image=image, apple_id=apple_id, cx_px=x, cy_px=y,
                x1=x, y1=y, x2=x, y2=y, area_px=1,
                branch_overlap=1, confidence='')
