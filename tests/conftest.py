"""Pytest setup — udostępnia importy z `src/` bez instalacji pakietu.

Dodaje na sys.path:
- `src/` → pozwala `import ml_project.settings`
- `src/pipeline/` → pozwala `import feature_store` (moduł DLT pipeline)
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "src" / "pipeline"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
