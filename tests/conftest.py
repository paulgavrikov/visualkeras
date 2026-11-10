from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def font_dir():
    return Path(__file__).resolve().parent / "assets" / "fonts"
