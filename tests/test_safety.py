from pathlib import Path

import pytest

from btcore.safety import StrategySafetyError, validate_strategy_file, validate_strategy_source


def test_strategy_file_must_be_under_strategies(tmp_path: Path):
    path = tmp_path / "not_strategy.py"
    path.write_text("class Strategy: pass", encoding="utf-8")

    with pytest.raises(StrategySafetyError):
        validate_strategy_file(path)


def test_blocks_forbidden_import_and_negative_index():
    with pytest.raises(StrategySafetyError):
        validate_strategy_source("import os\n")
    with pytest.raises(StrategySafetyError):
        validate_strategy_source("def f(x):\n    return x[-1]\n")
