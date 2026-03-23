"""Tests for src/utils/wandb_logger.py."""

import sys
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def mock_wandb_module():
    """Mock the wandb module to avoid import errors."""
    mock_wandb = MagicMock()
    saved = sys.modules.get("wandb")
    sys.modules["wandb"] = mock_wandb

    # Force reimport of wandb_logger
    sys.modules.pop("src.utils.wandb_logger", None)

    yield mock_wandb

    if saved is not None:
        sys.modules["wandb"] = saved
    else:
        sys.modules.pop("wandb", None)
    sys.modules.pop("src.utils.wandb_logger", None)


class TestWandbLogger:
    def test_init_wandb_calls_init(self, mock_wandb_module):
        from src.utils.wandb_logger import init_wandb
        cfg = MagicMock()
        cfg.wandb.project = "test-project"
        cfg.wandb.entity = None
        cfg.wandb.tags = ["a", "b"]
        cfg.wandb.mode = "disabled"
        cfg.experiment.name = "test_exp"
        cfg.model.short_name = "test_model"
        init_wandb(cfg)
        mock_wandb_module.init.assert_called_once()

    def test_log_dose_response(self, mock_wandb_module):
        from src.utils.wandb_logger import log_dose_response
        metrics = {
            "refusal_rate": 0.5,
            "mean_delta_cond": -0.1,
            "mean_delta_refusal": -0.2,
            "mean_delta_comply": 0.1,
            "mean_policy_score": -0.3,
        }
        log_dose_response("praise", 5.0, metrics)
        mock_wandb_module.log.assert_called_once()
        call_args = mock_wandb_module.log.call_args[0][0]
        assert "praise/alpha" in call_args
        assert call_args["praise/alpha"] == 5.0

    def test_log_damage_profile(self, mock_wandb_module):
        from src.utils.wandb_logger import log_damage_profile
        metrics = {"delta_cond": -0.1, "delta_refusal": -0.2}
        log_damage_profile("v_agree", metrics)
        mock_wandb_module.log.assert_called_once()
        call_args = mock_wandb_module.log.call_args[0][0]
        assert "damage/v_agree/delta_cond" in call_args

    def test_log_conditional_attack(self, mock_wandb_module):
        from src.utils.wandb_logger import log_conditional_attack
        log_conditional_attack("v_defer", 5, "praise", True, 0.3)
        mock_wandb_module.log.assert_called_once()
