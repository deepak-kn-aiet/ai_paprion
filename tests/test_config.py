"""Unit tests for central configuration and Streamlit secrets fallback."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import config


class TestConfigHelper:
    """Tests for get_config, get_bool_config, and get_int_config helpers."""

    def test_get_config_from_environ(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "env_val"}):
            assert config.get_config("TEST_KEY") == "env_val"

    def test_get_config_from_streamlit_secrets_fallback(self) -> None:
        mock_st = MagicMock()
        mock_st.secrets = {"TEST_SECRET_KEY": "secret_val"}

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict("sys.modules", {"streamlit": mock_st}),
        ):
            assert config.get_config("TEST_SECRET_KEY") == "secret_val"

    def test_get_config_returns_default_when_missing(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict("sys.modules", {"streamlit": MagicMock(secrets={})}),
        ):
            assert config.get_config("NON_EXISTENT_KEY", default="my_default") == "my_default"

    def test_environ_precedence_over_streamlit_secrets(self) -> None:
        mock_st = MagicMock()
        mock_st.secrets = {"DUAL_KEY": "from_secrets"}
        with (
            patch.dict(os.environ, {"DUAL_KEY": "from_env"}),
            patch.dict("sys.modules", {"streamlit": mock_st}),
        ):
            assert config.get_config("DUAL_KEY") == "from_env"

    def test_get_bool_config_truthy_and_falsy(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FLAG_TRUE_1": "true",
                "FLAG_TRUE_2": "1",
                "FLAG_TRUE_3": "yes",
                "FLAG_FALSE": "false",
                "FLAG_INVALID": "not_a_boolean",
            },
        ):
            assert config.get_bool_config("FLAG_TRUE_1") is True
            assert config.get_bool_config("FLAG_TRUE_2") is True
            assert config.get_bool_config("FLAG_TRUE_3") is True
            assert config.get_bool_config("FLAG_FALSE") is False
            assert config.get_bool_config("FLAG_INVALID", default=False) is False
            assert config.get_bool_config("UNSET_FLAG", default=True) is True
            assert config.get_bool_config("UNSET_FLAG", default=False) is False

    def test_get_int_config(self) -> None:
        with patch.dict(os.environ, {"COUNT_VALID": "42", "COUNT_INVALID": "not_an_int"}):
            assert config.get_int_config("COUNT_VALID", default=10) == 42
            assert config.get_int_config("COUNT_INVALID", default=10) == 10
            assert config.get_int_config("COUNT_MISSING", default=100) == 100
