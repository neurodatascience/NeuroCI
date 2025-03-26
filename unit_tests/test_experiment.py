import pytest
import os
from unittest.mock import patch, MagicMock
from neuroci_code.experiment import Experiment


def test_experiment_init_valid_definition():
    """Test successful initialization with a valid experiment definition."""
    with patch.dict(os.environ, {"SSH_TARGET_HOST": "localhost", "SSH_PRIVATE_KEY": "dummy_key"}):
        with patch("neuroci_code.experiment.Experiment._setup_ssh_config", return_value="/fake/path"):
            with patch("neuroci_code.experiment.Experiment._ensure_known_hosts"):
                with patch("neuroci_code.experiment.Experiment._create_connection", return_value=(MagicMock(), [])):
                    experiment_definition = {"datasets": ["data1"], "pipelines": ["pipe1"]}
                    experiment = Experiment(experiment_definition)
                    assert experiment.datasets == ["data1"]
                    assert experiment.pipelines == ["pipe1"]


def test_experiment_init_missing_datasets():
    """Test initialization failure when datasets are missing."""
    experiment_definition = {"pipelines": ["pipe1"]}
    with pytest.raises(ValueError):
        Experiment(experiment_definition)


def test_experiment_init_missing_pipelines():
    """Test initialization failure when pipelines are missing."""
    experiment_definition = {"datasets": ["data1"]}
    with pytest.raises(ValueError):
        Experiment(experiment_definition)


def test_experiment_init_missing_ssh_env():
    """Test initialization failure when SSH env vars are missing."""
    experiment_definition = {"datasets": ["data1"], "pipelines": ["pipe1"]}
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError):
            Experiment(experiment_definition)


def test_setup_ssh_config_no_config_file():
    """Test that SSH setup skips when no config file exists."""
    experiment = MagicMock()
    experiment._setup_ssh_config.return_value = None
    assert experiment._setup_ssh_config("hostname", "key", "~/.ssh/missing_config") is None





# Running tests and coverage:
# To run: 'pytest path/to/tests/ -v'
# For coverage: 'pytest --cov=NeuroCI.neuroci_code.experiment path/to/tests/'

