import pytest
import os
from experiment import Experiment
from unittest.mock import patch

# ----------------------------- Fixtures ----------------------------- #

# Automatically set required env var for SSH mock
@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["SSH_PRIVATE_KEY"] = "dummy"

# A complete base definition fixture
@pytest.fixture
def base_definition():
    return {
        "datasets": {"ds1": "/path/to/ds1"},
        "pipelines": {"pipeline1": "1.0"},
        "extractors": {},
        "userscripts": {},
        "target_host": "example.com",
        "prefix_cmd": "source env.sh",
        "scheduler": "slurm"
    }

# ----------------------------- Tests ----------------------------- #

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_valid_experiment_definition(mock_file_ops, mock_ssh_manager, base_definition):
    instance = Experiment(base_definition)
    assert instance.datasets == base_definition["datasets"]
    assert instance.pipelines == base_definition["pipelines"]
    assert instance.userscripts == base_definition["userscripts"]
    assert instance.target_host == base_definition["target_host"]
    assert instance.prefix_cmd == base_definition["prefix_cmd"]
    assert instance.scheduler == base_definition["scheduler"]

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_missing_datasets_raises(mock_file_ops, mock_ssh_manager, base_definition):
    del base_definition["datasets"]
    with pytest.raises(ValueError, match="datasets"):
        Experiment(base_definition)

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_missing_pipelines_raises(mock_file_ops, mock_ssh_manager, base_definition):
    del base_definition["pipelines"]
    with pytest.raises(ValueError, match="pipelines"):
        Experiment(base_definition)

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_missing_target_host_raises(mock_file_ops, mock_ssh_manager, base_definition):
    del base_definition["target_host"]
    with pytest.raises(ValueError, match="target_host"):
        Experiment(base_definition)

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_missing_scheduler_raises(mock_file_ops, mock_ssh_manager, base_definition):
    del base_definition["scheduler"]
    with pytest.raises(ValueError, match="scheduler"):
        Experiment(base_definition)

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_empty_datasets_raises(mock_file_ops, mock_ssh_manager, base_definition):
    base_definition["datasets"] = {}
    with pytest.raises(ValueError, match="datasets"):
        Experiment(base_definition)

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_empty_pipelines_raises(mock_file_ops, mock_ssh_manager, base_definition):
    base_definition["pipelines"] = {}
    with pytest.raises(ValueError, match="pipelines"):
        Experiment(base_definition)

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_empty_target_host_raises(mock_file_ops, mock_ssh_manager, base_definition):
    base_definition["target_host"] = ""
    with pytest.raises(ValueError, match="target_host"):
        Experiment(base_definition)

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_empty_scheduler_raises(mock_file_ops, mock_ssh_manager, base_definition):
    base_definition["scheduler"] = ""
    with pytest.raises(ValueError, match="scheduler"):
        Experiment(base_definition)

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_init_uses_default_prefix_cmd_if_missing(mock_file_ops, mock_ssh_manager, base_definition):
    del base_definition["prefix_cmd"]
    instance = Experiment(base_definition)
    assert instance.prefix_cmd == ""

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_file_ops_initialized(mock_file_ops, mock_ssh_manager, base_definition):
    instance = Experiment(base_definition)
    mock_file_ops.assert_called_once()
    assert hasattr(instance, 'file_ops')

@patch("experiment.SSHConnectionManager")
@patch("experiment.FileOperations")
def test_ssh_manager_initialized(mock_file_ops, mock_ssh_manager, base_definition):
    instance = Experiment(base_definition)
    mock_ssh_manager.assert_called_once_with(
        target_host=base_definition["target_host"],
        prefix_cmd=base_definition["prefix_cmd"],
        scheduler=base_definition["scheduler"]
    )
    assert hasattr(instance, 'ssh_manager')

