import pytest
import os
import json
from unittest import mock
from ssh_utils import SSHConnectionManager

# ----------------------------- Fixtures ----------------------------- #

@pytest.fixture
def mock_env():
    with mock.patch.dict(os.environ, {
        "SSH_PRIVATE_KEY": "fake_private_key",
        "SSH_CONFIG_PATH": "/fake/.ssh/config"
    }):
        yield

# ----------------------------- Tests ----------------------------- #

@mock.patch("ssh_utils.Connection")
@mock.patch("ssh_utils.SSHConnectionManager._test_connection")
@mock.patch("ssh_utils.SSHConnectionManager._setup_ssh_config", return_value="/fake/.ssh/id_rsa")
@mock.patch("ssh_utils.SSHConnectionManager._ensure_known_hosts")
@mock.patch("ssh_utils.SSHConnectionManager._create_connection", return_value=(mock.Mock(), []))
def test_init_success(mock_create_conn, mock_known_hosts, mock_setup_key, mock_test_conn, mock_conn, mock_env):
    manager = SSHConnectionManager("myhost", "source env.sh", "SLURM")
    assert manager.target_host == "myhost"
    assert manager.scheduler == "SLURM"
    assert manager.conn is not None
    mock_setup_key.assert_called()
    mock_known_hosts.assert_called()
    mock_create_conn.assert_called()
    mock_test_conn.assert_called()

@mock.patch("ssh_utils.SSHConnectionManager._setup_ssh_config")
@mock.patch("ssh_utils.SSHConnectionManager._ensure_known_hosts")
@mock.patch("ssh_utils.SSHConnectionManager._create_connection")
@mock.patch("ssh_utils.SSHConnectionManager._test_connection")
@mock.patch.dict(os.environ, {"SSH_PRIVATE_KEY": "dummy_key"})
def test_setup_connection_success(mock_test_conn, mock_create_conn, mock_known_hosts, mock_setup_key):
    mock_create_conn.return_value = (mock.Mock(), [])
    mock_setup_key.return_value = "/fake/.ssh/id_rsa"
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.target_host = "myhost"
    manager._setup_connection()
    mock_setup_key.assert_called_once_with("myhost", "dummy_key", "~/.ssh/config")
    mock_known_hosts.assert_called_once_with("myhost", "~/.ssh/config")
    mock_create_conn.assert_called_once_with("myhost", "~/.ssh/config")
    mock_test_conn.assert_called_once()

@mock.patch.dict(os.environ, {})
def test_setup_connection_missing_private_key():
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.target_host = "myhost"
    with pytest.raises(EnvironmentError):
        manager._setup_connection()

@mock.patch.dict(os.environ, {"SSH_PRIVATE_KEY": "dummy_key"})
def test_setup_connection_missing_target_host():
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.target_host = None
    with pytest.raises(EnvironmentError):
        manager._setup_connection()

@mock.patch("ssh_utils.SSHConnectionManager._setup_ssh_config", side_effect=Exception("Setup failed"))
@mock.patch.dict(os.environ, {"SSH_PRIVATE_KEY": "dummy_key"})
def test_setup_connection_setup_ssh_config_failure(mock_setup_key):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.target_host = "myhost"
    with pytest.raises(Exception):
        manager._setup_connection()

@mock.patch("ssh_utils.open", new_callable=mock.mock_open)
@mock.patch("ssh_utils.os.chmod")
@mock.patch("ssh_utils.os.makedirs")
@mock.patch("ssh_utils.os.path.exists", return_value=True)
@mock.patch("ssh_utils.SSHConfig")
def test_setup_ssh_config(mock_config_class, mock_exists, mock_makedirs, mock_chmod, mock_open):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    mock_config = mock.Mock()
    mock_config.lookup.return_value = {"identityfile": ["~/.ssh/test_key"]}
    mock_config_class.return_value = mock_config

    path = manager._setup_ssh_config("myhost", "private_key_content", "/path/to/config")
    assert path.endswith("test_key")
    mock_open.assert_called()
    mock_chmod.assert_called()

@mock.patch("ssh_utils.open", new_callable=mock.mock_open)
@mock.patch("ssh_utils.os.chmod")
@mock.patch("ssh_utils.os.makedirs")
@mock.patch("ssh_utils.os.path.exists", return_value=True)
@mock.patch("ssh_utils.SSHConfig")
def test_setup_ssh_config_writes_private_key(mock_config_class, mock_exists, mock_makedirs, mock_chmod, mock_open):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    mock_config = mock.Mock()
    mock_config.lookup.return_value = {"identityfile": ["~/.ssh/test_key"]}
    mock_config_class.return_value = mock_config
    manager._setup_ssh_config("myhost", "private_key_content", "/path/to/config")
    mock_open.return_value.write.assert_called_once_with("private_key_content")

@mock.patch("ssh_utils.open", new_callable=mock.mock_open)
@mock.patch("ssh_utils.os.chmod")
@mock.patch("ssh_utils.os.makedirs")
@mock.patch("ssh_utils.os.path.exists", return_value=False)
@mock.patch("ssh_utils.SSHConfig")
def test_setup_ssh_config_config_file_not_found(mock_config_class, mock_exists, mock_makedirs, mock_chmod, mock_open):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    mock_config = mock.Mock()
    mock_config_class.return_value = mock_config
    path = manager._setup_ssh_config("myhost", "private_key_content", "/path/to/config")
    assert path is None

@mock.patch("ssh_utils.subprocess.run")
@mock.patch("ssh_utils.os.path.exists", return_value=True)
@mock.patch("ssh_utils.open", new_callable=mock.mock_open)
@mock.patch("ssh_utils.SSHConfig")
def test_ensure_known_hosts(mock_config_class, mock_open, mock_exists, mock_run):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    mock_config = mock.Mock()
    mock_config.lookup.return_value = {"hostname": "fakehost"}
    mock_config_class.return_value = mock_config
    manager._ensure_known_hosts("myhost", "/path/to/config")
    mock_run.assert_called()

@mock.patch("ssh_utils.subprocess.run")
@mock.patch("ssh_utils.os.path.exists", return_value=True)
@mock.patch("ssh_utils.open", new_callable=mock.mock_open)
@mock.patch("ssh_utils.SSHConfig")
def test_ensure_known_hosts_proxy_hosts(mock_config_class, mock_open, mock_exists, mock_run):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    mock_config = mock.Mock()
    mock_config.lookup.side_effect = [
        {"hostname": "fakehost", "proxyjump": "proxyhost"},
        {"hostname": "proxyhost"}
    ]
    mock_config_class.return_value = mock_config
    manager._ensure_known_hosts("myhost", "/path/to/config")
    assert mock_run.call_count == 2

@mock.patch("ssh_utils.os.path.exists", return_value=False)
@mock.patch("ssh_utils.SSHConfig")
def test_ensure_known_hosts_config_file_not_found(mock_config_class, mock_exists):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    with pytest.raises(FileNotFoundError):
        manager._ensure_known_hosts("myhost", "/path/to/config")

@mock.patch("ssh_utils.Connection")
@mock.patch("ssh_utils.SSHConfig")
@mock.patch("ssh_utils.os.path.exists", return_value=True)
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="")  
def test_create_connection(mock_open_file, mock_exists, mock_config_class, mock_connection):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.ssh_key_path = "/path/to/key"
    mock_config = mock.Mock()
    mock_config.lookup.return_value = {"hostname": "fakehost", "user": "fakeuser"}
    mock_config_class.return_value = mock_config
    connection, proxy_chain = manager._create_connection("myhost", "/path/to/config")
    assert connection is not None
    assert proxy_chain == []

@mock.patch("ssh_utils.Connection")
@mock.patch("ssh_utils.SSHConfig")
@mock.patch("ssh_utils.os.path.exists", return_value=True)
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="")
def test_create_connection_proxy_chain(mock_open_file, mock_exists, mock_config_class, mock_connection):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.ssh_key_path = "/path/to/key"
    mock_config = mock.Mock()
    mock_config.lookup.side_effect = [
        {"hostname": "fakehost", "user": "fakeuser", "proxyjump": "proxyhost"},
        {"hostname": "proxyhost", "user": "proxyuser"}
    ]
    mock_config_class.return_value = mock_config
    connection, proxy_chain = manager._create_connection("myhost", "/path/to/config")
    assert connection is not None
    assert len(proxy_chain) == 1

@mock.patch("ssh_utils.Connection")
@mock.patch("ssh_utils.SSHConfig")
@mock.patch("ssh_utils.os.path.exists", return_value=True)
@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="") 
def test_create_connection_connection_kwargs(mock_open_file, mock_exists, mock_config_class, mock_connection):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.ssh_key_path = "/path/to/key"
    mock_config = mock.Mock()
    mock_config.lookup.return_value = {"hostname": "fakehost", "user": "fakeuser", "port": 2222}
    mock_config_class.return_value = mock_config
    manager._create_connection("myhost", "/path/to/config")
    mock_connection.assert_called_once_with(
        host="fakehost",
        user="fakeuser",
        port=2222,
        connect_kwargs={"key_filename": "/path/to/key", "allow_agent": True},
    )

@mock.patch("ssh_utils.os.path.exists", return_value=False)
@mock.patch("ssh_utils.SSHConfig")
def test_create_connection_config_file_not_found(mock_config_class, mock_exists):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    with pytest.raises(FileNotFoundError):
        manager._create_connection("myhost", "/path/to/config")

@mock.patch("logging.info")
@mock.patch("logging.error")
def test_test_connection_success(mock_error, mock_info):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.conn = mock.Mock()
    result_mock = mock.Mock(ok=True, stdout="connected\n")
    manager.conn.run.return_value = result_mock
    manager._test_connection()
    manager.conn.run.assert_called_once_with("whoami", hide=True)
    mock_info.assert_called_with("SSH connection test passed: Logged in as connected")

@mock.patch("logging.error")
def test_test_connection_failure(mock_error):
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.conn = mock.Mock()
    result_mock = mock.Mock(ok=False)
    manager.conn.run.return_value = result_mock
    with pytest.raises(ConnectionError):
        manager._test_connection()
    manager.conn.run.assert_called_once_with("whoami", hide=True)
    mock_error.assert_called_with("SSH connection test failed. Check credentials and host configuration.")

@mock.patch("logging.info")
@mock.patch("logging.error")
@mock.patch("ssh_utils.Connection")
def test_run_nipoppy_command_success(mock_connection_class, mock_error, mock_info):
    mock_conn_instance = mock.MagicMock()
    mock_connection_class.return_value = mock_conn_instance
    mock_conn_instance.run.return_value.ok = True
    mock_conn_instance.run.return_value.stdout = "Success!"
    with mock.patch.dict(os.environ, {"SSH_PRIVATE_KEY": "dummy_key"}):
        ssh_mgr = SSHConnectionManager("user", "host", "SLURM")
        ssh_mgr.conn = mock_conn_instance
        ssh_mgr.prefix_cmd = "source env.sh"
        ssh_mgr.scheduler = "SLURM"
        ssh_mgr.run_nipoppy_command("run", "dataset_name", "/path/to/dataset", "pipeline_name", "pipeline_version")
        mock_info.assert_called_with("Successfully started pipeline for dataset_name - pipeline_name (pipeline_version)")

@mock.patch("logging.error")
@mock.patch("ssh_utils.SSHConnectionManager._setup_connection")  # 👈 Patch this
@mock.patch("ssh_utils.Connection")
def test_run_nipoppy_command_failure(mock_connection_class, mock_setup_connection, mock_error):
    mock_conn_instance = mock.MagicMock()
    mock_connection_class.return_value = mock_conn_instance
    mock_conn_instance.run.return_value.ok = False
    mock_conn_instance.run.return_value.stderr = "Error message"

    with mock.patch.dict(os.environ, {"SSH_PRIVATE_KEY": "dummy_key"}):
        ssh_mgr = SSHConnectionManager("user", "host", "SLURM")
        ssh_mgr.conn = mock_conn_instance  # manually set conn
        ssh_mgr.prefix_cmd = "source env.sh"
        ssh_mgr.scheduler = "SLURM"
        ssh_mgr.run_nipoppy_command("run", "dataset_name", "/path/to/dataset", "pipeline_name", "pipeline_version")
        mock_error.assert_any_call("Failed to start pipeline for dataset_name - pipeline_name (pipeline_version)")

@mock.patch("logging.info")
@mock.patch("ssh_utils.Connection")
def test_check_dataset_compliance_pass(mock_conn, mock_info):
    dataset = {"ds1": "/remote/path"}
    pipelines = {"fmriprep": "1.0.0"}
    extractors = {}
    global_config = {
        "SUBSTITUTIONS": {"[[NIPOPPY_DPATH_CONTAINERS]]": "/containers"},
        "PROC_PIPELINES": [{"NAME": "fmriprep", "VERSION": "1.0.0"}]
    }
    mock_conn.run.side_effect = [
        mock.Mock(stdout=json.dumps(global_config), ok=True),
        mock.Mock(),  # singularity inspect
        mock.Mock(stdout="invocation json content")  # invocation.json
    ]
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.conn = mock_conn
    manager.check_dataset_compliance(dataset, pipelines, extractors)
    mock_info.assert_called_with("All datasets comply with the experiment definition.")

@mock.patch("logging.error")
@mock.patch("ssh_utils.Connection")
def test_check_dataset_compliance_version_mismatch(mock_conn, mock_error):
    dataset = {"ds1": "/remote/path"}
    pipelines = {"fmriprep": "1.0.0"}
    extractors = {}
    global_config = {
        "SUBSTITUTIONS": {"[[NIPOPPY_DPATH_CONTAINERS]]": "/containers"},
        "PROC_PIPELINES": [{"NAME": "fmriprep", "VERSION": "2.0.0"}]
    }
    mock_conn.run.return_value = mock.Mock(stdout=json.dumps(global_config), ok=True)
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.conn = mock_conn
    with pytest.raises(ValueError):
        manager.check_dataset_compliance(dataset, pipelines, extractors)
    mock_error.assert_called_with("Dataset ds1 does not use the expected pipeline version: Expected fmriprep-1.0.0, Found 2.0.0")

def test_close_connection():
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.conn = mock.Mock()
    manager.conn.is_connected = True

    manager.close_connection()
    manager.conn.close.assert_called_once()

def test_close_connection_noop():
    manager = SSHConnectionManager.__new__(SSHConnectionManager)
    manager.conn = mock.Mock()
    manager.conn.is_connected = False

    manager.close_connection()
    manager.conn.close.assert_not_called()
