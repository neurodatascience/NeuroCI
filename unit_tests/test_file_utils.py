import pytest
import os
import json
import pathlib
import subprocess
from unittest import mock
from file_utils import FileOperations

# ----------------------------- Tests ----------------------------- #

# -------------------- push_state_to_repo --------------------
@mock.patch("ssh_utils.SSHConnectionManager")
@mock.patch("pathlib.Path.mkdir")
@mock.patch("pathlib.Path.exists")
@mock.patch("shutil.rmtree")
@mock.patch("logging.info")
@mock.patch("logging.warning")
def test_push_state_to_repo_calls_helpers(mock_warning, mock_info, mock_rmtree, mock_exists, mock_mkdir, mock_conn):
    instance = FileOperations()
    datasets = {"ds1": "/remote/path"}
    pipelines = {"fmriprep": "1.0.0"}

    instance._download_dataset_configs = mock.Mock()
    instance._download_pipeline_outputs = mock.Mock()
    instance._save_container_metadata = mock.Mock()
    instance._commit_and_push = mock.Mock()

    mock_exists.return_value = False

    instance.push_state_to_repo(mock_conn, datasets, pipelines, max_dl_size_per_dataset_tool_mb=50)

    instance._download_dataset_configs.assert_called_once()
    instance._download_pipeline_outputs.assert_called_once()
    instance._save_container_metadata.assert_called_once()
    instance._commit_and_push.assert_called_once_with("Update experiment state")


# -------------------- _download_dataset_configs --------------------
def test_download_dataset_configs_success_and_failure(tmp_path):
    instance = FileOperations()
    dataset_path = "/remote/path"
    dest_base = tmp_path

    conn = mock.Mock()
    # manifest.tsv succeeds, global_config.json fails, derivatives/processing_status.tsv succeeds
    conn.get.side_effect = [None, Exception("fail"), None]

    instance._download_dataset_configs(conn, dataset_path, dest_base)

    assert conn.get.call_count == 3
    called_files = [call.args[0] for call in conn.get.call_args_list]
    assert f"{dataset_path}/manifest.tsv" in called_files
    assert f"{dataset_path}/global_config.json" in called_files
    assert f"{dataset_path}/derivatives/processing_status.tsv" in called_files


# -------------------- _download_pipeline_outputs --------------------
def test_download_pipeline_outputs_calls_helpers(tmp_path):
    instance = FileOperations()
    dataset_name = "ds1"
    dataset_path = "/remote/path"
    dest_base = tmp_path
    pipelines = {"fmriprep": "1.0.0"}

    conn = mock.Mock()
    instance._download_directory = mock.Mock()
    instance._resolve_tracker_paths = mock.Mock(return_value=["file1", "file2"])
    instance._download_tarball_from_remote_dir = mock.Mock()

    instance._download_pipeline_outputs(conn, dataset_name, dataset_path, dest_base, pipelines, 50)

    instance._download_directory.assert_called_once()
    instance._resolve_tracker_paths.assert_called_once()
    instance._download_tarball_from_remote_dir.assert_called_once()
    args, kwargs = instance._download_tarball_from_remote_dir.call_args
    assert "file_paths_to_download" in kwargs
    assert kwargs["file_paths_to_download"] == ["file1", "file2"]


# -------------------- _save_container_metadata --------------------
def test_save_container_metadata_success(tmp_path):
    instance = FileOperations()
    dataset_path = "/remote/path"
    dest_base = tmp_path
    pipelines = {"fmriprep": "1.0.0"}

    conn = mock.Mock()
    # mock global_config.json returning container_store path
    global_config = {"SUBSTITUTIONS": {"[[NIPOPPY_DPATH_CONTAINERS]]": "/containers"}}
    conn.run.side_effect = [
        mock.Mock(stdout=json.dumps(global_config)),  # cat global_config.json
        mock.Mock(stdout='{"some": "metadata"}'),    # singularity inspect
    ]

    instance._save_container_metadata(conn, dataset_path, dest_base, pipelines)

    local_json_path = dest_base / "containers" / "fmriprep_1.0.0.json"
    assert local_json_path.exists()
    with open(local_json_path) as f:
        data = f.read()
    assert "metadata" in data


def test_save_container_metadata_missing_store(tmp_path):
    instance = FileOperations()
    dataset_path = "/remote/path"
    dest_base = tmp_path
    pipelines = {"fmriprep": "1.0.0"}

    conn = mock.Mock()
    # Force exception on reading global_config.json
    conn.run.side_effect = Exception("fail")

    instance._save_container_metadata(conn, dataset_path, dest_base, pipelines)
    # should not crash, and not write anything
    container_dir = dest_base / "containers"
    assert list(container_dir.glob("*.json")) == []


# -------------------- _resolve_tracker_paths --------------------
def test_resolve_tracker_paths_expands_placeholders(tmp_path):
    instance = FileOperations()
    dataset_path = "/remote/path"
    manifest_path = f"{dataset_path}/manifest.tsv"
    tracker_path = f"{dataset_path}/pipelines/processing/fmriprep-1.0.0/tracker.json"

    conn = mock.Mock()
    # Simulated manifest.tsv with 1 row
    manifest_content = "participant_id\tsession_id\n01\t1\n"
    conn.run.side_effect = [
        mock.Mock(stdout=manifest_content),
        mock.Mock(stdout=json.dumps({"PATHS": ["[[NIPOPPY_BIDS_PARTICIPANT_ID]]/[[NIPOPPY_BIDS_SESSION_ID]]"]}))
    ]

    resolved = instance._resolve_tracker_paths(conn, manifest_path, tracker_path, dataset_path, "fmriprep", "1.0.0")

    assert any("sub-01" in p and "ses-1" in p for p in resolved)
    assert all(p.startswith(f"{dataset_path}/derivatives/") for p in resolved)

@mock.patch("ssh_utils.Connection")
@mock.patch("pathlib.Path.mkdir")
@mock.patch("logging.info")
@mock.patch("logging.warning")
def test_download_directory_recursive(mock_warning, mock_info, mock_mkdir, mock_conn):
    instance = FileOperations() 
    mock_conn.run.side_effect = [
        mock.Mock(stdout="file1\ndir1"),
        mock.Mock(stdout="file2")
    ]
    instance._is_directory = mock.Mock(side_effect=[True, False, False])
    
    instance._download_directory(mock_conn, "/remote/dir", pathlib.Path("/local/dir"))

    assert mock_conn.run.call_count == 2
    assert mock_conn.get.call_count == 2


@mock.patch("ssh_utils.Connection")
@mock.patch("logging.warning")
def test_download_directory_list_failure(mock_warning, mock_conn):
    instance = FileOperations()  
    mock_conn.run.side_effect = Exception("List failed")
    
    instance._download_directory(mock_conn, "/remote/dir", pathlib.Path("/local/dir"))

    mock_warning.assert_called_once_with("Failed to list /remote/dir: List failed")


@mock.patch("ssh_utils.Connection")
@mock.patch("pathlib.Path.mkdir")
@mock.patch("logging.warning")
def test_download_directory_download_failure(mock_warning, mock_mkdir, mock_conn):
    instance = FileOperations()  
    mock_conn.run.return_value.stdout = "file1"
    instance._is_directory = mock.Mock(return_value=False)
    mock_conn.get.side_effect = Exception("Download failed")
    
    instance._download_directory(mock_conn, "/remote/dir", pathlib.Path("/local/dir"))

    mock_warning.assert_called_once_with("Failed to download /remote/dir/file1: Download failed")

@mock.patch("ssh_utils.Connection")
@mock.patch("logging.warning")
def test_is_directory_true(mock_warning, mock_conn):
    instance = FileOperations()  # Replace with the actual class name
    mock_conn.run.return_value.stdout = "1"
    assert instance._is_directory(mock_conn, "/remote/path") is True
    mock_warning.assert_not_called()

@mock.patch("ssh_utils.Connection")
@mock.patch("logging.warning")
def test_is_directory_false(mock_warning, mock_conn):
    instance = FileOperations()  # Replace with the actual class name
    mock_conn.run.return_value.stdout = "0"
    assert instance._is_directory(mock_conn, "/remote/path") is False
    mock_warning.assert_not_called()

@mock.patch("ssh_utils.Connection")
@mock.patch("logging.warning")
def test_is_directory_exception(mock_warning, mock_conn):
    instance = FileOperations()  # Replace with the actual class name
    mock_conn.run.side_effect = Exception("Test failed")
    assert instance._is_directory(mock_conn, "/remote/path") is False
    mock_warning.assert_called_once_with("Could not determine if directory: /remote/path — Test failed")

@mock.patch("ssh_utils.Connection")
def test_is_directory_strip_output(mock_conn):
    instance = FileOperations()  # Replace with the actual class name
    mock_conn.run.return_value.stdout = " 1 \n"
    assert instance._is_directory(mock_conn, "/remote/path") is True

@mock.patch("ssh_utils.Connection")
@mock.patch("logging.warning")
def test_is_directory_true(mock_warning, mock_conn):
    instance = FileOperations()
    mock_conn.run.return_value.stdout = "1"
    assert instance._is_directory(mock_conn, "/remote/path") is True
    mock_warning.assert_not_called()

@mock.patch("ssh_utils.Connection")
@mock.patch("logging.warning")
def test_is_directory_false(mock_warning, mock_conn):
    instance = FileOperations()
    mock_conn.run.return_value.stdout = "0"
    assert instance._is_directory(mock_conn, "/remote/path") is False
    mock_warning.assert_not_called()

@mock.patch("ssh_utils.Connection")
@mock.patch("logging.warning")
def test_is_directory_exception(mock_warning, mock_conn):
    instance = FileOperations()
    mock_conn.run.side_effect = Exception("Test failed")
    assert instance._is_directory(mock_conn, "/remote/path") is False
    mock_warning.assert_called_once_with("Could not determine if directory: /remote/path — Test failed")

@mock.patch("ssh_utils.Connection")
def test_is_directory_strip_output(mock_conn):
    instance = FileOperations() 
    mock_conn.run.return_value.stdout = " 1 \n"
    assert instance._is_directory(mock_conn, "/remote/path") is True

@mock.patch("subprocess.run")
@mock.patch("logging.info")
def test_commit_and_push(mock_info, mock_run):
    instance = FileOperations()
    mock_run.side_effect = [
        None,  # git config user.name
        None,  # git config user.email
        None,  # git add
        subprocess.CompletedProcess(args=[], returncode=1),  # git diff (changes to commit)
        None,  # git commit
        None  # git push
    ]
    instance._commit_and_push("Test commit message")
    assert mock_run.call_count == 6
    mock_info.assert_called_with("✓ Pushed updated experiment state to remote repo.")

@mock.patch("subprocess.run")
@mock.patch("logging.info")
def test_commit_and_push_no_changes(mock_info, mock_run):
    instance = FileOperations()
    mock_run.side_effect = [
        None,  # git config user.name
        None,  # git config user.email
        None,  # git add
        subprocess.CompletedProcess(args=[], returncode=0)  # git diff (no changes)
    ]
    instance._commit_and_push("Test commit message")
    assert mock_run.call_count == 4
    mock_info.assert_called_with("✓ No changes to commit.")

@mock.patch("subprocess.run")
@mock.patch("logging.error")
def test_commit_and_push_git_failure(mock_error, mock_run):
    instance = FileOperations() 
    mock_run.side_effect = subprocess.CalledProcessError(1, ["git", "commit"])
    with pytest.raises(subprocess.CalledProcessError):
        instance._commit_and_push("Test commit message")
    mock_error.assert_called_once_with("✗ Git operation failed: Command '['git', 'commit']' returned non-zero exit status 1.")

@mock.patch("subprocess.run")
@mock.patch("pathlib.Path.iterdir")
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.info")
@mock.patch("logging.error")
def test_run_user_scripts(mock_error, mock_info, mock_exists, mock_iterdir, mock_run):
    instance = FileOperations()
    userscripts = {"script1": "script1.py"}
    mock_exists.return_value = True
    mock_iterdir.return_value = [mock.Mock()]  # Simulate non-empty dir
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    instance._commit_and_push = mock.Mock()
    instance.run_user_scripts(userscripts)
    mock_info.assert_any_call("✓ Successfully executed: script1.py")
    instance._commit_and_push.assert_called_once_with("Update experiment state from user scripts")


@mock.patch("subprocess.run")
@mock.patch("pathlib.Path.iterdir", return_value=[pathlib.Path("/tmp/neuroci_idp_state/file.txt")])
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.info")
@mock.patch("logging.error")
def test_run_user_scripts_not_found(mock_error, mock_info, mock_exists, mock_iterdir, mock_run):
    instance = FileOperations()
    userscripts = {"script1": "script1.py"}
    script_path = instance.repo_root / "user_scripts" / "script1.py"
    state_path = pathlib.Path("/tmp/neuroci_idp_state")
    
    mock_exists.side_effect = [
        True,  # for state_path.exists()
        False  # for script_path.exists()
    ]
    
    instance.run_user_scripts(userscripts)
    
    mock_error.assert_called_once_with(f"User script not found: {script_path}")

@mock.patch("subprocess.run")
@mock.patch("pathlib.Path.iterdir", return_value=[pathlib.Path("/tmp/neuroci_idp_state/file.txt")])
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.info")
@mock.patch("logging.error")
def test_run_user_scripts_failure(mock_error, mock_info, mock_exists, mock_iterdir, mock_run):
    instance = FileOperations()
    userscripts = {"script1": "script1.py"}
    script_path = instance.repo_root / "user_scripts" / "script1.py"
    state_path = pathlib.Path("/tmp/neuroci_idp_state")
    
    mock_exists.side_effect = [
        True,  # for state_path.exists()
        True   # for script_path.exists()
    ]
    
    # Simulate a script failure
    mock_run.side_effect = subprocess.CalledProcessError(1, ["python"])
    instance._commit_and_push = mock.Mock()
    
    with pytest.raises(RuntimeError):
        instance.run_user_scripts(userscripts)
    
    # Get the actual error call argument and assert the start of the message
    error_msg = mock_error.call_args[0][0]
    assert error_msg.startswith("✗ Error executing script1.py: ")
    instance._commit_and_push.assert_not_called()



@mock.patch("subprocess.run")
@mock.patch("pathlib.Path.iterdir", return_value=[pathlib.Path("/tmp/neuroci_idp_state/file.txt")])
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.info")
def test_run_user_scripts_commit(mock_info, mock_exists, mock_iterdir, mock_run):
    instance = FileOperations()
    userscripts = {"script1": "script1.py"}
    script_path = instance.repo_root / "user_scripts" / "script1.py"
    state_path = pathlib.Path("/tmp/neuroci_idp_state")
    
    mock_exists.side_effect = [
        True,  # for state_path.exists()
        True   # for script_path.exists()
    ]
    
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    instance._commit_and_push = mock.Mock()
    
    instance.run_user_scripts(userscripts)
    
    instance._commit_and_push.assert_called_once_with("Update experiment state from user scripts")
    mock_info.assert_called_with("Committing results of user scripts to repo...")
