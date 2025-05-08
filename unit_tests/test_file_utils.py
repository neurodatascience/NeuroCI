import pytest
import os
import json
import pathlib
import subprocess
from unittest import mock
from file_utils import FileOperations

# ----------------------------- Tests ----------------------------- #

def test_repo_root_path_relative():
    instance = FileOperations() 
    assert instance.repo_root.resolve() == pathlib.Path(__file__).resolve().parents[1]

@mock.patch("ssh_utils.SSHConnectionManager")
@mock.patch("pathlib.Path.mkdir")
@mock.patch("pathlib.Path.exists")
@mock.patch("shutil.rmtree")
@mock.patch("logging.info")
@mock.patch("logging.warning")
def test_push_state_to_repo(mock_warning, mock_info, mock_rmtree, mock_exists, mock_mkdir, mock_conn):
    instance = FileOperations()  
    datasets = {"ds1": "/remote/path"}
    pipelines = {"fmriprep": "1.0.0"}
    extractors = {}
    mock_conn.get.side_effect = [None, Exception("Download failed")]
    instance.push_state_to_repo(mock_conn, datasets, pipelines, extractors)
    assert mock_conn.get.call_count == 3  # manifest.tsv, global_config.json, and derivatives/imaging_bagel.tsv

@mock.patch("ssh_utils.SSHConnectionManager")
@mock.patch("pathlib.Path.mkdir")
@mock.patch("pathlib.Path.exists")
@mock.patch("shutil.rmtree")
@mock.patch("logging.info")
@mock.patch("logging.warning")
def test_push_state_to_repo_download_directory(mock_warning, mock_info, mock_rmtree, mock_exists, mock_mkdir, mock_conn):
    instance = FileOperations() 
    datasets = {"ds1": "/remote/path"}
    pipelines = {"fmriprep": "1.0.0"}
    extractors = {"some_extractor": "1.0.0"}
    instance._download_directory = mock.Mock()
    instance.push_state_to_repo(mock_conn, datasets, pipelines, extractors)
    assert instance._download_directory.call_count == 3  # pipeline, extractor, and IDP outputs

@mock.patch("ssh_utils.SSHConnectionManager")
@mock.patch("pathlib.Path.mkdir")
@mock.patch("pathlib.Path.exists")
@mock.patch("shutil.rmtree")
@mock.patch("logging.info")
@mock.patch("logging.warning")
def test_push_state_to_repo_commit_and_push(mock_warning, mock_info, mock_rmtree, mock_exists, mock_mkdir, mock_conn):
    instance = FileOperations() 
    datasets = {"ds1": "/remote/path"}
    pipelines = {"fmriprep": "1.0.0"}
    extractors = {}
    instance._commit_and_push = mock.Mock()
    instance.push_state_to_repo(mock_conn, datasets, pipelines, extractors)
    instance._commit_and_push.assert_called_once_with("Update experiment state")

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
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.info")
@mock.patch("logging.error")
def test_run_user_scripts(mock_error, mock_info, mock_exists, mock_run):
    instance = FileOperations()  # Replace with the actual class name
    userscripts = {"script1": "script1.py"}
    mock_exists.return_value = True
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    instance._commit_and_push = mock.Mock()
    instance.run_user_scripts(userscripts)
    mock_info.assert_any_call("✓ Successfully executed: script1.py")
    instance._commit_and_push.assert_called_once_with("Update experiment state from user scripts")

@mock.patch("subprocess.run")
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.info")
@mock.patch("logging.error")
def test_run_user_scripts_not_found(mock_error, mock_info, mock_exists, mock_run):
    instance = FileOperations()  # Replace with the actual class name
    userscripts = {"script1": "script1.py"}
    mock_exists.return_value = False

    instance.run_user_scripts(userscripts)

    script_path = instance.repo_root / "user_scripts" / "script1.py"
    mock_error.assert_called_once_with(f"User script not found: {script_path}")

@mock.patch("subprocess.run")
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.info")
@mock.patch("logging.error")
def test_run_user_scripts_failure(mock_error, mock_info, mock_exists, mock_run):
    instance = FileOperations()  # Replace with the actual class name
    userscripts = {"script1": "script1.py"}
    mock_exists.return_value = True

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
@mock.patch("pathlib.Path.exists")
@mock.patch("logging.info")
def test_run_user_scripts_commit(mock_info, mock_exists, mock_run):
    instance = FileOperations()  # Replace with the actual class name
    userscripts = {"script1": "script1.py"}
    mock_exists.return_value = True
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
    instance._commit_and_push = mock.Mock()
    instance.run_user_scripts(userscripts)
    instance._commit_and_push.assert_called_once_with("Update experiment state from user scripts")
    mock_info.assert_called_with("Committing results of user scripts to repo...")
