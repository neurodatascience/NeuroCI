import logging
import subprocess
import shutil
from pathlib import Path

class FileOperations:
    """
    Handles file operations for syncing dataset states, downloading pipeline outputs,
    executing user scripts, and managing version control via Git.
    """
    def __init__(self):
        # Set the root of the repository (assumes this script is two levels deep inside the repo)
        self.repo_root = Path(__file__).resolve().parents[1]

    def push_state_to_repo(self, conn, datasets, pipelines):
        """
        Downloads relevant files and pipeline outputs from remote datasets via SSH,
        stores them in a local 'experiment_state' directory, and pushes them to the Git repo.
        
        Args:
            conn: SSH connection manager object.
            datasets: Dictionary mapping dataset names to remote paths.
            pipelines: Dictionary of pipelines with their versions.
        """
        target_dir = self.repo_root / "experiment_state"
        logging.info(f"Syncing experiment state to local repo at: {target_dir}")

        for dataset_name, dataset_path in datasets.items():
            logging.info(f"Processing dataset: {dataset_name} from {dataset_path}")
            dest_base = target_dir / dataset_name

            # Remove any previously saved state for the dataset
            if dest_base.exists():
                logging.warning(f"Cleaning up old state in: {dest_base}")
                shutil.rmtree(dest_base)
            dest_base.mkdir(parents=True, exist_ok=True)

            # Download essential configuration files from the dataset
            for file in ["manifest.tsv", "global_config.json", "derivatives/processing_status.tsv"]:
                remote_path = f"{dataset_path}/{file}"
                local_path = dest_base / file
                local_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(f"Downloading file: {remote_path} -> {local_path}")
                try:
                    conn.get(remote_path, str(local_path))
                    logging.info(f"✓ Downloaded {file}")
                except Exception as e:
                    logging.warning(f"✗ Failed to download {file}: {e}")

            # Download pipeline outputs
            for tool, version in pipelines.items():
                pipeline_dir = f"pipelines/processing/{tool}-{version}"
                self._download_directory(conn, f"{dataset_path}/{pipeline_dir}", dest_base / pipeline_dir)

                # Also fetch the IDP outputs for the pipeline
                idp_dir = f"derivatives/{tool}/{version}/idp"
                self._download_directory(conn, f"{dataset_path}/{idp_dir}", Path("/tmp") / "neuroci_idp_state" / dataset_name / idp_dir)

        # Commit and push all the newly downloaded data to Git
        self._commit_and_push("Update experiment state")

    def _download_directory(self, conn, remote_dir, local_dir):
        """
        Recursively downloads a directory from the remote host, ignoring empty directories.
        
        Args:
            conn: SSH connection object.
            remote_dir: Remote directory to download.
            local_dir: Local target directory.
        """
        logging.info(f"Listing directory: {remote_dir}")
        try:
            result = conn.run(f"ls -1A {remote_dir}", hide=True)
        except Exception as e:
            logging.warning(f"Failed to list {remote_dir}: {e}")
            return

        for item in result.stdout.strip().splitlines():
            remote_path = f"{remote_dir}/{item}"
            local_path = local_dir / item

            if self._is_directory(conn, remote_path):
                local_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"[dir] {remote_path} — descending...")
                self._download_directory(conn, remote_path, local_path)
            else:
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    conn.get(remote_path, str(local_path))
                    logging.info(f"[file] {remote_path} -> {local_path}")
                except Exception as e:
                    logging.warning(f"Failed to download {remote_path}: {e}")

    def _is_directory(self, conn, remote_path):
        """
        Checks if the given remote path is a directory.
        
        Args:
            conn: SSH connection object.
            remote_path: Remote path to check.
        
        Returns:
            True if it's a directory, False otherwise.
        """
        try:
            result = conn.run(f"test -d {remote_path} && echo 1 || echo 0", hide=True)
            return result.stdout.strip() == "1"
        except Exception as e:
            logging.warning(f"Could not determine if directory: {remote_path} — {e}")
            return False

    def _commit_and_push(self, message):
        """
        Commits and pushes the experiment state directory to the remote Git repository.
        
        Args:
            message: Commit message to use.
        """
        try:
            subprocess.run(["git", "config", "user.name", "github_username"], check=True)
            subprocess.run(["git", "config", "user.email", "github_email@example.com"], check=True)
            subprocess.run(["git", "add", "experiment_state"], check=True)

            # Check if there are changes to commit
            result = subprocess.run(["git", "diff", "--cached", "--quiet"])
            if result.returncode == 0:
                logging.info("✓ No changes to commit.")
                return

            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
            logging.info("✓ Pushed updated experiment state to remote repo.")
        except subprocess.CalledProcessError as e:
            logging.error(f"✗ Git operation failed: {e}")
            raise

    def run_user_scripts(self, userscripts):
        """
        Executes user-defined Python scripts for post-processing or analysis.
        
        Args:
            userscripts: Dictionary mapping script keys to filenames.
        """
        script_dir = self.repo_root / "user_scripts"
        logging.info("Starting user-defined processing scripts...")

        for key, script_name in userscripts.items():
            script_path = script_dir / script_name
            logging.info(f"Executing user script [{key}]: {script_path}")

            if not script_path.exists():
                logging.error(f"User script not found: {script_path}")
                continue

            try:
                subprocess.run(["python", str(script_path)], check=True)
                logging.info(f"✓ Successfully executed: {script_name}")
            except subprocess.CalledProcessError as e:
                logging.error(f"✗ Error executing {script_name}: {e}")
                raise RuntimeError(f"User script failed: {script_name}") from e

        logging.info("Committing results of user scripts to repo...")
        self._commit_and_push("Update experiment state from user scripts")
