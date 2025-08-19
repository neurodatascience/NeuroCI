import logging
import subprocess
import shutil
import json
import csv
import os
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
        Also saves container metadata using `singularity inspect --json`.

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


                manifest_path = f"{dataset_path}/manifest.tsv"
                tracker_path = f"{dataset_path}/pipelines/processing/{tool}-{version}/tracker.json"
                file_paths_to_download = self._resolve_tracker_paths(conn, manifest_path, tracker_path, dataset_path, tool, version)

                # Prepare local tarball path
                local_tar_path = Path("/tmp") / "neuroci_output_state" / dataset_name / f"{tool}_{version}_output.tar.gz"

                # Download selected files as a tarball
                self._download_tarball_from_remote_dir(
                    conn,
                    remote_base=dataset_path,
                    local_tar_path=local_tar_path,
                    remote_tar_name=f"/tmp/{tool}_{version}_output.tar.gz",
                    file_paths_to_download=file_paths_to_download
                )

            # Save Singularity container inspection output
            container_dir = dest_base / "containers"
            container_dir.mkdir(parents=True, exist_ok=True)

            global_config_path = f"{dataset_path}/global_config.json"
            try:
                result = conn.run(f"cat {global_config_path}", hide=True)
                global_config = json.loads(result.stdout)
                container_store = global_config["SUBSTITUTIONS"]["[[NIPOPPY_DPATH_CONTAINERS]]"]
            except Exception as e:
                logging.warning(f"Failed to read container store from {global_config_path}: {e}")
                container_store = None

            if container_store:
                for tool, version in pipelines.items():
                    container_path = os.path.join(container_store, f"{tool}_{version}.sif")
                    local_json_path = container_dir / f"{tool}_{version}.json"
                    try:
                        logging.info(f"Inspecting container: {container_path}")
                        inspect_result = conn.run(f"singularity inspect --json {container_path}", hide=True)
                        with open(local_json_path, "w") as f:
                            f.write(inspect_result.stdout)
                        logging.info(f"✓ Saved container metadata for {tool}-{version}")
                    except Exception as e:
                        logging.warning(f"✗ Failed to inspect container {container_path}: {e}")

        # Commit and push all the newly downloaded data to Git
        self._commit_and_push("Update experiment state")


    def _resolve_tracker_paths(self, conn, manifest_path, tracker_path, dataset_path, tool, version):
        """
        Reads a manifest.tsv and a tracker.json directly from the HPC via Fabric,
        and returns a list of file paths with placeholders substituted.
        """

        # Read manifest remotely
        result = conn.run(f"cat {manifest_path}", hide=True)
        reader = csv.DictReader(result.stdout.splitlines(), delimiter="\t")
        participants = [
            {"participant_id": row["participant_id"], "session_id": row["session_id"]}
            for row in reader
        ]

        # Read tracker remotely
        result = conn.run(f"cat {tracker_path}", hide=True)
        tracker = json.loads(result.stdout)
        paths_template = tracker.get("PATHS", [])

        # Resolve paths
        resolved_paths = []
        for p in participants:
            sub_id = f"sub-{p['participant_id']}"
            ses_id = f"ses-{p['session_id']}"
            for t_path in paths_template:
                path = t_path.replace("[[NIPOPPY_BIDS_PARTICIPANT_ID]]", sub_id)
                path = path.replace("[[NIPOPPY_BIDS_SESSION_ID]]", ses_id)

                # Prepend dataset derivatives path
                full_path = os.path.join(dataset_path, "derivatives", tool, version, "output", path)
                resolved_paths.append(full_path)

        return resolved_paths


    def _download_tarball_from_remote_dir(self, conn, remote_base, local_tar_path, remote_tar_name, file_paths_to_download):
        """
        Archives and downloads selected files from a remote directory as a tar.gz file,
        then extracts it locally and deletes the archive.

        Args:
            conn: SSH connection object.
            remote_base: Base directory relative to which tar should archive.
            local_tar_path: Local file path to save the downloaded tar.gz.
            remote_tar_name: Remote file path for temporary tar.gz (e.g., /tmp/foo_v1_output.tar.gz).
            file_paths_to_download: List of file paths (relative to remote_base) to include in tarball.
        """
        import shlex
        local_tar_path.parent.mkdir(parents=True, exist_ok=True)

        # Trim paths to start from derivatives/tool/... for clean extraction
        def relative_for_tar(path):
            parts = path.split('/')
            if 'derivatives' in parts:
                idx = parts.index('derivatives')
                return '/'.join(parts[idx:])
            return path

        tar_paths = [relative_for_tar(p) for p in file_paths_to_download]

        # Quote each path to handle spaces/special characters
        quoted_paths = " ".join(shlex.quote(p) for p in tar_paths)

        try:
            logging.info(f"Creating tarball on remote: {remote_tar_name} with selected files")
            conn.run( f"tar -czf {shlex.quote(remote_tar_name)} --ignore-failed-read -C {shlex.quote(remote_base)} {quoted_paths}", hide=True )

            # Check tarball size
            result = conn.run(f"stat -c %s {shlex.quote(remote_tar_name)}", hide=True)
            tar_size_bytes = int(result.stdout.strip())
            max_tar_size_mb = 25
            if tar_size_bytes > max_tar_size_mb * 1024 * 1024:
                size_mb = tar_size_bytes / (1024 * 1024)
                conn.run(f"rm -f {shlex.quote(remote_tar_name)}", hide=True)
                raise ValueError(f"Tarball too large: {size_mb:.2f} MB. Reduce size below {max_tar_size_mb} MB.")

            logging.info(f"Downloading tarball: {remote_tar_name} -> {local_tar_path}")
            conn.get(remote_tar_name, str(local_tar_path))
            conn.run(f"rm -f {shlex.quote(remote_tar_name)}", hide=True)
            logging.info(f"✓ Downloaded and cleaned up tarball")

            # Extract
            extract_dir = local_tar_path.parent
            logging.info(f"Extracting tarball: {local_tar_path} -> {extract_dir}")
            shutil.unpack_archive(str(local_tar_path), extract_dir)
            logging.info(f"✓ Extracted to: {extract_dir}")

            # Delete local archive
            local_tar_path.unlink()
            logging.info(f"✓ Deleted archive: {local_tar_path}")

        except Exception as e:
            logging.warning(f"✗ Failed to handle tarball: {e}")


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
        state_dir = Path("/tmp") / "neuroci_output_state"
        subprocess.run(["tree", str(state_dir)])

        if not (state_dir.exists() and any(state_dir.iterdir())):
            logging.info("Skipping user scripts: /tmp/neuroci_output_state does not exist or is empty.")
            return

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
