import logging
import os
import subprocess
import json
import shutil
from pathlib import Path

from fabric import Connection
from paramiko.config import SSHConfig


class Experiment:
    def __init__(self, experiment_definition):
        # Validate required fields
        if 'datasets' not in experiment_definition or not experiment_definition['datasets']:
            logging.error('No datasets found in experiment definition.')
            raise ValueError("Experiment definition must include datasets.")

        if 'pipelines' not in experiment_definition or not experiment_definition['pipelines']:
            logging.error('No pipelines found in experiment definition.')
            raise ValueError("Experiment definition must include pipelines.")

        self.datasets = experiment_definition['datasets']
        self.pipelines = experiment_definition['pipelines']
        self.extractors = experiment_definition['extractors']
        self.target_host = experiment_definition.get('target_host')
        self.prefix_cmd = experiment_definition.get('prefix_cmd', '') 
        self.scheduler = experiment_definition.get('scheduler', 'slurm')  # Default to slurm if not specified

        logging.info(f'Experiment initialized with datasets: {self.datasets}, pipelines: {self.pipelines}, and extractors: {self.extractors}')
        logging.info(f'Target host: {self.target_host}')
        logging.info(f'Prefix command: {self.prefix_cmd}')
        logging.info(f'Scheduler: {self.scheduler}')

        # Fetch SSH credentials from environment variables
        private_key = os.getenv("SSH_PRIVATE_KEY")
        ssh_config_path = os.getenv("SSH_CONFIG_PATH", "~/.ssh/config")

        if not self.target_host or not private_key:
            logging.error("Missing SSH target host or private key.")
            raise EnvironmentError("SSH target host and private key are required.")

        # Parse SSH config and write private key to the correct location
        self.ssh_key_path = self._setup_ssh_config(self.target_host, private_key, ssh_config_path)

        # Ensure known_hosts is populated for all involved hosts
        self._ensure_known_hosts(self.target_host, ssh_config_path)

        # Establish SSH Connection
        try:
            self.conn, _ = self._create_connection(self.target_host, ssh_config_path)
            logging.info("SSH connection to HPC established successfully.")
        except Exception as e:
            logging.error(f"Failed to establish SSH connection: {e}")
            self.conn = None
            raise ConnectionError("Could not establish SSH connection.")

        # Test SSH connection with a simple command
        logging.info("Running SSH connection test...")
        result = self.conn.run("whoami", hide=True)
        if result.ok:
            logging.info(f"SSH connection test passed: Logged in as {result.stdout.strip()}")
        else:
            logging.error("SSH connection test failed. Check credentials and host configuration.")
            raise ConnectionError("SSH connection test failed.")


    def _setup_ssh_config(self, hostname, private_key, config_path):
        """Parses the SSH config file, extracts IdentityFile, and writes the private key there."""
        config_file = os.path.expanduser(config_path)
        ssh_config = SSHConfig()

        if not os.path.exists(config_file):
            logging.warning(f"SSH config file not found at {config_file}, proceeding without it.")
            return None

        with open(config_file) as f:
            ssh_config.parse(f)

        host_info = ssh_config.lookup(hostname)
        key_path = os.path.expanduser(host_info.get("identityfile", ["~/.ssh/id_rsa"])[0])

        # Write the private key to the file
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, "w") as key_file:
            key_file.write(private_key)
            os.chmod(key_path, 0o600)  # Set correct permissions

        logging.info(f"Private key written to {key_path}")
        return key_path


    def _ensure_known_hosts(self, hostname, config_path):
        """Scans and adds target and proxy hosts to known_hosts to avoid interactive prompts."""
        known_hosts_path = os.path.expanduser("~/.ssh/known_hosts")
        config_file = os.path.expanduser(config_path)
        ssh_config = SSHConfig()

        # Parse SSH config file
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"SSH config file not found at {config_file}")

        with open(config_file) as f:
            ssh_config.parse(f)

        # Collect all unique hosts (target + any proxies)
        host_info = ssh_config.lookup(hostname)
        all_hosts = {host_info.get("hostname", hostname)}

        # Include ProxyJump and ProxyCommand hosts
        proxy_hosts = (
            host_info.get("proxyjump", "").split(",") if "proxyjump" in host_info
            else [host_info["proxycommand"].split()[-1]] if "proxycommand" in host_info
            else []
        )
        for proxy in proxy_hosts:
            proxy_info = ssh_config.lookup(proxy.strip())
            all_hosts.add(proxy_info.get("hostname", proxy.strip()))

        # Scan and add each host to known_hosts if missing
        for host in all_hosts:
            try:
                logging.info(f"Ensuring {host} is in known_hosts...")
                subprocess.run(
                    ["ssh-keyscan", "-H", host],
                    stdout=open(known_hosts_path, "a"),
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to add {host} to known_hosts: {e}")

    def _create_connection(self, hostname, ssh_config_path):
        """Creates a Fabric SSH connection, handling proxies from the config."""
        config_file = os.path.expanduser(ssh_config_path)
        ssh_config = SSHConfig()

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"SSH config file not found at {config_file}")

        with open(config_file) as f:
            ssh_config.parse(f)

        host_info = ssh_config.lookup(hostname)

        conn_kwargs = {
            "host": host_info.get("hostname", hostname),
            "user": host_info.get("user"),
            "port": host_info.get("port", 22),
            "connect_kwargs": {"key_filename": self.ssh_key_path, "allow_agent": True},
        }

        proxy_chain = []
        proxy_hosts = (
            host_info.get("proxyjump", "").split(",") if "proxyjump" in host_info
            else [host_info["proxycommand"].split()[-1]] if "proxycommand" in host_info
            else []
        )

        for proxy_host in proxy_hosts:
            proxy_info = ssh_config.lookup(proxy_host.strip())
            proxy_chain.append(
                Connection(
                    host=proxy_info.get("hostname", proxy_host),
                    user=proxy_info.get("user"),
                    port=proxy_info.get("port", 22),
                    connect_kwargs={"key_filename": self.ssh_key_path},
                )
            )

        if proxy_chain:
            conn_kwargs["gateway"] = proxy_chain[-1]

        return Connection(**conn_kwargs), proxy_chain

    def HPC_logout(self):
        """Closes the SSH connection if it is active."""
        if self.conn and self.conn.is_connected:
            self.conn.close()
            logging.info("SSH connection closed successfully.")
        else:
            logging.warning("SSH connection was already closed or never established.")


    def check_dataset_compliance(self):
        """Checks if all datasets comply with the experiment definition, including both pipelines and extractors."""
        logging.info("Starting dataset compliance check...")

        dataset_paths = self.datasets  # {dataset_name: path}

        # Expected tools by type
        expected_tools = {
            "pipeline": self.pipelines,    # e.g., {"fmriprep": "23.1.3"}
            "extractor": self.extractors   # e.g., {"filecount": "1.0.0"}
        }

        # Where to look for each tool type in global_config
        config_sections = {
            "pipeline": "PROC_PIPELINES",
            "extractor": "EXTRACTION_PIPELINES"
        }

        # Storage for consistency checks
        seen_containers = {}
        seen_invocations = {}

        for dataset_name, dataset_path in dataset_paths.items():
            logging.info(f"Checking dataset: {dataset_name} at {dataset_path}")

            global_config_path = os.path.join(dataset_path, "global_config.json")

            try:
                result = self.conn.run(f"cat {global_config_path}", hide=True)
                global_config = json.loads(result.stdout)
            except Exception as e:
                logging.error(f"Failed to read global_config.json from {dataset_name}: {e}")
                raise

            container_store = global_config["SUBSTITUTIONS"]["[[NIPOPPY_DPATH_CONTAINERS]]"]

            for tool_type, tools in expected_tools.items():
                config_key = config_sections[tool_type]
                tool_list = global_config.get(config_key, [])
                found_versions = {tool["NAME"]: tool["VERSION"] for tool in tool_list}

                for tool_name, expected_version in tools.items():
                    actual_version = found_versions.get(tool_name)

                    # Validate version match
                    if actual_version != expected_version:
                        logging.error(
                            f"Dataset {dataset_name} does not use the expected {tool_type} version: "
                            f"Expected {tool_name}-{expected_version}, Found {actual_version or 'MISSING'}"
                        )
                        raise ValueError("Dataset compliance check failed due to version mismatch.")

                    key = f"{tool_type}:{tool_name}"
                    container_path = os.path.join(container_store, f"{tool_name}_{expected_version}.sif")

                    # Check container only if it's a pipeline or the file exists
                    if tool_type == "pipeline":
                        check_container = True
                    else:
                        # For extractors: check if container exists before inspecting
                        check_container = False
                        try:
                            self.conn.run(f"test -f {container_path}", hide=True)
                            check_container = True
                        except Exception:
                            logging.info(f"No container found for extractor {tool_name}, skipping container check.")

                    if check_container:
                        try:
                            container_info = self.conn.run(f"singularity inspect --json {container_path}", hide=True).stdout
                            if key not in seen_containers:
                                seen_containers[key] = container_info
                            elif seen_containers[key] != container_info:
                                logging.error(f"Inconsistent container detected for {tool_type} {tool_name}.")
                                raise ValueError("Dataset compliance check failed due to container inconsistency.")
                        except Exception as e:
                            logging.error(f"Failed to inspect container {container_path}: {e}")
                            raise

                    # Validate invocation consistency
                    tool_dir = os.path.join(dataset_path, "pipelines", f"{tool_name}-{expected_version}")
                    invocation_path = os.path.join(tool_dir, "invocation.json")
                    try:
                        invocation_content = self.conn.run(f"cat {invocation_path}", hide=True).stdout
                        if key not in seen_invocations:
                            seen_invocations[key] = invocation_content
                        elif seen_invocations[key] != invocation_content:
                            logging.error(f"Inconsistent Boutiques invocation detected for {tool_type} {tool_name}.")
                            raise ValueError("Dataset compliance check failed due to invocation inconsistency.")
                    except Exception as e:
                        logging.error(f"Failed to read {invocation_path} from {dataset_name}: {e}")
                        raise

        logging.info("All datasets comply with the experiment definition.")


    def _run_nipoppy_command(self, action, dataset, dataset_path, pipeline, pipeline_version, use_bash=False):
        """
        General method to construct and execute a Nipoppy command remotely via SSH.
        """
        log_action = {
            "track": "tracker info",
            "run": "pipeline",
            "extract": "extractor"
        }.get(action, action)

        logging.info(f'Running {log_action} for dataset: {dataset} at {dataset_path}, pipeline: {pipeline} ({pipeline_version})')

        # Build the base command
        base_command = f"nipoppy {action} --dataset {dataset_path} --pipeline {pipeline} --pipeline-version {pipeline_version}"
    
        if action in ['run', 'extract']:
            base_command += f" --hpc {self.scheduler} --keep-workdir"

        # Include virtual environment activation
        full_command = f"{self.prefix_cmd} && {base_command}"
        if use_bash:
            full_command = f"bash -l -c '{full_command}'"

        try:
            result = self.conn.run(full_command, hide=True)
            if result.ok:
                logging.info(f"Successfully started {log_action} for {dataset} - {pipeline} ({pipeline_version})")
            else:
                logging.error(f"Failed to start {log_action} for {dataset} - {pipeline} ({pipeline_version})")
                logging.error(result.stderr)
        except Exception as e:
            logging.error(f"Error while running {log_action} for {dataset} - {pipeline}: {e}")


    def update_tracker_info(self, dataset, dataset_path, pipeline, pipeline_version):
        """
        Runs the Nipoppy 'track' command to update the computation status for the dataset.
        """
        self._run_nipoppy_command("track", dataset, dataset_path, pipeline, pipeline_version, use_bash=False)


    def run_pipeline(self, dataset, dataset_path, pipeline, pipeline_version):
        """
        Runs the Nipoppy 'run' command to process the dataset using the specified pipeline.
        """
        self._run_nipoppy_command("run", dataset, dataset_path, pipeline, pipeline_version, use_bash=True)


    def run_extractor(self, dataset, dataset_path, pipeline, pipeline_version):
        """
        Runs the Nipoppy 'extract' command to extract results from the dataset with the given pipeline.
        """
        self._run_nipoppy_command("extract", dataset, dataset_path, pipeline, pipeline_version, use_bash=True)


    def push_state_to_repo(self):
        repo_root = Path(__file__).resolve().parents[1]
        target_dir = repo_root / "experiment_state"
        logging.info(f"Syncing experiment state to local repo at: {target_dir}")

        for dataset_name, dataset_path in self.datasets.items():
            logging.info(f"Processing dataset: {dataset_name} from remote path: {dataset_path}")
            dest_base = target_dir / dataset_name

            if dest_base.exists():
                logging.warning(f"Destination directory already exists. Removing: {dest_base}")
                shutil.rmtree(dest_base)

            dest_base.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created clean destination directory: {dest_base}")

            # Files to copy from HPC
            files_to_fetch = [
                "manifest.tsv",
                "global_config.json",
                "derivatives/imaging_bagel.tsv"
            ]

            for file in files_to_fetch:
                remote_path = f"{dataset_path}/{file}"
                local_path = dest_base / file
                local_path.parent.mkdir(parents=True, exist_ok=True)

                logging.info(f"Downloading file: {remote_path} -> {local_path}")
                try:
                    self.conn.get(remote_path, str(local_path))
                    logging.info(f"Successfully downloaded: {remote_path}")
                except Exception as e:
                    logging.warning(f"Failed to download file: {remote_path} — {e}")

            for pipeline, version in self.pipelines.items():
                pipeline_dir = f"pipelines/{pipeline}-{version}"
                remote_pipeline_dir = f"{dataset_path}/{pipeline_dir}"
                local_pipeline_dir = dest_base / pipeline_dir

                local_pipeline_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Fetching pipeline dir: {remote_pipeline_dir}")
                self._download_directory(remote_pipeline_dir, local_pipeline_dir)

                idp_dir = f"derivatives/{pipeline}/{version}/"
                remote_idp_path = f"{dataset_path}/{idp_dir}"
                local_idp_path = dest_base / idp_dir

                local_idp_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Fetching IDP dir: {remote_idp_path}")
                self._download_directory(remote_idp_path, local_idp_path)

        # Git operations
        logging.info("Running git commit and push...")
        subprocess.run(["git", "config", "user.name", "github_username"])
        subprocess.run(["git", "config", "user.email", "github_email@example.com"])
        subprocess.run(["git", "add", "experiment_state"], check=True)
        subprocess.run(["git", "commit", "-m", "Update experiment state"], check=True)
        subprocess.run(["git", "push"], check=True)
        logging.info("Experiment state pushed to repository successfully.")

    def _download_directory(self, remote_dir, local_dir):
        """Manually iterate through a remote directory and download each file."""
        logging.info(f"Listing contents of remote directory: {remote_dir}")
        try:
            remote_files = self.conn.run(f"ls {remote_dir}", hide=True).stdout.splitlines()
        except Exception as e:
            logging.warning(f"Failed to list directory {remote_dir}: {e}")
            return

        for remote_file in remote_files:
            remote_path = f"{remote_dir}/{remote_file}"
            local_path = local_dir / remote_file

            is_dir = self._is_directory(remote_path)
            if is_dir:
                logging.info(f"Found directory: {remote_path}, descending into it...")
                local_path.mkdir(parents=True, exist_ok=True)
                self._download_directory(remote_path, local_path)
            else:
                logging.info(f"Downloading file: {remote_path} -> {local_path}")
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    self.conn.get(remote_path, str(local_path))
                    logging.info(f"Successfully downloaded: {remote_path}")
                except Exception as e:
                    logging.warning(f"Failed to download file: {remote_path} — {e}")

    def _is_directory(self, remote_path):
        """Check if the given remote path is a directory."""
        try:
            result = self.conn.run(f"test -d {remote_path} && echo 'dir' || echo 'file'", hide=True)
            return result.stdout.strip() == 'dir'
        except Exception as e:
            logging.warning(f"Error checking if remote path is a directory: {remote_path} — {e}")
            return False

'''


    def run_user_processing(self):
        logging.info('Running user-defined processing steps...')
        # Implement user-defined processing logic
        pass
'''
