# ssh_utils.py
import logging
import os
import subprocess
import json
from pathlib import Path
from fabric import Connection
from paramiko.config import SSHConfig

class SSHConnectionManager:
    def __init__(self, target_host, prefix_cmd, scheduler):
        self.target_host = target_host
        self.prefix_cmd = prefix_cmd
        self.scheduler = scheduler
        self.conn = None
        self.ssh_key_path = None
        
        self._setup_connection()

    def _setup_connection(self):
        private_key = os.getenv("SSH_PRIVATE_KEY")
        ssh_config_path = os.getenv("SSH_CONFIG_PATH", "~/.ssh/config")

        if not self.target_host or not private_key:
            logging.error("Missing SSH target host or private key.")
            raise EnvironmentError("SSH target host and private key are required.")

        self.ssh_key_path = self._setup_ssh_config(self.target_host, private_key, ssh_config_path)
        self._ensure_known_hosts(self.target_host, ssh_config_path)
        self.conn, _ = self._create_connection(self.target_host, ssh_config_path)
        self._test_connection()

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

        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, "w") as key_file:
            key_file.write(private_key)
            os.chmod(key_path, 0o600)

        logging.info(f"Private key written to {key_path}")
        return key_path

    def _ensure_known_hosts(self, hostname, config_path):
        """Scans and adds target and proxy hosts to known_hosts to avoid interactive prompts."""
        known_hosts_path = os.path.expanduser("~/.ssh/known_hosts")
        config_file = os.path.expanduser(config_path)
        ssh_config = SSHConfig()

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"SSH config file not found at {config_file}")

        with open(config_file) as f:
            ssh_config.parse(f)

        host_info = ssh_config.lookup(hostname)
        all_hosts = {host_info.get("hostname", hostname)}

        proxy_hosts = (
            host_info.get("proxyjump", "").split(",") if "proxyjump" in host_info
            else [host_info["proxycommand"].split()[-1]] if "proxycommand" in host_info
            else []
        )
        
        for proxy in proxy_hosts:
            proxy_info = ssh_config.lookup(proxy.strip())
            all_hosts.add(proxy_info.get("hostname", proxy.strip()))

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

    def _test_connection(self):
        """Tests the SSH connection with a simple command."""
        logging.info("Running SSH connection test...")
        result = self.conn.run("whoami", hide=True)
        if result.ok:
            logging.info(f"SSH connection test passed: Logged in as {result.stdout.strip()}")
        else:
            logging.error("SSH connection test failed. Check credentials and host configuration.")
            raise ConnectionError("SSH connection test failed.")

    def close_connection(self):
        """Closes the SSH connection if it is active."""
        if self.conn and self.conn.is_connected:
            self.conn.close()
            logging.info("SSH connection closed successfully.")
        else:
            logging.warning("SSH connection was already closed or never established.")

    def run_nipoppy_command(self, action, dataset, dataset_path, pipeline, pipeline_version, use_bash=False):
        """General method to construct and execute a Nipoppy command remotely via SSH."""
        log_action = {
            "track": "tracker info",
            "run": "pipeline",
            "extract": "extractor"
        }.get(action, action)

        logging.info(f'Running {log_action} for dataset: {dataset} at {dataset_path}, pipeline: {pipeline} ({pipeline_version})')

        base_command = f"nipoppy {action} --dataset {dataset_path} --pipeline {pipeline} --pipeline-version {pipeline_version}"
    
        if action in ['run', 'extract']:
            base_command += f" --hpc {self.scheduler}"

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

    def check_dataset_compliance(self, datasets, pipelines, extractors):
        """Checks if all datasets comply with the experiment definition."""
        logging.info("Starting dataset compliance check...")

        seen_containers = {}
        seen_invocations = {}

        for dataset_name, dataset_path in datasets.items():
            logging.info(f"Checking dataset: {dataset_name} at {dataset_path}")

            global_config_path = os.path.join(dataset_path, "global_config.json")

            try:
                result = self.conn.run(f"cat {global_config_path}", hide=True)
                global_config = json.loads(result.stdout)
            except Exception as e:
                logging.error(f"Failed to read global_config.json from {dataset_name}: {e}")
                raise

            container_store = global_config["SUBSTITUTIONS"]["[[NIPOPPY_DPATH_CONTAINERS]]"]

            for tool_type, tools in [("pipeline", pipelines), ("extractor", extractors)]:
                config_key = "PROC_PIPELINES" if tool_type == "pipeline" else "EXTRACTION_PIPELINES"
                tool_list = global_config.get(config_key, [])
                found_versions = {tool["NAME"]: tool["VERSION"] for tool in tool_list}

                for tool_name, expected_version in tools.items():
                    actual_version = found_versions.get(tool_name)

                    if actual_version != expected_version:
                        logging.error(
                            f"Dataset {dataset_name} does not use the expected {tool_type} version: "
                            f"Expected {tool_name}-{expected_version}, Found {actual_version or 'MISSING'}"
                        )
                        raise ValueError("Dataset compliance check failed due to version mismatch.")

                    key = f"{tool_type}:{tool_name}"
                    container_path = os.path.join(container_store, f"{tool_name}_{expected_version}.sif")

                    if tool_type == "pipeline":
                        check_container = True
                    else:
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
