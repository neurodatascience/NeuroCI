import logging
import os
import subprocess
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
        self.prefix_cmd = experiment_definition.get('prefix_cmd', '')  
        self.state_backup = experiment_definition.get('state_backup', None)  

        logging.info(f'Experiment initialized with datasets: {self.datasets} and pipelines: {self.pipelines}.')
        logging.info(f'Prefix command: {self.prefix_cmd}')
        logging.info(f'State backup: {self.state_backup}')

        # Fetch SSH credentials from environment variables
        self.target_host = os.getenv("SSH_TARGET_HOST")
        private_key = os.getenv("SSH_PRIVATE_KEY")
        ssh_config_path = os.getenv("SSH_CONFIG_PATH", "~/.ssh/config")

        if not self.target_host or not private_key:
            logging.error("Missing SSH target host or private key in environment variables.")
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
            
'''
    def check_dataset_compliance(self):
        logging.info('Checking dataset compliance...')
        # Implement dataset compliance logic here
        pass

    def update_tracker_info(self, dataset, dataset_path, pipeline, pipeline_version):
        logging.info(f'Updating tracker info for dataset: {dataset} at {dataset_path}, pipeline: {pipeline} ({pipeline_version})')
        # Implement tracker info update logic here
        pass

    def backup_state_on_hpc(self, dataset, dataset_path, pipeline, pipeline_version):
        logging.info(f'Backing up state on HPC for dataset: {dataset} at {dataset_path}, pipeline: {pipeline} ({pipeline_version})')
        # Logic to back up state to HPC
        # Example: Save to a specific folder defined in the experiment definition
        hpc_backup_path = os.path.join(self.container_store, dataset, pipeline)
        os.makedirs(hpc_backup_path, exist_ok=True)
        # Mock example of copying a state file to backup
        state_file = os.path.join(dataset_path, f"{pipeline}_state.txt")
        backup_file = os.path.join(hpc_backup_path, f"{pipeline}_state_backup.txt")
        if os.path.exists(state_file):
            shutil.copy(state_file, backup_file)
        else:
            logging.warning(f'State file for {dataset}/{pipeline} not found for backup.')

    def fetch_state_from_HPC(self, dataset, dataset_path, pipeline, pipeline_version):
        logging.info(f'Fetching state from HPC for dataset: {dataset} at {dataset_path}, pipeline: {pipeline} ({pipeline_version})')
        # Logic to fetch state from HPC to workspace
        pass

    def push_state_to_repo(self, dataset, dataset_path, pipeline, pipeline_version):
        logging.info(f'Pushing state to repo for dataset: {dataset} at {dataset_path}, pipeline: {pipeline} ({pipeline_version})')
        # Logic to push state to repository
        pass

    def run_pipelines_on_datasets(self, dataset, dataset_path, pipeline, pipeline_version):
        logging.info(f'Running pipeline: {pipeline} ({pipeline_version}) on dataset: {dataset} at {dataset_path}')
        # Implement logic to run pipelines
        pass

    def extract_from_derivatives(self, dataset, dataset_path, pipeline, pipeline_version):
        logging.info(f'Extracting derivatives for dataset: {dataset} at {dataset_path}, pipeline: {pipeline} ({pipeline_version})')
        # Implement logic to extract derivatives
        pass

    def run_user_processing(self):
        logging.info('Running user-defined processing steps...')
        # Implement user-defined processing logic
        pass
'''
