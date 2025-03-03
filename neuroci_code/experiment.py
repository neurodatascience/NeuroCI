import logging
import os
import shutil
from fabric import Connection

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
        self.prefix_cmd = experiment_definition.get('prefix_cmd', '')  # Defaults to empty string
        self.state_backup = experiment_definition.get('state_backup', None)  # Defaults to None

        logging.info(f'Experiment initialized with datasets: {self.datasets} and pipelines: {self.pipelines}.')
        logging.info(f'Prefix command: {self.prefix_cmd}')
        logging.info(f'State backup: {self.state_backup}')

        # Fetch credentials from environment variables
        self.host = os.getenv("SSH_HOST")
        self.username = os.getenv("SSH_USER")
        ssh_password = os.getenv("SSH_PASSWORD")  # Fetch but do not store

        if not self.host or not self.username or not ssh_password:
            logging.error("Missing SSH credentials in environment variables.")
            raise EnvironmentError("SSH credentials are required.")

        # Establish SSH Connection
        try:
            self.conn = Connection(host=self.host, user=self.username, connect_kwargs={"password": ssh_password})
            logging.info("SSH connection to HPC established successfully.")
        except Exception as e:
            logging.error(f"Failed to establish SSH connection: {e}")
            self.conn = None
            raise ConnectionError("Could not establish SSH connection.")


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
