# experiment.py
import logging
from pathlib import Path
from ssh_utils import SSHConnectionManager
from file_utils import FileOperations

class Experiment:
    def __init__(self, experiment_definition):
        self._validate_experiment_definition(experiment_definition)
        
        self.datasets = experiment_definition['datasets']
        self.pipelines = experiment_definition['pipelines']
        self.extractors = experiment_definition['extractors']
        self.userscripts = experiment_definition['userscripts']
        self.target_host = experiment_definition.get('target_host')
        self.prefix_cmd = experiment_definition.get('prefix_cmd', '')
        self.scheduler = experiment_definition.get('scheduler', 'slurm')
        
        self._log_experiment_config()
        
        self.ssh_manager = SSHConnectionManager(
            target_host=self.target_host,
            prefix_cmd=self.prefix_cmd,
            scheduler=self.scheduler
        )
        self.file_ops = FileOperations()

    def _validate_experiment_definition(self, experiment_definition):
        if 'datasets' not in experiment_definition or not experiment_definition['datasets']:
            logging.error('No datasets found in experiment definition.')
            raise ValueError("Experiment definition must include datasets.")

        if 'pipelines' not in experiment_definition or not experiment_definition['pipelines']:
            logging.error('No pipelines found in experiment definition.')
            raise ValueError("Experiment definition must include pipelines.")

    def _log_experiment_config(self):
        logging.info(f'Experiment initialized with datasets: {self.datasets}, pipelines: {self.pipelines}')
        logging.info(f'Extractors: {self.extractors}')
        logging.info(f'User scripts: {self.userscripts}')
        logging.info(f'Target host: {self.target_host}')
        logging.info(f'Prefix command: {self.prefix_cmd}')
        logging.info(f'Scheduler: {self.scheduler}')

    def check_dataset_compliance(self):
        self.ssh_manager.check_dataset_compliance(
            datasets=self.datasets,
            pipelines=self.pipelines,
            extractors=self.extractors
        )

    def update_tracker_info(self, dataset, dataset_path, pipeline, pipeline_version):
        self.ssh_manager.run_nipoppy_command(
            action="track",
            dataset=dataset,
            dataset_path=dataset_path,
            pipeline=pipeline,
            pipeline_version=pipeline_version
        )

    def run_pipeline(self, dataset, dataset_path, pipeline, pipeline_version):
        self.ssh_manager.run_nipoppy_command(
            action="run",
            dataset=dataset,
            dataset_path=dataset_path,
            pipeline=pipeline,
            pipeline_version=pipeline_version,
            use_bash=True
        )

    def run_extractor(self, dataset, dataset_path, pipeline, pipeline_version):
        self.ssh_manager.run_nipoppy_command(
            action="extract",
            dataset=dataset,
            dataset_path=dataset_path,
            pipeline=pipeline,
            pipeline_version=pipeline_version,
            use_bash=True
        )

    def push_state_to_repo(self):
        self.file_ops.push_state_to_repo(
            conn=self.ssh_manager.conn,
            datasets=self.datasets,
            pipelines=self.pipelines,
            extractors=self.extractors
        )

    def run_user_processing(self):
        self.file_ops.run_user_scripts(self.userscripts)

    def HPC_logout(self):
        self.ssh_manager.close_connection()
