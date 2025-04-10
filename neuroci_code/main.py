# main.py
import yaml
import logging
from experiment import Experiment

def main(experiment_definition):
    logger.info('Starting experiment')

    this_experiment = Experiment(experiment_definition)

    this_experiment.check_dataset_compliance()

    for dataset, dataset_path in this_experiment.datasets.items():
        for pipeline, pipeline_version in this_experiment.pipelines.items():
            this_experiment.update_tracker_info(dataset, dataset_path, pipeline, pipeline_version)
        for extractor, extractor_version in this_experiment.extractors.items():
            this_experiment.run_extractor(dataset, dataset_path, extractor, extractor_version)
    
    this_experiment.push_state_to_repo()
    this_experiment.run_user_processing()
    this_experiment.HPC_logout()

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load experiment definition
try:
    with open('./experiment_definition.yaml') as file:
        experiment_definition = yaml.safe_load(file)
except yaml.YAMLError as exception:
    logger.error('The Experiment Definition file is not valid')
    raise exception

main(experiment_definition)
