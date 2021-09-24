import argparse
import os

import cacheOps
import cbrainAPI
import neuroCIdata
import utils

##################################################################################

_json_extension = "json"


def run(args, cbrain_api, cbrain_ids, experiment_definition):

    for dataset in experiment_definition['Datasets']:

        # Downloads newest cache to json file
        dataset_filename = os.path.extsep.join(dataset, _json_extension)
        cacheOps.download_cache(
            dataset_filename, args.CCI_token, args.artifacts_url)
        print(f'Downloaded newest cache for: {dataset_filename}')
        # Gets the complete list of tasks for the user on CBRAIN
        task_list = cbrain_api.get_all_tasks(args.cbrain_token)
        print('Fetched the list of tasks for the CBRAIN user')

        # Updates the contents of a cache to reflect CBRAIN task statuses
        with utils.measure_time() as execution_time:
            cacheOps.update_statuses(dataset_filename, task_list)

        print((f"Updated statuses in cache for: "
               f"{dataset_filename} in {execution_time()}"))

        for pipeline in experiment_definition['Pipelines']:

            with utils.measure_time() as execution_time:
                # Populates a cache with any new files found
                blocklist = experiment_definition['Datasets'][dataset]['Blocklist']
                cbrain_dataset = cbrain_ids['Data_Provider_IDs'][dataset]
                cacheOps.populate_cache_filenames(dataset_filename,
                                                  args.cbrain_token,
                                                  blocklist,
                                                  pipeline,
                                                  cbrain_dataset,
                                                  experiment_definition)

            print((f"Populated cache filenames for: "
                   f"{dataset_filename}, {pipeline} in {execution_time()}"))
            cacheOps.pipeline_manager(args.cbrain_token, experiment_definition,
                                      cbrain_ids, pipeline, dataset)
            print(f'Posted tasks for: {dataset_filename}, {pipeline}')

        cacheOps.populate_results(cbrain_api, dataset_filename)
        print(f'Populated results for {dataset_filename}')
  # extract_results()
# analysis(expdef[script])

# start = time.time()
# update_statuses(dataset  + '.json', cbrain_token)
# end = time.time()
# print('Updated statuses in cache for: ' + dataset  + '.json in' + str(datetime.timedelta(seconds=(end - start))))

##################################################################################

# Obtain login credentials from args, stored in CI environment variables.


def parse_args():

    parser = argparse.ArgumentParser('NeuroCI', help='to add...')
    parser.add_argument('--cbrain-user', help="CBRAIN user", required=True)
    parser.add_argument('--cbrain-password',
                        help="CBRAIN password", required=True)
    parser.add_argument('--CCI-token', help="CCI_token", required=True)
    parser.add_argument('--artifacts-url', help="artifacts-url", required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cbrain_api = cbrainAPI.CbrainAPI(args.cbrain_user, args.cbrain_password)

    ##################################################################################

    # Main code execution section

    experiment_definition = utils.get_yaml_file(
        neuroCIdata.experiment_definition_path, "Experiment Definition")

    # Load mappings for all CBRAIN DP_IDs and toolconfig IDs
    cbrain_ids = utils.get_yaml_file(
        neuroCIdata.cbrain_ids_path, "Configuration")

    print(f"Using artifacts from : {args.artifacts_url}")

    run(args, cbrain_api, cbrain_ids, experiment_definition)

    print("Finished the scheduled computations")

    ##################################################################################
