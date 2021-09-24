import datetime
import sys
import time

import yaml

from cacheOps import (
	download_cache,
	populate_cache_filenames,
	update_statuses,
	pipeline_manager,
	populate_results
)
from cbrainAPI import (
	cbrain_login,
	cbrain_logout,
	cbrain_get_all_tasks
)

##################################################################################

def main(cbrain_token, CCI_token, experiment_definition, cbrain_ids, latest_artifacts_url):

	for dataset in experiment_definition['Datasets']:

		download_cache(dataset  + '.json', CCI_token, latest_artifacts_url)	#Downloads newest cache to json file
		print('Downloaded newest cache for: ' + dataset  + '.json')
		
		task_list = cbrain_get_all_tasks(cbrain_token) #Gets the complete list of tasks for the user on CBRAIN
		print('Fetched the list of tasks for the CBRAIN user')
		
		start = time.time()
		update_statuses(dataset  + '.json', task_list)	#Updates the contents of a cache to reflect CBRAIN task statuses
		end = time.time()
		print('Updated statuses in cache for: ' + dataset  + '.json in' + str(datetime.timedelta(seconds=(end - start))))
		
		for pipeline in experiment_definition['Pipelines']:
			
			start = time.time()
			populate_cache_filenames(dataset  + '.json', cbrain_token, experiment_definition['Datasets'][dataset]['Blocklist'], pipeline, cbrain_ids['Data_Provider_IDs'][dataset], experiment_definition)	#Populates a cache with any new files found
			end = time.time()
			print('Populated cache filenames for: ' + dataset  + '.json' + ', ' +  pipeline + " in" + str(datetime.timedelta(seconds=(end - start))))
			
			pipeline_manager(cbrain_token, experiment_definition, cbrain_ids, pipeline, dataset)
			print('Posted tasks for: ' + dataset  + '.json' + ', ' +  pipeline)
		
		populate_results(dataset  + '.json', cbrain_token)
		print('Populated results for ' + dataset + '.json')
		#extract_results()
		#analysis(expdef[script])
		
		#start = time.time()
		#update_statuses(dataset  + '.json', cbrain_token)
		#end = time.time()
		#print('Updated statuses in cache for: ' + dataset  + '.json in' + str(datetime.timedelta(seconds=(end - start))))

##################################################################################

#Obtain login credentials from args, stored in CI environment variables.

cbrain_user = sys.argv[1]
cbrain_password = sys.argv[2]
CCI_token = sys.argv[3]
latest_artifacts_url = sys.argv[4]
cbrain_token = cbrain_login(cbrain_user, cbrain_password)

##################################################################################

#Main code execution section

with open('Experiment_Definition.yaml') as file: #Load experiment definition
	try:
		experiment_definition  = yaml.safe_load(file)
	except yaml.YAMLError as exception: #yaml file not valid
		print('The Experiment Definition file is not valid')
		print(exception)


with open('./Config_Files/CBRAIN_IDs.yaml') as file: #Load mappings for all CBRAIN DP_IDs and toolconfig IDs
	try:
		cbrain_ids  = yaml.safe_load(file)
	except yaml.YAMLError as exception: #yaml file not valid
		print('The configuration file is not valid')
		print(exception)

print("Using artifacts from : " + latest_artifacts_url)

main(cbrain_token, CCI_token, experiment_definition, cbrain_ids, latest_artifacts_url)

print("Finished the scheduled computations")

cbrain_logout(cbrain_token)
##################################################################################


