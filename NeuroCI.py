import requests
import yaml
import json
import sys
import os
from github import Github
from ast import literal_eval

from cbrainAPI import *
from cacheOps import *

##################################################################################

def main(cbrain_token, CCI_token, experiment_definition, cbrain_ids, latest_artifacts_url):

	for dataset in experiment_definition['Datasets']:

		download_cache(dataset  + '.json', CCI_token, latest_artifacts_url)	#Downloads newest cache to json file
		print('Downloaded newest cache for: ' + dataset  + '.json')
		
		update_statuses(dataset  + '.json', cbrain_token)	#Updates the contents of a cache to reflect CBRAIN task statuses
		print('Updated statuses in cache for: ' + dataset  + '.json')
		
		for pipeline in experiment_definition['Pipelines']:
			
			populate_cache_filenames(dataset  + '.json', cbrain_token, experiment_definition['Datasets'][dataset]['Blocklist'], pipeline, cbrain_ids['Data_Provider_IDs'][dataset])	#Populates a cache with any new files found
			print('Populated cache filenames for: ' + dataset  + '.json' + ', ' +  pipeline)
			
			pipeline_manager(cbrain_token, experiment_definition, cbrain_ids, pipeline, dataset)
			print('Posted tasks for: ' + dataset  + '.json' + ', ' +  pipeline)
		
		populate_results(dataset  + '.json', cbrain_token)
		print('Populated results for ' + dataset + '.json')
		#extract_results()
		#analysis(expdef[script])
		update_statuses(dataset  + '.json', cbrain_token)
		print('Updated statuses in cache for: ' + dataset  + '.json')	

##################################################################################

#Obtain login credentials from args, stored in CI environment variables.

cbrain_user = sys.argv[1]
cbrain_password = sys.argv[2]
CCI_token = sys.argv[3]
#CCI_user = sys.argv[4]
#CCI_repo = sys.argv[5]
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

#latest_artifacts_url = "https://circleci.com/api/v1.1/project/github/" + CCI_user + "/" + CCI_repo + "/latest/artifacts"
print("latest artifacts url: " + latest_artifacts_url)

main(cbrain_token, CCI_token, experiment_definition, cbrain_ids, latest_artifacts_url)

print("Finished the scheduled computations")

##################################################################################


