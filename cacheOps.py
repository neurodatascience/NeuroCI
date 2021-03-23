import requests
import yaml
import json
import sys
import os
from github import Github
from ast import literal_eval

from cbrainAPI import *

#############################################

'''Downloads newst cache file to json, or if it's not found in the circleCI artifacts, creates a new cache file'''
def download_cache(cache_file, CCI_token, latest_artifacts_url):

	headers = {'Circle-Token': CCI_token}
	response = requests.get(str(latest_artifacts_url), headers=headers)	#finds the link to the cache file amongst all the artifacts
	#example URL for this repo: https://circleci.com/api/v1.1/project/github/jacobsanz97/NDR-CI/latest/artifacts

	link_to_cache = "http://"
	if response.status_code == 200:
		literal_list = literal_eval(response.text) #convert text to dictionary so we can browse it
		for file in literal_list:
			if cache_file in file['url']:
				link_to_cache = file['url'] 
	else:
		print("Error loading CircleCI artifacts")
		print(response.text)
	
	response = requests.get(link_to_cache, headers=headers)	#download the cache file to json
	if response.status_code == 200:
		json_cache = json.loads(response.text)
	else:	#Cache file couldn't be loaded, so we create an empty json
		print(response.text)
		json_cache = json.loads("{}")
		print("Cache file not initialized...Creating a new one.")

	with open(cache_file, 'w') as outfile: #create cache file for CI
		json.dump(json_cache, outfile)
	print('written cache to temp file')


'''Creates a template for a cache entry (cbrain data provider file), for a specific pipeline. Provides a userfile ID as a starting point for task computations'''
def generate_cache_subject(nifti_file, cbrain_userfile_ID, pipeline, experiment_definition):

	data = { nifti_file: {
		pipeline: {}}}
		
	result = {"result": None, "isUsed": None}
	
	component_number = 0 #Keeps track of the order of the component (we need to flag the first one)
	for pipeline_component in experiment_definition['Pipelines'][pipeline]['Components']:

		if component_number == 0:
			component_record = {
					"inputID": cbrain_userfile_ID, #only do this for first component
					"toolConfigID": None,
					"taskID": None,
					"status": None,
					"outputID": None,
					"isUsed": None
			}
		else:
			component_record = {
					"inputID": None,
					"toolConfigID": None,
					"taskID": None,
					"status": None,
					"outputID": None,
					"isUsed": None
			}			
		
		data[nifti_file][pipeline][pipeline_component] = component_record	#add this component to the cache
		component_number = component_number + 1
		
	data[nifti_file][pipeline]['Result'] = result	#add the results section after all the component sections
	return data


'''Generates the template for every file in a cache, for a specific pipeline'''
def populate_cache_filenames(cache_file, cbrain_token, blocklist, pipeline, data_provider_id, experiment_definition):

	filelist = []
	data_provider_browse = cbrain_list_data_provider(str(data_provider_id), cbrain_token) #Query CBRAIN to list all files in data provider.
	
	for entry in data_provider_browse:
		if 'userfile_id' in entry: #if it's a registered file, add to filelist.
			filelist.append([entry['name'], entry['userfile_id']])
			
	with open(cache_file, "r+") as file:
		data = json.load(file)
		for entry in filelist:
			if entry[0] not in data and entry[0] not in blocklist:	#if entry[name] is not in cache AND is not in the blocklist...add to cache
				leaf = generate_cache_subject(entry[0], entry[1], pipeline, experiment_definition)
				data.update(leaf)
		
		file.seek(0)	# rewind
		json.dump(data, file, indent=2)
		file.truncate() 
		return data



'''Updates a cache file with the newest task statuses from CBRAIN'''
def update_statuses(cache_filename, cbrain_token):
	
	with open(cache_filename, "r+") as cache_file:
		data = json.load(cache_file)
		for (file, pipeline) in data.items(): #Parse the json
			for (pipeline_name, task_name) in pipeline.items():
				for (task_name_str, params) in task_name.items():
					
					if task_name_str != "Result" and params["taskID"] != None: #If this is a task (not a result) with an existent ID on CBRAIN...
						jayson = cbrain_get_task_info(cbrain_token, str(params["taskID"])) #Query CBRAIN for the task info
						
						if jayson['status'] == "Completed":
							#Task completed, update status and get output file ID
							data[file][pipeline_name][task_name_str]["status"] = jayson["status"]
							#differentiate between one and many outputs
							if '_cbrain_output_outputs' in jayson['params']:
								data[file][pipeline_name][task_name_str]["outputID"] = jayson['params']['_cbrain_output_outputs'][0]
							if '_cbrain_output_output' in jayson['params']:
								data[file][pipeline_name][task_name_str]["outputID"] = jayson['params']['_cbrain_output_output'][0]
							if '_cbrain_output_outfile' in jayson['params']:
								data[file][pipeline_name][task_name_str]["outputID"] = jayson['params']['_cbrain_output_outfile'][0]
								
						else:
							#Task not completed, just update status
							data[file][pipeline_name][task_name_str]["status"] = jayson["status"]
		cache_file.seek(0)
		json.dump(data, cacheFile, indent=2)
		cache_file.truncate()


'''Iterates over each component in a pipelines, organizes, and feeds the necessary data to the functions which post tasks on CBRAIN and update the caches'''
def pipeline_manager(cbrain_token, experiment_definition, cbrain_ids, pipeline, dataset):

	component_number = 0 #Keeps track of the order of the component (we need to flag the first one)
	
	for pipeline_component in experiment_definition['Pipelines'][pipeline]['Components']:
		
		with open(experiment_definition['Pipelines']['Components'][pipeline_component]['Parameter_dictionary'], "r+") as param_file:	#Load parameters for current pipeline component
			parameter_dictionary = json.load(param_file)
			
			if component_number == 0:										
				first_task_handler(cbrain_token, parameter_dictionary, cbrain_ids['Tool_Config_IDs'][pipeline_component], dataset + '.json', pipeline_component, pipeline)
			else:
				nth_task_handler(cbrain_token, parameter_dictionary, cbrain_ids['Tool_Config_IDs'][pipeline_component], dataset + '.json', pipeline_component, previous_pipeline_component, pipeline)

		previous_pipeline_component = pipeline_component
		component_number = component_number + 1


'''Handles the cache writing for the first task in a pipeline, and calls to post the task to CBRAIN'''
def first_task_handler(cbrain_token, parameter_dictionary, tool_config_id, cache_file, pipeline_component, pipeline_name):
	
	with open(cache_file, "r+") as file:
		data = json.load(file)
		for filename in data:
			if data[filename][pipeline_name][pipeline_component]['isUsed'] == None:
				
				userfile_id = data[filename][pipeline][pipeline_component]['inputID']
				jayson = cbrain_post_task(cbrain_token, userfile_id, tool_config_id, parameter_dictionary)
				data[filename][pipeline_name][pipeline_component]['toolConfigID'] = jayson[0]['tool_config_id']
				data[filename][pipeline_name][pipeline_component]['taskID'] = jayson[0]["id"]
				data[filename][pipeline_name][pipeline_component]['status'] = jayson[0]["status"]
				data[filename][pipeline_name][pipeline_component]['isUsed'] = True
				
		file.seek(0)	# rewind
		json.dump(data, file, indent=2)
		file.truncate()


'''Handles the cache writing and task posting for any pipeline component except the first task'''
def nth_task_handler(cbrain_token, parameter_dictionary, tool_config_id, cache_file, pipeline_component, previous_pipeline_component, pipeline_name):	
		
	with open(cache_file, "r+") as file:
		data = json.load(file)
		for filename in data:
			if data[filename][pipeline_name][pipeline_component]['isUsed'] == None and data[file][pipeline_name][previous_pipeline_component]['status'] == "Completed":
				
				userfile_id = data[filename][pipeline][previous_pipeline_component]['outputID']	#output of last task
				jayson = cbrain_post_task(cbrain_token, userfile_id, tool_config_id, parameter_dictionary)
				data[filename][pipeline_name][pipeline_component]['inputID'] = userfile_id
				data[filename][pipeline_name][pipeline_component]['toolConfigID'] = jayson[0]['tool_config_id']
				data[filename][pipeline_name][pipeline_component]['taskID'] = jayson[0]["id"]
				data[filename][pipeline_name][pipeline_component]['status'] = jayson[0]["status"]
				data[filename][pipeline_name][pipeline_component]['isUsed'] = True
				
		file.seek(0)	# rewind
		json.dump(data, file, indent=2)
		file.truncate()


'''Fetches the text from a file on CBRAIN and writes it to the cache. Originally this designed for extracting a hippocampal volume from an FSL Stats text output'''
def populateResults(cache_filename, cbrain_token):
	
	with open(cache_filename, "r+") as cache_file:
		data = json.load(cache_file)
		for (file, pipeline) in data.items():
			for (pipeline_name, pipeline_component) in pipeline.items():
				previous_string = None
				for (pipeline_component_str, params) in pipeline_component.items():
					
					if pipeline_component_str == "Result": #Find the task before the result in the json
						if data[file][pipeline_name]['Result']['isUsed'] == None and data[file][pipeline_name][previous_string]['status'] == "Completed":
							fileID = data[file][pipeline_name][previous_string]['outputID']
							vol = cbrain_download_text(fileID, cbrain_token)
							data[file][pipeline_name]['Result']['result'] = vol
							data[file][pipeline_name]['Result']['isUsed'] = True
					previous_string = pipeline_component_str
					
		cache_file.seek(0)	# rewind
		json.dump(data, cache_file, indent=2)
		cache_file.truncate()

