#This script is a 'panic button' that will iterate through all the uncompleted tasks in a cache file and terminate them on CBRAIN. I believe only tasks 'on CPU' can be terminated, so you might have to run it a couple of times. Run it with your cbrain username and password as input in the terminal.

import json
import requests
import sys


def cbrain_login(username, password):
	
	headers = {
		'Content-Type': 'application/x-www-form-urlencoded',
		'Accept': 'application/json',
	}
	data = {
	'login': username,
	'password': password
	}
	
	response = requests.post('https://portal.cbrain.mcgill.ca/session', headers=headers, data=data)
	
	if response.status_code == 200:
		print("Login success")
		print(response.content)
		jsonResponse = response.json()
		return jsonResponse["cbrain_api_token"]
	else:
		print("Login failure")
		return 1



def cbrain_terminate_task(task_ID_list, cbrain_token):

	headers = {
		'Content-Type': 'application/json',
		'Accept': 'application/json',
	}
	params = (
		('cbrain_api_token', cbrain_token),
	)
	data = {
			"operation": "terminate",
			"tasklist": task_ID_list
			}

	y = json.dumps(data) #convert data field to JSON:
	response = requests.post('https://portal.cbrain.mcgill.ca/tasks/operation', headers=headers, params=params, data=y, allow_redirects=False)

	if response.status_code == 200:
		print("Terminated task(s): " + str(task_ID_list))   
		return 0
	else:
		print("Task termination failed.")
		print(response.status_code)
		return 1   



def terminate_all_tasks(cache_filename, cbrain_token):

	list_of_IDs = []
	with open(cache_filename, "r+") as cache_file:
		data = json.load(cache_file)
		for (file, pipeline) in data.items():
			for (pipeline_name, pipeline_component) in pipeline.items():
				for (pipeline_component_str, params) in pipeline_component.items():
					

					if 'taskID' in params.keys() and params['status'] != 'Completed':
						list_of_IDs.append(str(params['taskID']))
					else:
						break

	print("Terminating " + str(list_of_IDs))
	cbrain_terminate_task(list_of_IDs, cbrain_token)

#################################################################################################
cbrain_user = sys.argv[1]
cbrain_password = sys.argv[2]
cbrain_token = cbrain_login(cbrain_user, cbrain_password)

cache_file_path = 'Prevent-AD-sofar.json'
terminate_all_tasks(cache_file_path, 'tok')
