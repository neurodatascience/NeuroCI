#Standalone script for registering files in a CBRAIN data provider
#Run in command line: python3 registrationTool.py username password

import requests
import json
import re
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

def cbrain_list_data_provider(data_provider_ID, cbrain_token):
	
	data_provider_ID = str(data_provider_ID)
	headers = {
		'Accept': 'application/json',
	}
	params = (
		('id', data_provider_ID),
		('cbrain_api_token', cbrain_token),
	)
	url = 'https://portal.cbrain.mcgill.ca/data_providers/' + data_provider_ID + '/browse'
	
	response = requests.get(url, headers=headers, params=params, allow_redirects=True)
	
	if response.status_code == 200:
		return response.json()
	else:
		print('DP browse failure')
		return 1

def cbrain_register(cbrain_token, data_provider_ID, filenames):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    params = (
        ('cbrain_api_token', cbrain_token),
        ('id', data_provider_ID)
    )
    data = {
            "basenames": filenames,
            "filetypes": [".nii.gz"], #can change this for other file types!
            "as_user_id": 0,
            "delete": False
            }
    
    url = "/data_providers/" + str(data_provider_ID) + "/register"
    y = json.dumps(data) #convert data field to JSON:
    response = requests.post('https://portal.cbrain.mcgill.ca' + url, headers=headers, params=params, data=y)
    
    if response.status_code == 200:
        print(response.text)
        jsonResponse = response.json()
        return jsonResponse
    else:
        print("Task posting failed.")
        print(response.content)
        return 

#####################################################################################

cbrain_user = sys.argv[1]
cbrain_password = sys.argv[2]

data_provider_ID = 318 #The data provider ID where the file is in
regex_filename = 'T1w' #The subset of the filename we wish to match, and then register. Can also just be a filename.

filelist = []
file_matches = []

cbrain_token = cbrain_login(cbrain_user, cbrain_password)
data_provider_browse = cbrain_list_data_provider(str(data_provider_ID), cbrain_token) #Query CBRAIN to list all files in data provider.
try:
	for entry in data_provider_browse:
		filelist.append(entry['name'])
except Exception as e:
	print("Error in browsing data provider")
	

file_matches = []
regex_filename = 'T1w' #the thing we wish to find in the filenames

#If we find a matching filename which contains the string.
for filename in filelist:
    if re.search(regex_filename, filename):
        file_matches.append(filename)

#Register all such files.
cbrain_register(cbrain_token, data_provider_ID, file_matches)
