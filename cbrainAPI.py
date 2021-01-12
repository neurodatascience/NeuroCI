import requests
import json
import sys
import os

##################################################################################

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
    


def cbrain_logout(token):
    headers = {
        'Accept': 'application/json',
    }
    params = (
        ('cbrain_api_token', token),
    )
    response = requests.delete('https://portal.cbrain.mcgill.ca/session', headers=headers, params=params)
    if response.status_code == 200:
        print("Logout success")
        return 0
    else:
        print("Logout failure")
        return 1
   


def cbrain_listDP(dataprovider_ID, token):
    dataprovider_ID = str(dataprovider_ID)
    headers = {
        'Accept': 'application/json',
    }
    params = (
        ('id', dataprovider_ID),
        ('cbrain_api_token', token),
    )
    url = 'https://portal.cbrain.mcgill.ca/data_providers/' + dataprovider_ID + '/browse'
    response = requests.get(url, headers=headers, params=params, allow_redirects=True)
    if response.status_code == 200:
        return response.json()
    else:
        print('DP browse failure')
        return 1


    
def cbrain_FSLFirst(token, fileID):
    fileID = str(fileID)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    params = (
        ('cbrain_api_token', token),
    )
    data = {
      "cbrain_task": {
        'tool_config_id': 721,
        'params': {
          'interface_userfile_ids': [fileID], 
          'input_file': fileID, 
          'prefix': 'output', 
          'brain_extracted': False, 
          'three_stage': False, 
          'verbose': False       
        }, 
        'run_number': None, 
        'results_data_provider_id': 179, 
        'cluster_workdir_size': None, 
        'workdir_archived': True,  
        'description': ''}
     }
    # convert into JSON:
    y = json.dumps(data)
    response = requests.post('https://portal.cbrain.mcgill.ca/tasks', headers=headers, params=params, data=y)
    if response.status_code == 200:
        print(response.text)
        jsonResponse = response.json()
        return jsonResponse
    else:
        print("Task posting failed.")
        return 1



def cbrain_FSLStats(token, fileID):
    fileID = str(fileID)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    params = (
        ('cbrain_api_token', token),
    )
    data = {
      "cbrain_task": {
        "tool_config_id": 1698,
        "params": {
          "interface_userfile_ids": [
            fileID
          ],
          "input_file": fileID,
          "t": False,
          "l": "16.5",
          "u": "17.5",
          "a": False,
          "n": False,
          "r": False,
          "R": False,
          "e": False,
          "E": False,
          "v": False,
          "V": True,
          "m": False,
          "M": False,
          "s": False,
          "S": False,
          "w": False,
          "x": False,
          "X": False,
          "c": False,
          "C": False},
          "run_number": None,
          "results_data_provider_id": 27,
          "cluster_workdir_size": 40960,
          "workdir_archived": False,
         "description": ""
      }
    }
    # convert into JSON:
    y = json.dumps(data)
    response = requests.post('https://portal.cbrain.mcgill.ca/tasks', headers=headers, params=params, data=y)
    if response.status_code == 200:
        print(response.text)
        jsonResponse = response.json()
        return jsonResponse
    else:
        print("Task posting failed.")
        return 1
    


def cbrain_getTaskInfo(token, taskID):
    taskID = str(taskID)
    headers = {
        'Accept': 'application/json',
    }
    params = (
        ('id', taskID),
        ('cbrain_api_token', token)
    )
    url = 'https://portal.cbrain.mcgill.ca/tasks/' + taskID
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        jsonResponse = response.json()
        return jsonResponse
    else:
        print("Task Info retrieval failed.")
        return 1



def cbrain_download_text(fileID, token):
    fileID = str(fileID)
    headers = {
        'Accept': 'text',
    }
    params = (
        ('cbrain_api_token', token),
    )
    url = 'https://portal.cbrain.mcgill.ca/userfiles/' + fileID + '/content'
    response = requests.get(url, headers=headers, params=params, allow_redirects=True)
    if response.status_code == 200:
        return response.text
    else:
        print('Download failure')
        return 1



def cbrain_SubfolderFileExtractor(token, fileID, filenameToExtract, fileNewName):
    #note that filenameToExtract should include the extension
    fileID = str(fileID)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    params = (
        ('cbrain_api_token', token),
    )
    data = {
      "cbrain_task": { 
        'tool_config_id': 2094,
        'params': {
          'interface_userfile_ids': [fileID],
          'infolder': fileID,
          'extracted': filenameToExtract,
          'new_name': fileNewName},
        'run_number': None, 
        'results_data_provider_id': 179, 
        'cluster_workdir_size': None, 
        'workdir_archived': True, 
        'description': ''}
     }
    # convert into JSON:
    y = json.dumps(data)
    response = requests.post('https://portal.cbrain.mcgill.ca/tasks', headers=headers, params=params, data=y)
    if response.status_code == 200:
        print(response.text)
        jsonResponse = response.json()
        return jsonResponse
    else:
        print("Task posting failed.")
        return 1
