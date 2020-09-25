import requests
import json
import sys
import os
from github import Github

from cbrainAPI import *

#############################################3

def generateCacheSubject(niftiFile, cbrain_userfileID): #inputs are string with filename, and int
    data = { niftiFile: {
          "FSL": { #For Future expansion to more pipelines...take 'FSL' as fn input?
            "Task1": {
              "inputID": cbrain_userfileID,
              "toolConfigID": None,
              "taskID": None,
              "status": None,
              "outputID": None,
              "isUsed": None
            },
            "Task2": {
              "inputID": None,
              "toolConfigID": None,
              "taskID": None,
              "status": None,
              "outputID": None,
              "isUsed": None
            },
            "Result": {
              "volume": None,
              "isUsed": None
          }
        }
      }}
    return data



def populateCacheFilenames(filename, token):
    #Appends the files from the DP that are not already in the json to the json
    filelist = []
    dpBrowse = cbrain_listDP('318', token)
    for entry in dpBrowse:
        if 'userfile_id' in entry: #and if key not in json? see comment below
            filelist.append([entry['name'],entry['userfile_id']])
    with open(filename, "r+") as file:
        data = json.load(file)
        for entry in filelist:
            #if entry[name] is not in json...maybe update already does this?
            leaf = generateCacheSubject(entry[0], entry[1])
            data.update(leaf)
        file.seek(0)  # rewind
        json.dump(data, file, indent=2)
        file.truncate() 
        


def postFSLfirst(cache_filename, token):
    #Calls the function to post an FSLFirst task many times, and updates the cache.
    with open(cache_filename, "r+") as file:
        data = json.load(file)
        for filename in data:
            if data[filename]['FSL']['Task1']['isUsed'] == None:
                jayson = cbrain_FSLFirst(token, data[filename]['FSL']['Task1']['inputID'])
                data[filename]['FSL']['Task1']['toolConfigID'] = jayson[0]['tool_config_id']
                data[filename]['FSL']['Task1']['taskID'] = jayson[0]["id"]
                data[filename]['FSL']['Task1']['status'] = jayson[0]["status"]
                data[filename]['FSL']['Task1']['isUsed'] = True
        file.seek(0)  # rewind
        json.dump(data, file, indent=2)
        file.truncate()



def updateStatuses(cache_filename, token):
    with open(cache_filename, "r+") as cacheFile:
        data = json.load(cacheFile)
        for (file, dataset) in data.items():
            for (dataset_Name, taskNum) in dataset.items():
                for (Task_NumStr, params) in taskNum.items():
                    if Task_NumStr != "Result" and params["taskID"] != None:
                        jayson = cbrain_getTaskInfo(token, str(params["taskID"]))
                        if jayson['status'] == "Completed":
                            #Task completed, update status and get output file ID
                            data[file]['FSL'][Task_NumStr]["status"] = jayson["status"]
                            #differentiate between one and many outputs
                            if '_cbrain_output_outputs' in jayson['params']:
                                data[file]['FSL'][Task_NumStr]["outputID"] = jayson['params']['_cbrain_output_outputs'][0]
                            if '_cbrain_output_output' in jayson['params']:
                                data[file]['FSL'][Task_NumStr]["outputID"] =  jayson['params']['_cbrain_output_output'][0]
                        else:
                            #Task not completed, just update status
                            data[file]['FSL'][Task_NumStr]["status"] = jayson["status"]
        cacheFile.seek(0)
        json.dump(data, cacheFile, indent=2)
        cacheFile.truncate()
          

                              
def postFSLstats(cache_filename, token):
    with open(cache_filename, "r+") as cacheFile:
        data = json.load(cacheFile)
        for file in data:
            if data[file]['FSL']['Task2']['isUsed'] == None and data[file]['FSL']['Task1']['status'] ==  "Completed":
                jayson = cbrain_FSLStats(token, data[file]['FSL']['Task1']['outputID'])
                data[file]['FSL']['Task2']['inputID'] = data[file]['FSL']['Task1']['outputID']
                data[file]['FSL']['Task2']['toolConfigID'] = jayson[0]['tool_config_id']
                data[file]['FSL']['Task2']['taskID'] = jayson[0]["id"]
                data[file]['FSL']['Task2']['status'] = jayson[0]["status"]
                data[file]['FSL']['Task2']['isUsed'] = True
        cacheFile.seek(0)  # rewind
        json.dump(data, cacheFile, indent=2)
        cacheFile.truncate() 



def populateResults(cache_filename, token):
    with open(cache_filename, "r+") as cacheFile:
        data = json.load(cacheFile)
        for (file, dataset) in data.items():
            for (dataset_Name, taskNum) in dataset.items():
                prevStr = None
                for (Task_NumStr, params) in taskNum.items():
                    if Task_NumStr == "Result": #Find the task before the result in the json
                        if data[file]['FSL']['Result']['isUsed'] == None and data[file]['FSL'][prevStr]['status'] ==  "Completed":
                            fileID = data[file]['FSL'][prevStr]['outputID']
                            vol = cbrain_download_text(fileID, token)
                            data[file]['FSL']['Result']['volume'] = vol
                            data[file]['FSL']['Result']['isUsed'] = True
                    prevStr = Task_NumStr
        cacheFile.seek(0)  # rewind
        json.dump(data, cacheFile, indent=2)
        cacheFile.truncate()   
