import requests
import json
import sys
import os
from github import Github

from cbrainAPI import *
from cacheOps import *

##################################################################################
#Obtain login credentials from args, stored in CI environment variables.

cbrain_user = sys.argv[1]
cbrain_password = sys.argv[2]
#github_token = sys.argv[3]

##################################################################################

#Logins
token = cbrain_login(cbrain_user, cbrain_password)
#github_instance = Github(github_token)

#Get newest version of cache from github 
#repo = github_instance.get_user().get_repo("NDR-CI")
#repo = github_instance.get_repo("jacobsanz97/NDR-CI")
#cache_file = repo.get_contents("/cache.json")
#raw_cache_data = cache_file.decoded_content #binary to string so able to write json
#base64_string = raw_cache_data.decode('UTF-8')

jsonBin = sys.argv[3]
jsonToken = sys.argv[4]


url = 'https://api.jsonbin.io/b/' + jsonBin + '/latest'
headers = {'secret-key': jsonToken}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    print("Downloaded cache file")
    jsonResponse = response.json()
else:
    print("Error downloading cache file")
    print(response.text)

with open('temp_CI_cache.json', 'w') as outfile: #create temporary cache file for CI
    json.dump(jsonResponse, outfile)

print('written cache to temp file')

#Perform computations and update cache
populateCacheFilenames('temp_CI_cache.json', token)
print('populated cache filenames')
updateStatuses('temp_CI_cache.json', token)
postFSLfirst('temp_CI_cache.json', token)
postFSLSubfolderExtractor('temp_CI_cache.json', token)
postFSLstats('temp_CI_cache.json', token)
print('posted tasks')
populateResults('temp_CI_cache.json', token)
updateStatuses('temp_CI_cache.json', token)

print("Done with scheduled computations")

with open('temp_CI_cache.json', 'r') as infile:
    data = json.load(infile)
    url = 'https://api.jsonbin.io/b/' + jsonBin
    headers = {'Content-Type': 'application/json', 'secret-key' : jsonToken}
    response = requests.put(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Uploaded cache file")
    else:
        print("Error uploading cache file")





#read the modified temporary json and update the permanent cache on github
#Deposit temporary cache in artefact extraction directory.
#with open('temp_CI_cache.json', 'r') as infile:
#    data = json.load(infile)
#    json_data = json.dumps(data, indent=2) 
#    repo.update_file("cache.json", "Updated computations in cache", json_data, cache_file.sha)

#Logout
cbrain_logout(token) 
