import requests
import json
import sys
import os
from github import Github
from ast import literal_eval

from cbrainAPI import *
from cacheOps import *

##################################################################################
#Obtain login credentials from args, stored in CI environment variables.

cbrain_user = sys.argv[1]
cbrain_password = sys.argv[2]
CCI_token = sys.argv[3]

##################################################################################

#Logins
token = cbrain_login(cbrain_user, cbrain_password)

headers = {'Circle-Token': CCI_token}
response = requests.get('https://circleci.com/api/v1.1/project/github/jacobsanz97/NDR-CI/latest/artifacts', headers=headers)

linkToCache = ""
if response.status_code == 200:
    literalList = literal_eval(response.text)
    for file in literalList:
        if 'temp_CI_cache.json' in file['url']:
            linkToCache = file['url']
else:
    print("Error loading CircleCI artifacts")
    print(response.text)

response = requests.get(linkToCache, headers=headers)
if response.status_code == 200:
    jsonCache = json.loads(response.text)
else:
    print(response.text)
    jsonCache = json.loads("{}")
    print("Cache file not initialized...Creating a new one.")

with open('temp_CI_cache.json', 'w') as outfile: #create temporary cache file for CI
    json.dump(jsonCache, outfile)


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

#Logout
cbrain_logout(token) 
