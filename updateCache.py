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
github_token = sys.argv[3]

##################################################################################

#Logins
token = cbrain_login(cbrain_user, cbrain_password)
github_instance = Github(github_token)

#Get newest version of cache from github 
repo = github_instance.get_repo("jacobsanz97/NDR-CI")
cache_file = repo.get_contents("/cache.json")
raw_cache_data = cache_file.decoded_content #binary to string so able to write json
base64_string = raw_cache_data.decode('UTF-8')
with open('temp_CI_cache.json', 'r+') as outfile: #create temporary cache file for CI
    outfile.write(base64_string + '\n')


#Perform computations and update cache
#populateCacheFilenames('temp_CI_cache.json', token)
updateStatuses('temp_CI_cache.json', token)
postFSLfirst('temp_CI_cache.json', token)
postFSLSubfolderExtractor('temp_CI_cache.json', token)
postFSLstats('temp_CI_cache.json', token)
populateResults('temp_CI_cache.json', token)
updateStatuses('temp_CI_cache.json', token)

#read the modified temporary json and update the permanent cache on github
#Deposit temporary cache in artefact extraction directory.
with open('temp_CI_cache.json', 'r') as infile:
    data = json.load(infile)
    json_data = json.dumps(data, indent=2) 
    repo.update_file("cache.json", "Updated computations in cache", json_data, cache_file.sha)

#Logout
cbrain_logout(token) 
