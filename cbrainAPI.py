import requests
import json
import atexit
import sys
import os
##################################################################################

'''Posts API call to CBRAIN to obtain a authentication token given a username and password'''


class CbrainAPI:

    _url_cbrain_portal = 'https://portal.cbrain.mcgill.ca'
    _url_cbrain_session = os.path.join(_url_cbrain_portal, 'session')
    _url_cbrain_data_providers = os.path.join(
        _url_cbrain_portal, 'data_providers')
    _url_cbrain_tasks = os.path.join(_url_cbrain_portal, 'tasks')
    _url_cbrain_userfiles = os.path.join(_url_cbrain_portal, 'userfiles')
    _url_cbrain_sync_multiple = os.path.join(
        _url_cbrain_userfiles, 'sync_multiple')

    def __init__(self, username, password):
        self._token = self.login(username, password)
        atexit.register(self.logout)

    def failure(self, msg):
        print(msg, file=sys.stderr)
        sys.exit(1)

    def get_token(self):
        return self._token

    def login(self, username, password):

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
        }

        data = {
            'login': username,
            'password': password
        }

        response = requests.post(
            self._url_cbrain_session, headers=headers, data=data)

        if response.status_code == requests.status_codes.codes['OK']:
            print("Login success")
            print(response.content)
            jsonResponse = response.json()
            return jsonResponse["cbrain_api_token"]
        else:
            self.failure("Login failure")

    def logout(self):
        '''End a CBRAIN session'''

        headers = {
            'Accept': 'application/json',
        }
        params = (
            ('cbrain_api_token', self.get_token()),
        )

        response = requests.delete(
            self._url_cbrain_session, headers=headers, params=params)

        if response.status_code == requests.status_codes.codes['OK']:
            print("Logout success")
            return 0
        else:
            self.failure("Logout failure")

    def list_data_provider(self, data_provider_ID):
        '''Lists all files in a CBRAIN data provider'''

        data_provider_ID = str(data_provider_ID)
        headers = {
            'Accept': 'application/json',
        }
        params = (
            ('id', data_provider_ID),
            ('cbrain_api_token', self.get_token()),
        )
        url = os.path.join(self._url_cbrain_data_providers,
                           data_provider_ID, 'browse')

        response = requests.get(url, headers=headers,
                                params=params, allow_redirects=True)

        if response.status_code == requests.status_codes.codes['OK']:
            return response.json()
        else:
            self.failure('DP browse failure')

    def post_task(self, userfile_id, tool_config_id, parameter_dictionary):
        '''Posts a task in CBRAIN'''

        userfile_id = str(userfile_id)

        # Parse the parameter dictionary json, and insert the userfile IDs.
        parameter_dictionary['interface_userfile_ids'] = [userfile_id]
        parameter_dictionary['input_file'] = userfile_id

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        params = (
            ('cbrain_api_token', self.get_token()),
        )
        data = {
            "cbrain_task": {
                'tool_config_id': tool_config_id,
                'params': parameter_dictionary,
                'run_number': None,
                'results_data_provider_id': 179,  # Using Beluga
                'cluster_workdir_size': None,
                'workdir_archived': True,
                'description': ''}
        }

        y = json.dumps(data)  # convert data field to JSON:
        response = requests.post(
            self._url_cbrain_tasks, headers=headers, params=params, data=y)

        if response.status_code == requests.status_codes.codes['OK']:
            print(response.text)
            jsonResponse = response.json()
            return jsonResponse
        else:
            self.failure(
                f"Task posting failed.{os.path.sep}{response.content}")

    def get_all_tasks(self):
        '''Gets the list of all the tasks of the user on CBRAIN'''

        headers = {
            'Accept': 'application/json',
        }
        params = {
            'cbrain_api_token': self.get_token(),
            'page': 1,
            'per_page': 1000
        }
        task_list = []

        while True:

            response = requests.get(
                self._url_cbrain_tasks, headers=headers, params=params)

            if response.status_code == requests.status_codes.codes['OK']:
                jsonResponse = response.json()
                task_list += jsonResponse
                params['page'] += 1
            else:
                self.failure("Task list retrieval failed.")

            if len(jsonResponse) < params['per_page']:
                break

        return task_list

    def get_task_info_from_list(self, task_list, task_ID):
        '''Obtains info on the progress of a single task, given the list of all tasks for the user'''

        for task in task_list:
            if task_ID == task['id'] or int(task_ID) == task['id']:
                return task

    def get_task_info(self, task_ID):
        '''Obtains information on the progress of a single task by querying for a single task'''

        task_ID = str(task_ID)
        headers = {
            'Accept': 'application/json',
        }
        params = (
            ('id', task_ID),
            ('cbrain_api_token', self.get_token())
        )

        url = os.path.join(self._url_cbrain_tasks, task_ID)

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == requests.status_codes.codes['OK']:
            jsonResponse = response.json()
            return jsonResponse
        else:
            self.failure("Task Info retrieval failed.")

    def download_text(self, userfile_ID):
        '''Downloads the text from a file on CBRAIN'''

        userfile_ID = str(userfile_ID)
        headers = {
            'Accept': 'text',
        }
        params = (
            ('cbrain_api_token', self.get_token()),
        )
        url = os.path.join(self._url_cbrain_userfiles, userfile_ID, 'content')

        response = requests.get(url, headers=headers,
                                params=params, allow_redirects=True)

        if response.status_code == requests.status_codes.codes['OK']:
            return response.text
        else:
            msg = f'Download failure{os.path.sep}{response.status_code}'
            self.failure(msg)

    def download_file(self, userfile_ID, filename):
        '''Downloads a file from CBRAIN and saves it, given a userfile ID'''

        fileID = str(userfile_ID)
        headers = {
            'Accept': 'application/json',
        }
        params = (
            ('id', fileID),
            ('cbrain_api_token', self.get_token()),
        )

        url = os.path.join(self._url_cbrain_userfiles, fileID, 'content')

        response = requests.get(url, headers=headers,
                                params=params, allow_redirects=True)
        if response.status_code == requests.status_codes.codes['OK']:
            open(filename, 'wb').write(response.content)
            print(f"Downloaded file {filename}")
            return 0
        else:
            self.failure(f'File download failure: {filename}')

    '''Given a filename and data provider, download the file from the data provider'''

    def download_DP_file(self, filename, data_provider_id):

        # Query CBRAIN to list all files in data provider.
        data_provider_browse = self.list_data_provider(str(data_provider_id))
        print(data_provider_browse)

        try:
            for entry in data_provider_browse:
                # if it's a registered file, and filename matches
                if 'userfile_id' in entry and entry['name'] == filename:
                    print(
                        f"Found registered file: {filename} in Data Provider with ID {data_provider_id}")
                    self.download_file(entry['userfile_id'], filename)
                    return 0
                else:
                    msg = f"File {filename} not found in Data Provider {data_provider_id}"
                    self.failure(msg)

        except Exception as e:
            self.failure("Error in browsing data provider or file download")

    '''Makes sure a file in a data provider is synchronized with CBRAIN'''

    def sync_file(self, userfile_id_list):
        # userfile_id_list can either be a string eg. '3663657', or a list eg. ['3663729', '3663714']
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

        params = (
            ('file_ids[]', userfile_id_list),
            ('cbrain_api_token', self.get_token()),
        )

        response = requests.post(
            self._url_cbrain_sync_multiple, headers=headers, params=params)

        if response.status_code == requests.status_codes.codes['OK']:
            print(f"Synchronized userfiles {userfile_id_list}"))
            return
        else:
            self.failure(
                f"Userfile sync failed for IDs: {userfile_id_list}{os.path.sep}{response.status_code}")
