import json

import requests

##################################################################################


def cbrain_login(username, password):
    """Posts API call to CBRAIN to obtain a authentication token given a username and password."""

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    data = {"login": username, "password": password}

    response = requests.post(
        "https://portal.cbrain.mcgill.ca/session", headers=headers, data=data
    )

    if response.status_code == 200:
        print("Login success")
        print(response.content)
        jsonResponse = response.json()
        return jsonResponse["cbrain_api_token"]
    else:
        print("Login failure")
        return 1


def cbrain_logout(cbrain_token):
    """Ends a CBRAIN session."""

    headers = {
        "Accept": "application/json",
    }
    params = (("cbrain_api_token", cbrain_token),)

    response = requests.delete(
        "https://portal.cbrain.mcgill.ca/session", headers=headers, params=params
    )

    if response.status_code == 200:
        print("Logout success")
        return 0
    else:
        print("Logout failure")
        return 1


def cbrain_list_data_provider(data_provider_ID, cbrain_token):
    """Lists all files in a CBRAIN data provider."""

    data_provider_ID = str(data_provider_ID)
    headers = {
        "Accept": "application/json",
    }
    params = (
        ("id", data_provider_ID),
        ("cbrain_api_token", cbrain_token),
    )
    url = (
        "https://portal.cbrain.mcgill.ca/data_providers/" + data_provider_ID + "/browse"
    )

    response = requests.get(url, headers=headers, params=params, allow_redirects=True)

    if response.status_code == 200:
        return response.json()
    else:
        print("DP browse failure")
        return 1


def cbrain_post_task(cbrain_token, userfile_id, tool_config_id, parameter_dictionary):
    """Posts a task in CBRAIN."""

    userfile_id = str(userfile_id)

    # Parse the parameter dictionary json, and insert the userfile IDs.
    parameter_dictionary["interface_userfile_ids"] = [userfile_id]
    parameter_dictionary["input_file"] = userfile_id

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    params = (("cbrain_api_token", cbrain_token),)
    data = {
        "cbrain_task": {
            "tool_config_id": tool_config_id,
            "params": parameter_dictionary,
            "run_number": None,
            "results_data_provider_id": 179,  # Using Beluga
            "cluster_workdir_size": None,
            "workdir_archived": True,
            "description": "",
        }
    }

    y = json.dumps(data)  # convert data field to JSON:
    response = requests.post(
        "https://portal.cbrain.mcgill.ca/tasks", headers=headers, params=params, data=y
    )

    if response.status_code == 200:
        print(response.text)
        jsonResponse = response.json()
        return jsonResponse
    else:
        print("Task posting failed.")
        print(response.content)
        return 1


def cbrain_get_all_tasks(cbrain_token):
    """Gets the list of all the tasks of the user on CBRAIN."""

    headers = {
        "Accept": "application/json",
    }
    params = {"cbrain_api_token": cbrain_token, "page": 1, "per_page": 1000}
    url = "https://portal.cbrain.mcgill.ca/tasks"
    task_list = []

    while True:

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            jsonResponse = response.json()
            task_list += jsonResponse
            params["page"] += 1
        else:
            print("Task list retrieval failed.")
            return 1

        if len(jsonResponse) < params["per_page"]:
            break

    return task_list


def cbrain_get_task_info_from_list(task_list, task_ID):
    """Obtains info on the progress of a single task, given the list of all tasks for the user."""

    for task in task_list:
        if task_ID == task["id"] or int(task_ID) == task["id"]:
            return task


def cbrain_get_task_info(cbrain_token, task_ID):
    """Obtains information on the progress of a single task by querying for a single task."""

    task_ID = str(task_ID)
    headers = {
        "Accept": "application/json",
    }
    params = (("id", task_ID), ("cbrain_api_token", cbrain_token))
    url = "https://portal.cbrain.mcgill.ca/tasks/" + task_ID

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        jsonResponse = response.json()
        return jsonResponse
    else:
        print("Task Info retrieval failed.")
        return 1


def cbrain_download_text(userfile_ID, cbrain_token):
    """Downloads the text from a file on CBRAIN."""

    userfile_ID = str(userfile_ID)
    headers = {
        "Accept": "text",
    }
    params = (("cbrain_api_token", cbrain_token),)
    url = "https://portal.cbrain.mcgill.ca/userfiles/" + userfile_ID + "/content"

    response = requests.get(url, headers=headers, params=params, allow_redirects=True)

    if response.status_code == 200:
        return response.text
    else:
        print("Download failure")
        print(response.status_code)
        return 1


def cbrain_download_file(userfile_ID, filename, cbrain_token):
    """Downloads a file from CBRAIN and saves it, given a userfile ID."""
    fileID = str(userfile_ID)
    headers = {
        "Accept": "application/json",
    }
    params = (
        ("id", fileID),
        ("cbrain_api_token", cbrain_token),
    )
    url = "https://portal.cbrain.mcgill.ca/userfiles/" + fileID + "/content"

    response = requests.get(url, headers=headers, params=params, allow_redirects=True)
    if response.status_code == 200:
        open(filename, "wb").write(response.content)
        print("Downloaded file " + filename)
        return 0
    else:
        print("File download failure: " + filename)
        return 1


def cbrain_download_DP_file(filename, data_provider_id, cbrain_token):
    """Given a filename and data provider, download the file from the data provider."""
    data_provider_browse = cbrain_list_data_provider(
        str(data_provider_id), cbrain_token
    )  # Query CBRAIN to list all files in data provider.
    print(data_provider_browse)

    try:
        for entry in data_provider_browse:
            if (
                "userfile_id" in entry and entry["name"] == filename
            ):  # if it's a registered file, and filename matches
                print(
                    "Found registered file: "
                    + filename
                    + " in Data Provider with ID "
                    + str(data_provider_id)
                )
                cbrain_download_file(entry["userfile_id"], filename, cbrain_token)
                return 0
            else:
                print(
                    "File "
                    + filename
                    + " not found in Data Provider "
                    + str(data_provider_id)
                )
                return 1

    except Exception as e:
        print("Error in browsing data provider or file download")
        return


def cbrain_sync_file(userfile_id_list, cbrain_token):
    """Makes sure a file in a data provider is synchronized with CBRAIN."""
    # userfile_id_list can either be a string eg. '3663657', or a list eg. ['3663729', '3663714']
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    params = (
        ("file_ids[]", userfile_id_list),
        ("cbrain_api_token", cbrain_token),
    )

    response = requests.post(
        "https://portal.cbrain.mcgill.ca/userfiles/sync_multiple",
        headers=headers,
        params=params,
    )

    if response.status_code == 200:
        print("Synchronized userfiles " + str(userfile_id_list))
        return
    else:
        print("Userfile sync failed for IDs: " + str(userfile_id_list))
        print(response.status_code)
        return
