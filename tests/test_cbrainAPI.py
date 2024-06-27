import unittest
import json
from unittest import TestCase
from unittest.mock import patch
from unittest.mock import MagicMock
from unittest import mock


import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cbrainAPI

##########################################################################

class TestcbrainAPI(unittest.TestCase):

	def test_cbrain_login_incorrect(self):
		with patch('cbrainAPI.requests.post') as mock_request:
			wrong_user = "wronguser"
			wrong_pass = "123"
			url = 'https://portal.cbrain.mcgill.ca/session'
			headers = {'Content-Type': 'application/x-www-form-urlencoded','Accept': 'application/json',}
			data = {'login': wrong_user, 'password': wrong_pass}
			#set status_code attribute to mock object, test.
			mock_request.return_value.status_code = 401
			self.assertEqual(cbrainAPI.cbrain_login(wrong_user, wrong_pass), 1)
			#test if request was posted with given credentials.
			mock_request.assert_called_once_with(url, data=data, headers=headers)
			
	def test_cbrain_login_correct(self):
		with patch('cbrainAPI.requests.post') as mock_request:
			correct_user = "correctuser"
			correct_pass = "123"
			url = 'https://portal.cbrain.mcgill.ca/session'
			headers = {'Content-Type': 'application/x-www-form-urlencoded','Accept': 'application/json',}
			data = {'login': correct_user, 'password': correct_pass}
			mock_request.return_value.status_code = 200
			expected = {"user_id":123,"cbrain_api_token":"testToken"}
			mock_request.return_value.json.return_value = expected
			self.assertEqual(cbrainAPI.cbrain_login(correct_user, correct_pass), expected['cbrain_api_token'])
			mock_request.assert_called_once_with(url, data=data, headers=headers)



	def test_cbrain_logout_incorrect(self):
		with patch('cbrainAPI.requests.delete') as mock_request:
			wrongToken = "wrongToken"
			url = 'https://portal.cbrain.mcgill.ca/session'
			headers = {'Accept': 'application/json'}
			params = (('cbrain_api_token', wrongToken),)
			mock_request.return_value.status_code = 401
			self.assertEqual(cbrainAPI.cbrain_logout(wrongToken), 1)
			mock_request.assert_called_once_with(url, headers=headers, params=params)
			
	def test_cbrain_logout_correct(self):
		with patch('cbrainAPI.requests.delete') as mock_request:
			correctToken = "correctToken"
			url = 'https://portal.cbrain.mcgill.ca/session'
			headers = {'Accept': 'application/json'}
			params = (('cbrain_api_token', correctToken),)
			mock_request.return_value.status_code = 200
			self.assertEqual(cbrainAPI.cbrain_logout(correctToken), 0)
			mock_request.assert_called_once_with(url, headers=headers, params=params)



	def test_cbrain_list_data_provider_incorrect(self):
		with patch('cbrainAPI.requests.get') as mock_request:
			token = "token"
			dataprovider_ID = 123
			url = 'https://portal.cbrain.mcgill.ca/data_providers/' + str(dataprovider_ID) + '/browse'
			headers = {'Accept': 'application/json',}
			params = (('id', str(dataprovider_ID)),('cbrain_api_token', token),)
			mock_request.return_value.status_code = 401
			self.assertEqual(cbrainAPI.cbrain_list_data_provider(dataprovider_ID, token), 1)
			mock_request.assert_called_once_with(url, headers=headers, params=params, allow_redirects=True)
			
	def test_cbrain_list_data_provider_correct(self):
		with patch('cbrainAPI.requests.get') as mock_request:
			token = "token"
			dataprovider_ID = 123
			url = 'https://portal.cbrain.mcgill.ca/data_providers/' + str(dataprovider_ID) + '/browse'
			headers = {'Accept': 'application/json',}
			params = (('id', str(dataprovider_ID)),('cbrain_api_token', token),)
			mock_request.return_value.status_code = 200
			expected = {"aJSON":123,"testing":"123"}
			mock_request.return_value.json.return_value = expected
			self.assertEqual(cbrainAPI.cbrain_list_data_provider(dataprovider_ID, token), expected)
			mock_request.assert_called_once_with(url, headers=headers, params=params, allow_redirects=True)
			
			
			
	def test_cbrain_get_task_info_incorrect(self):
		with patch('cbrainAPI.requests.get') as mock_request:
			token = "token"
			taskID = 123
			url = 'https://portal.cbrain.mcgill.ca/tasks/' + str(taskID)
			headers = {'Accept': 'application/json',}
			params = (('id', str(taskID)),('cbrain_api_token', token),)
			mock_request.return_value.status_code = 401
			self.assertEqual(cbrainAPI.cbrain_get_task_info(token, taskID), 1)
			mock_request.assert_called_once_with(url, headers=headers, params=params)
			
	def test_cbrain_get_task_info_correct(self):
		with patch('cbrainAPI.requests.get') as mock_request:
			token = "token"
			taskID = 123
			url = 'https://portal.cbrain.mcgill.ca/tasks/' + str(taskID)
			headers = {'Accept': 'application/json',}
			params = (('id', str(taskID)),('cbrain_api_token', token),)
			mock_request.return_value.status_code = 200
			expected = {"aJSON":123,"testing":"123"}
			mock_request.return_value.json.return_value = expected
			self.assertEqual(cbrainAPI.cbrain_get_task_info(token, taskID), expected)
			mock_request.assert_called_once_with(url, headers=headers, params=params)		
			
			
			
	def test_cbrain_download_text_incorrect(self):
		with patch('cbrainAPI.requests.get') as mock_request:
			token = "token"
			fileID = 123
			url = 'https://portal.cbrain.mcgill.ca/userfiles/' + str(fileID) + '/content'
			headers = {'Accept': 'text',}
			params = (('cbrain_api_token', token),)
			mock_request.return_value.status_code = 401
			self.assertEqual(cbrainAPI.cbrain_download_text(fileID, token), 1)
			mock_request.assert_called_once_with(url, headers=headers, params=params, allow_redirects=True)
			
	def test_cbrain_download_text_correct(self):
		with patch('cbrainAPI.requests.get') as mock_request:
			token = "token"
			fileID = 123
			url = 'https://portal.cbrain.mcgill.ca/userfiles/' + str(fileID) + '/content'
			headers = {'Accept': 'text',}
			params = (('cbrain_api_token', token),)
			mock_request.return_value.status_code = 200
			expected = "return string"
			mock_request.return_value.text = expected
			self.assertEqual(cbrainAPI.cbrain_download_text(fileID, token), expected)
			mock_request.assert_called_once_with(url, headers=headers, params=params, allow_redirects=True)
			
			
	def test_cbrain_post_task_incorrect(self):
		with patch('cbrainAPI.requests.post') as mock_request:
			cbrain_token = "token"
			userfile_id = "123"
			tool_config_id = 456
			url = 'https://portal.cbrain.mcgill.ca/tasks'
			parameter_dictionary = {
				"interface_userfile_ids": ["fileID"], 
				"input_file": "fileID", 
				"prefix": "output", 
				"brain_extracted": False, 
				"three_stage": False, 
				"verbose": False       
			}
			parameter_dictionary['interface_userfile_ids'] = [str(userfile_id)]
			parameter_dictionary['input_file'] = str(userfile_id)
			headers = {'Content-Type': 'application/json', 'Accept': 'application/json',}
			params = (('cbrain_api_token', cbrain_token),)
			data = {
				"cbrain_task": {
					'tool_config_id': tool_config_id,
					'params': parameter_dictionary, 
				'run_number': None, 
				'results_data_provider_id': 179, 
				'cluster_workdir_size': None, 
				'workdir_archived': True,  
				'description': ''}
			}
			y = json.dumps(data)
			mock_request.return_value.status_code = 401
			self.assertEqual(cbrainAPI.cbrain_post_task(cbrain_token, userfile_id, tool_config_id, parameter_dictionary), 1)
			mock_request.assert_called_once_with(url, headers=headers, params=params, data=y)

	def test_cbrain_post_task_correct(self):
		with patch('cbrainAPI.requests.post') as mock_request:
			cbrain_token = "token"
			userfile_id = "123"
			tool_config_id = 456
			url = 'https://portal.cbrain.mcgill.ca/tasks'
			parameter_dictionary = {
				"interface_userfile_ids": ["fileID"], 
				"input_file": "fileID", 
				"prefix": "output", 
				"brain_extracted": False, 
				"three_stage": False, 
				"verbose": False       
			}
			parameter_dictionary['interface_userfile_ids'] = [str(userfile_id)]
			parameter_dictionary['input_file'] = str(userfile_id)
			headers = {'Content-Type': 'application/json', 'Accept': 'application/json',}
			params = (('cbrain_api_token', cbrain_token),)
			data = {
				"cbrain_task": {
					'tool_config_id': tool_config_id,
					'params': parameter_dictionary, 
				'run_number': None, 
				'results_data_provider_id': 179, 
				'cluster_workdir_size': None, 
				'workdir_archived': True,  
				'description': ''}
			}
			y = json.dumps(data)
			mock_request.return_value.status_code = 200
			expected = {"aJSON":123,"testing":"123"}
			mock_request.return_value.json.return_value = expected
			self.assertEqual(cbrainAPI.cbrain_post_task(cbrain_token, userfile_id, tool_config_id, parameter_dictionary), expected)
			mock_request.assert_called_once_with(url, headers=headers, params=params, data=y)



	def test_cbrain_download_file_incorrect(self):
		with patch('cbrainAPI.requests.get') as mock_request:
			token = "token"
			fileID = 123
			filename = "blah.nii.gz"
			url = 'https://portal.cbrain.mcgill.ca/userfiles/' + str(fileID) + '/content'
			headers = {'Accept': 'application/json',}
			params = (('id', str(fileID)),('cbrain_api_token', token),)
			mock_request.return_value.status_code = 401
			self.assertEqual(cbrainAPI.cbrain_download_file(fileID, filename, token), 1)
			mock_request.assert_called_once_with(url, headers=headers, params=params, allow_redirects=True)

#	def test_cbrain_download_file_correct(self):
#		with patch('cbrainAPI.requests.get') as mock_request:
#			token = "token"
#			fileID = 123
#			filename = "blah.nii.gz"
#			url = 'https://portal.cbrain.mcgill.ca/userfiles/' + str(fileID) + '/content'
#			headers = {'Accept': 'application/json',}
#			params = (('id', str(fileID)),('cbrain_api_token', token),)
#			mock_request.return_value.status_code = 200
#			expected = {"aJSON":123,"testing":"123"}
#			mock_request.return_value.text = expected
#			#self.assertEqual(cbrainAPI.cbrain_download_file(fileID, filename, token), expected)
#			#mock_request.assert_called_once_with(url, headers=headers, params=params, allow_redirects=True)

	def test_cbrain_download_DP_file(self):
		token = 'token'
		file_ID = 123
		DP_ID = 456
		filename = 'test.csv'
		with patch('cbrainAPI.cbrain_list_data_provider') as mock:
			mock(str(file_ID), str(DP_ID), token)
			mock.assert_called_with(str(file_ID), str(DP_ID), token)			
		with patch('cbrainAPI.cbrain_download_file') as mock2:
			mock2([str(file_ID)], filename, token)
			mock2.assert_called_with([str(file_ID)], filename, token)			

				
if __name__ == '__main__':
	unittest.main()
