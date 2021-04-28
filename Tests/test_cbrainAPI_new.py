import unittest
import json
from unittest import TestCase
from unittest.mock import patch
from unittest.mock import MagicMock

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



if __name__ == '__main__':
	unittest.main()
