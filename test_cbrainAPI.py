import unittest
import cbrainAPI
from unittest.mock import patch

class TestcbrainAPI(unittest.TestCase):

	def test_cbrain_login(self):

		#with patch('cbrainAPI.requests.post') as mocked_post:
		#
		#with patch('cbrainAPI.cbrain_login') as mocked_login:
		#
		#	mocked_post.return_value.ok = True
		#	mocked_get.return_value.text = 'Success'
		#	schedule = self.emp_1.monhtly_schedule('May').
		#	mocked_get.assert_called_with('the input str')
		#	self.assertEqual(schedule, "Success')		

		#	cbrainAPI.cbrain_login()
		#	self.assertEqual(json_data, {"key1": "value1"})

		#	self.assertIn(mock.call('http://someurl.com/test.json'), mock_get.call_args_list)
   		#    self.assertIn(mock.call('http://someotherurl.com/anothertest.json'), mock_get.call_args_list



if __name__ == '__main__':
	unittest.main()
