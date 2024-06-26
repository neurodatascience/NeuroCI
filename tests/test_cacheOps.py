import unittest
import json
from unittest import TestCase
from unittest.mock import patch
from unittest.mock import MagicMock

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cacheOps
##########################################################################
#At a later stage replace returns of 1 with exceptions and test those.
class TestcacheOps(unittest.TestCase):

	def test_generateCacheSubject(self):
		niftiFile = "test.nii.gz"
		cbrain_userfileID = "123"
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
				"Task3": {
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
		self.assertEqual(cacheOps.generateCacheSubject(niftiFile, cbrain_userfileID), data)
		
		
	def test_generate_cache_subject(self):
		niftiFile = "test.nii.gz"
		cbrain_userfileID = "123"
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
				"Task3": {
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
		self.assertEqual(cacheOps.generateCacheSubject(niftiFile, cbrain_userfileID), data)

			#run listDP and make a fake output from its real output
			#json.load some fake data in with patch json.load
			#json.dump assert correct dump?

	def test_populateCacheFilenames(self):
		with patch('cacheOps.cbrain_listDP') as mock_DP, patch('cacheOps.json.load') as mock_load, patch('cacheOps.json.dump') as mock_dump, patch('cacheOps.open') as file_mock:
			mock_DP.return_value =  [{'name': 'sub-1004359_ses-PREBL00_run-001_T1w.nii.gz', 'symbolic_type': 'regular', 'size': 14444183, 'uid': 3100308, 'gid': 6049200, 'atime': 1601006069, 'mtime': 1593535830, 'permissions': 33060, 'userfile_id': 2933481, 'message': '', 'state_ok': True}, {'name': 'sub-1004359_ses-PREBL00_run-001_T2star.nii.gz', 'symbolic_type': 'regular', 'size': 17134035, 'uid': 3100308, 'gid': 6049200, 'atime': 1598419123, 'mtime': 1593535869, 'permissions': 33060, 'message': '', 'state_ok': True}]
			mock_load.return_value = {}
			expected_data = { "sub-1004359_ses-PREBL00_run-001_T1w.nii.gz": {
				"FSL": { #For Future expansion to more pipelines...take 'FSL' as fn input?
					"Task1": {
						"inputID": 2933481,
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
					"Task3": {
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
			self.assertEqual(cacheOps.populateCacheFilenames(file_mock, "blahToken"), expected_data)



	def test_populateCacheFilenames(self):
				

if __name__ == '__main__':
	unittest.main()
