import unittest
import json
from unittest import TestCase
from unittest.mock import patch
from unittest.mock import MagicMock

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
		assert cacheOps.generateCacheSubject(niftiFile, cbrain_userfileID) == data



	

if __name__ == '__main__':
	unittest.main()
