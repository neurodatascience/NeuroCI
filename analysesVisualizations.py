import requests
import json
import yaml
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
import sys
import os

from cbrainAPI import *
from cacheOps import *

###########################################################################################################################
#General functions

def boxplot(volume_list, pipeline_name, dataset_name):
    data = np.array(volume_list).astype(np.float)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Right? Hippocampal Volumes')
    ax1.boxplot(data)
    plt.xticks([1], [dataset_name + '/' + pipeline_name])
    plt.savefig('./artifacts/' + dataset_name + '_' + pipeline_name + '_box' + '.png') # Saves in artifact directory
    #plt.show()
    

def corrplot(volume_list, hearing_loss_list, pipeline_name, dataset_name):
    x = np.array(hearing_loss_list).astype(np.float)
    y = np.array(volume_list).astype(np.float)
    b, m = polyfit(x, y, 1)
    plt.plot(x, y, '.')
    plt.plot(x, b + m * x, '-')
    plt.title('Right? Hippocampal Volumes vs Worse_ear_dsi')
    plt.xlabel('Worse_ear_dsi')
    plt.ylabel('Hippocampal Volume')
    plt.savefig('./artifacts/' + dataset_name + '_' + pipeline_name + '_corr' + '.png') # Saves in artifact directory
    #plt.show()
    
####################################################################################################################
#Prevent-AD and hearing loss

def preventAD_get_labels_from_filename(filename):
	subject = filename[4:11]
	visit = filename[16:23]
	return (subject, visit)


def preventAD_get_measure_from_csv(subject, visit, data_file):
	with open(data_file, 'r') as read_obj:
		csv_reader = reader(read_obj)
		for row in csv_reader:
			if row[1] == subject and row[2]==visit:
				return row[19] #change this to get a different column in CSV


#Process the cache results for a single pipeline
def preventAD_process(data_file, cache_file, pipeline_name):
	
	hearing_loss_list = []
	volume_list = []
	with open(cache_file, "r") as file:
		cache = json.load(file)
		for entry in cache:
			
			if cache[entry][pipeline_name]['Result']['result'] != None:
				
				volume = cache[entry][pipeline_name]['Result']['result']
				volume = volume.partition(' ')[0]
				subject, visit = preventAD_get_labels_from_filename(entry)
				hearing_loss = preventAD_get_measure_from_csv(subject, visit, data_file)
				
				if hearing_loss != None: #only visualize if we have a hearing loss measure for subject/visit
					hearing_loss_list.append(hearing_loss)
					volume_list.append(volume)

	corrplot(volume_list, hearing_loss_list, pipeline_name, 'Prevent-AD')
	boxplot(volume_list, pipeline_name, 'Prevent-AD')

preventAD_data_file = 'Auditory_processing_Registered_PREVENTAD.csv'
preventAD_cache_file = 'Prevent-AD.json'

#########################################################################################################
#Compass-ND

#########################################################################################################
#UK-BioBank

#########################################################################################################
# Main section of Analyses

cbrain_user = sys.argv[1]
cbrain_password = sys.argv[2]
cbrain_token = cbrain_login(cbrain_user, cbrain_password)

#cbrain_download_DP_file('Auditory_processing_Registered_PREVENTAD.csv', 318, cbrain_token) #use this if you know the file name but not the ID. Takes a long time though.
cbrain_download_file(3497558, preventAD_data_file, cbrain_token) #use this (quicker) if you know the CBRAIN userfileID

with open('Experiment_Definition.yaml') as file: #Load experiment definition
	try:
		experiment_definition  = yaml.safe_load(file)
	except yaml.YAMLError as exception: #yaml file not valid
		print('The Experiment Definition file is not valid')
		print(exception)
	
	for pipeline in experiment_definition['Pipelines']:
		preventAD_process(preventAD_data_file, preventAD_cache_file, pipeline)

cbrain_logout(cbrain_token)
