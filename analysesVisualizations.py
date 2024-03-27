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

### Note that this Python file is the 'designated' place for the user to provide their own code for visualization and artifact production purposes.
### It makes to have a space for 'custom processing' as datasets can vary. The next version may be BIDS only.
### I have provided some basic static image file artifact examples that were used in the NeuroCI eScience paper. The interactive visualization is via Nerv.

'''Generates a simple boxplot, not used for now.'''
def boxplot(volume_list, pipeline_name, dataset_name):
    data = np.array(volume_list).astype(np.float)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Left Hippocampal Volumes (mm3)')
    ax1.boxplot(data)
    plt.xticks([1], [dataset_name + '/' + pipeline_name])
    plt.savefig('./artifacts/' + dataset_name + '_' + pipeline_name + '_box' + '.png') # Saves in artifact directory
    #plt.show()
    
'''Scatter plot and line of best fit for phenotypic data'''
def corrplot(volume_list, hearing_loss_list, pipeline_name, dataset_name, title, x_label, y_label, QC_method):
	
    new_hl_list = []
    new_vol_list = []
    index = 0
    for elem in hearing_loss_list:
        if elem != 'NA': #Append to new list if value is not NA
            new_hl_list.append(hearing_loss_list[index])
            new_vol_list.append(volume_list[index])
        index = index + 1
    
    x = np.array(new_hl_list).astype(np.float)
    y = np.array(new_vol_list).astype(np.float)
    b, m = polyfit(x, y, 1)
    plt.plot(x, y, '.')
    plt.plot(x, b + m * x, '-')
    plt.ylim(0, 6000)
    plt.title(title + '\n' + dataset_name + ' with ' + pipeline_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('./artifacts/' + dataset_name + '_' + pipeline_name + '_line_' + QC_method + '.png') # Saves in artifact directory
    plt.close() #so we have separate figures and not overlaid.
    #plt.show()

'''Scatter plot to compare 2 pipeline correlations'''
def XY_plot(volume_list_x, volume_list_y, pipeline_name_x, pipeline_name_y, dataset_name, title, x_lim, y_lim, x_label, y_label, QC_method):
    fig = plt.figure()
    plt.title(title + '\n' + pipeline_name_x + ' vs ' + pipeline_name_y + ' on ' + dataset_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, x_lim)
    plt.xlim(0, y_lim)
    plt.plot(volume_list_x, volume_list_y, '+')
    print("Correlation of " + pipeline_name_x + "/" + pipeline_name_y + " on " + dataset_name + " " + str(np.corrcoef(volume_list_x, volume_list_y)[0][1]))
    fig.savefig('./artifacts/' + pipeline_name_x + "_" + pipeline_name_y + "_" + dataset_name + "_xy_" + QC_method + ".png")

####################################################################################################################
#Prevent-AD and hearing loss

def preventAD_get_labels_from_filename(filename):
	subject = filename[4:11]
	visit = filename[16:23]
	return (subject, visit)


def preventAD_get_measure_from_csv(subject, visit, data_file, row_index):
	with open(data_file, 'r') as read_obj:
		csv_reader = reader(read_obj)
		for row in csv_reader:
			if row[1] == subject and row[2]==visit:
				return row[row_index] #change this to get a different column in CSV


#Process the Prevent-AD cache results for a single pipeline, and remove outliers below the lower_bound and above the upper_bound
def preventAD_process_bounded_line(data_file, cache_file, pipeline_name, lower_bound, upper_bound):
	
	plotted_data_points = 0
	removed_data_points = 0
	corrupt_data_points = 0
	hearing_loss_list = []
	volume_list = []
	with open(cache_file, "r") as file:
		cache = json.load(file)
		for entry in cache:
			
			if cache[entry][pipeline_name]['Result']['result'] != None and "T1" in entry:
				
				volume = cache[entry][pipeline_name]['Result']['result']
				
				if volume != 1 and float(volume) > lower_bound and float(volume) < upper_bound: #If >1 word in result string - necessary for FSL, but maybe not for other pipelines.
					volume = volume.partition(' ')[0] #Get the first word 
					subject, visit = preventAD_get_labels_from_filename(entry)
					plotted_data_points += 1
					try:
						hearing_loss = preventAD_get_measure_from_csv(subject, visit, data_file, 5) #19 = hearing loss row, 5 = age
					except Exception as e:
						print("Error getting CSV file measures for Prevent-AD.")
						return #skips the plotting
					
					if hearing_loss != None: #only visualize if we have a hearing loss measure for subject/visit
						hearing_loss_list.append(hearing_loss)
						volume_list.append(volume)
				else:
					removed_data_points += 1
			else:
				corrupt_data_points += 1
		
	if len(volume_list) >= 1 and len(hearing_loss_list)>=1: #If there is at least one data point.
		#corrplot(volume_list, hearing_loss_list, pipeline_name, 'Prevent-AD', 'Left Hippocampal Volumes vs Age', 'Worse_ear_dsi', 'Hippocampal Volume ($mm^3$)')
		corrplot(volume_list, hearing_loss_list, pipeline_name, 'Prevent-AD', 'Left Hippocampal Volumes vs Age (Bounded QC)', 'Age (months)', 'Hippocampal Volume ($mm^3$)', 'bounded')
		#boxplot(volume_list, pipeline_name, 'Prevent-AD')
		print('Generated bound line plots for ' + cache_file + '/' + pipeline_name + ". Points plotted: " + str(plotted_data_points) + ", points removed: " + str(removed_data_points) + ", points corrupted: " + str(corrupt_data_points))



#Should print r score and output graph after QC
def preventAD_process_discrepancy_line(data_file, cache_filename, in_pipeline_name, threshold):

    plotted_data_points = 0
    removed_data_points = 0
    hearing_loss_list = []
    volume_list = []

    with open(cache_filename, "r+") as cache_file:
        data = json.load(cache_file)
        for (file, pipeline) in data.items(): #Parse the json
            results_for_file = []
            for (pipeline_name, task_name) in pipeline.items():
                for (task_name_str, params) in task_name.items():
                    if task_name_str == "Result":
                        results_for_file.append((pipeline_name, data[file][pipeline_name][task_name_str]["result"]))

            if not flag_result_discrepancy(results_for_file, threshold) and data[file][in_pipeline_name]['Result']['result'] != None: #if not, can proceed with graphing
                volume = data[file][in_pipeline_name]['Result']['result']                   
                if volume != 1:
                    volume = volume.partition(' ')[0] #Get the first word 
                    subject, visit = preventAD_get_labels_from_filename(file)
                    plotted_data_points += 1
                    try:
                        hearing_loss = preventAD_get_measure_from_csv(subject, visit, data_file, 5) #19 = hearing loss row, 5 = age
                    except Exception as e:
                        print("Error getting CSV file measures for Prevent-AD.")
                        return #skips the plotting
                    
                    if hearing_loss != None: #only visualize if we have a hearing loss measure for subject/visit
                        hearing_loss_list.append(hearing_loss)
                        volume_list.append(volume)
            else:
                removed_data_points += 1

    if len(volume_list) >= 1 and len(hearing_loss_list)>=1: #If there is at least one data point.
        corrplot(volume_list, hearing_loss_list, in_pipeline_name, 'Prevent-AD', 'Left Hippocampal Volumes vs Age (discrepancy QC)', 'Age (months)', 'Hippocampal Volume ($mm^3$)', 'discrepancy')
        print('Generated discrepancy line plots for ' + cache_filename + '/' + pipeline_name + ". Points plotted: " + str(plotted_data_points) + ", points removed: " + str(removed_data_points))
                    
def flag_result_discrepancy(results_for_file, threshold):
    for entry in results_for_file:
        for nested_entry in results_for_file:
            if entry[1] is not None and nested_entry[1] is not None: #are we capturing none's correctly?
                if abs(float(entry[1])-float(nested_entry[1])) > threshold:
                    return True
            else:
                return True
    return False

def get_results_for_file(cache_file_dict, entry_name):
    results_for_file = []
    d = cache_file_dict
    entry = d[entry_name]
    for pipeline in entry:
        results_for_file.append((pipeline, d[entry_name][pipeline]['Result']["result"]))
    return results_for_file

def preventAD_process_bounded_scatter(cache_file, pipeline_x, pipeline_y, lower_bound, upper_bound, x_lim, y_lim):
    d = json.load(open(cache_file))
    ids = list(d.keys())
    not_none_ids = [ i for i in ids if d[i][pipeline_x]["Result"]["result"] is not None and "T1" in i 
                               and d[i][pipeline_y]["Result"]["result"] is not None ]
    no_outliers_ids = [ i for i in not_none_ids if float(d[i][pipeline_x]["Result"]["result"]) < upper_bound and float(d[i][pipeline_x]["Result"]["result"]) > lower_bound and "T1" in i] 
    volume_list_x = [ float(d[i][pipeline_x]["Result"]["result"]) for i in no_outliers_ids ]
    volume_list_y = [ float(d[i][pipeline_y]["Result"]["result"]) for i in no_outliers_ids ]
    XY_plot(volume_list_x, volume_list_y, 'FSL', 'FreeSurfer', cache_file[:-5], 'Left Hippocampal Volumes', 6000, 6000, 'FSL volume (mm$^3$)', 'FreeSurfer volume (mm$^3$)', 'bounded')
    print('Generated bound XY plots for ' + cache_file + '+' + pipeline_x + '/' + pipeline_y + ". Points plotted: " + str(len(volume_list_x)))

def preventAD_process_discrepancy_scatter(cache_file, pipeline_x, pipeline_y, threshold, x_lim, y_lim):
    d = json.load(open(cache_file))
    ids = list(d.keys())
    
    not_none_ids = []
    for i in ids:
        if d[i][pipeline_x]["Result"]["result"] is not None and "T1" in i and d[i][pipeline_y]["Result"]["result"] is not None:
            not_none_ids.append(i)
    
    no_outliers_ids = []
    for i in not_none_ids:
        if not flag_result_discrepancy(get_results_for_file(d, i), threshold):
            no_outliers_ids.append(i)
    
    volume_list_x = [ float(d[i][pipeline_x]["Result"]["result"]) for i in no_outliers_ids ]
    volume_list_y = [ float(d[i][pipeline_y]["Result"]["result"]) for i in no_outliers_ids ]
    XY_plot(volume_list_x, volume_list_y, 'FSL', 'FreeSurfer', cache_file[:-5], 'Left Hippocampal Volumes', 6000, 6000, 'FSL volume (mm$^3$)', 'FreeSurfer volume (mm$^3$)', 'discrepancy')
    print('Generated discrepancy XY plots for ' + cache_file + '+' + pipeline_x + '/' + pipeline_y + ". Points plotted: " + str(len(volume_list_x)))
    
preventAD_data_file = 'Auditory_processing_Registered_PREVENTAD.csv'
preventAD_cache_file = 'Prevent-AD.json'

#########################################################################################################

# Main section of Analyses

cbrain_user = sys.argv[1]
cbrain_password = sys.argv[2]
cbrain_token = cbrain_login(cbrain_user, cbrain_password)

# Downloading phenotipic data from CBRAIN
#cbrain_download_DP_file('Auditory_processing_Registered_PREVENTAD.csv', 318, cbrain_token) #use this line if you know the file name but not the ID. Takes a long time though.
cbrain_download_file(3497558, preventAD_data_file, cbrain_token) #use this line (quicker) if you know the CBRAIN userfileID

with open('Experiment_Definition.yaml') as file: #Load experiment definition
	try:
		experiment_definition  = yaml.safe_load(file)
	except yaml.YAMLError as exception: #yaml file not valid
		print('The Experiment Definition file is not valid')
		print(exception)
	
	for pipeline in experiment_definition['Pipelines']:
		preventAD_process_bounded_line(preventAD_data_file, preventAD_cache_file, pipeline, 1500.0, 6000.0) # Only keeps values between 1500-6000mm^3
		#preventAD_process_bounded_line(preventAD_data_file, preventAD_cache_file, pipeline, 0.0, 99999.0) # Doesn't remove any outliers.
		preventAD_process_discrepancy_line(preventAD_data_file, preventAD_cache_file, pipeline, 450) #average hippocampus is ~3000, and 15% of 3000 is 450
	preventAD_process_bounded_scatter('Prevent-AD.json', 'FSL', 'FreeSurfer', 1500, 6000, 6000, 6000)
	#preventAD_process_bounded_scatter('Prevent-AD.json', 'FSL', 'FreeSurfer', 0, 99999, 6000, 6000)
	preventAD_process_discrepancy_scatter('Prevent-AD.json', 'FSL', 'FreeSurfer', 450, 6000, 6000)
cbrain_logout(cbrain_token)
