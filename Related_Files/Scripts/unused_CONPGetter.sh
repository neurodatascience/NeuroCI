#!/bin/bash

#Run this script from the directory where you want the dataset to be located

#Before running this script, install dependencies:
#sudo apt-get install git-annex-standalone
#sudo apt-get install datalad
#sudo apt-get install neurodebian
#git config --global user.name "yourusername" git config --global user.email "your.name@your.institution.ca"

#Clones and installs the CONP dataset in the current directory
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
echo "$current_time ------------------------ Command: datalad install -r http://github.com/CONP-PCNO/conp-dataset ------------------------" >> logCONP.txt
echo "$current_time ------------------------ Command: datalad install -r http://github.com/CONP-PCNO/conp-dataset ------------------------"
datalad install -r http://github.com/CONP-PCNO/conp-dataset &>> logCONP.txt

cd ./conp-dataset/projects

#Loop through all directories getting the dataset
for d in * ; do
	current_time=$(date "+%Y.%m.%d-%H.%M.%S")
    echo "$current_time ------------------------ Command: datalad get $d ------------------------">> ./../../logCONP.txt
	echo "$current_time ------------------------ Command: datalad get $d ------------------------"
	datalad get $d &>> ./../../logCONP.txt
done

echo " " >>  ./../../logCONP.txt
