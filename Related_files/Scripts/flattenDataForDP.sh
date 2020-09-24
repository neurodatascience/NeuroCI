#!/bin/bash
#Place in the path of your Flat Data Provider directory and run from within it to copy your Datalad files to it.

for f in PATH_REDACTED/cbrain-conp/conp-dataset/projects/preventad-open-bids/BIDS_dataset/sub*/ses*/anat/*.nii.gz; do
        #vari=$(readlink -f "$f")
        #fname=$(basename "$f")
        #echo $vari
        echo $f
        echo ""
        cp -lLr $f .
done
