import yaml
import hashlib
import subprocess

#Copy the cache files to the server where NeRV can deal with them.

with open('Experiment_Definition.yaml') as file: #Load experiment definition
    try:
        experiment_definition  = yaml.safe_load(file)
    except yaml.YAMLError as exception: #yaml file not valid
        print('The Experiment Definition file is not valid')
        print(exception)
     
    #Make a hash of the experiment name in the experiment definition, keep first 12 chars   
    name_string = experiment_definition['Experiment_name']
    m = hashlib.md5()
    m.update(name_string.encode("utf-8"))
    hashed = str(m.hexdigest())[0:12]
    print(hashed)
    
    #Make a unique directory for this experiment, using the hash, copy cache files to it
    subprocess.call(["mkdir", hashed])
    cp_command = "cp *.json " + hashed
    subprocess.call(cp_command, shell=True)
    
    #copy files to NeRV server
    subprocess.call(["scp", "-o", "StrictHostKeyChecking=no", "-r", hashed, "ubuntu@206.167.181.134:~/nerv-prototype/nerv/data/"])
