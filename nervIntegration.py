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
    print("Using the experiment identifier: " + hashed)
    
    #Make a unique directory for this experiment, using the hash, copy cache files to it
    subprocess.call(["mkdir", hashed])
    cp_command = "cp *.json " + hashed
    subprocess.call(cp_command, shell=True)
    
    #Copy files to NeRV server
    print("Copying newest cache files to the NerV visualization server...")
    subprocess.call(["scp", "-o", "StrictHostKeyChecking=no", "-r", hashed, "ubuntu@206.167.181.134:~/nerv-prototype/nerv/data/"])

    #Instruct the user how to access the visualization
    print("An interactive visualization for the results of this NeuroCI run has been created using the NeRV tool.")
    print("To access this visualization please locally run the command 'ssh -L 80:localhost:80 -i <path_to_your_ssh_key> ubuntu@206.167.181.134' to access the server, and then visit http://localhost/" + hashed + " in your browser.")
