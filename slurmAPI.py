import paramiko
from scp import SCPClient

##################################################################################

def login(hostname, username, password, port=22):
    """Establishes SSH connection.
    
    Connects to an SSH server and authenticates to it. The server's host key is 
    checked against The loaded system host (load_system_host_keys()). If the server's
    hostname is not found, It is automatically added (set_missing_host_key_policy()).

    Args:
        hostname: A string representing the server to connect to.
        username: A string representing the username to authenticate as.
        password: A string representing password to authenticate.
        port: An integer representing the server port to connect to (default 22).

    Returns:
        A paramiko.client.SSHClient object representing the client object with SSH connection.
    """

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, password)
    print('Connection to', host, 'established.')

    return client


def logout(client):
    """Tears down the SSH connection.

    Closes the paramiko.client.SSHClient object and it's underlying '.Transport'.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to be closed.

    Returns:
        None.
    """

    client.close()
    print('Connection was closed.')


def list_data_provider(client, path, exit_status=True):
    """Lists existing files in the given path.

    Generates a list of the available files in the input path using SSH connection
    provided by client object to execute 'ls -r' command. If the command executes
    with no error,(exit status is printed and) the list is returned otherwise the 
    stderr (and exit status) is printed and None is returned.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to execute the shell command.
        path: A string representing the path of the designated location.
        exit_status: boolean representing a flag when true the function prints the exit status of the executed shell command (default True).

    Returns:
        A list of strings where each string represents the available files in the given path
        if the command executes with no error otherwise None.
    """

    stdin, stdout, stderr = client.exec_command('ls -r ' + path)
    out, err = ''.join(stdout.readlines())[:-1], ''.join(stderr.readlines())[:-1]

    if exit_status:
        if err:
            print(err, '\nExit code for \"ls -r\" command in list_data_provider:', str(stderr.channel.recv_exit_status()))
        else:
            print('Exit code for \"ls -r\" command in list_data_provider:', str(stdout.channel.recv_exit_status()))
            return out.split('\n')
    else:
        if err:
            print(err) 
        else:
            return out.split('\n')


def post_task(client, path, exit_status=True):
    """Posts a task located in the given path to Slurm.

    Submits a task located in the input path to Slurm using the SSH connection 
    provided by client object to execute 'cd' command to first change the 
    current working directory to where the task file is located (cd path),
    then submit the task (task_file) by executing 'sbatch' command, and 
    prints the stdout (and the exit status) if the command executes with 
    no error otherwise prints the error (and the exit status).

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to execute shell commands.
        path: A string representing the path to the task file's location.
        exit_status: boolean representing a flag when true the function prints the exit status of the executed shell command (default True).
    
    Returns:
        None.
    """

    cd_path = '/'.join(path.split('/')[:-1])
    task_file = path.split('/')[-1]
    stdin, stdout, stderr = client.exec_command('cd ' + cd_path + ';sbatch ' + task_file)
    out, err = ''.join(stdout.readlines())[:-1], ''.join(stderr.readlines())[:-1]
    
    if exit_status:
        if err:
            print(err, '\nExit code for \"sbatch\" command in post_task:', str(stderr.channel.recv_exit_status()))
        else:
            print(out, '\nExit code for \"sbatch\" command in post_task:', str(stdout.channel.recv_exit_status()))
    else:
        print(err) if err else print(out)


def get_all_tasks(client, exit_status=True):
    """Retrieves all tasks in queue.

    Gets all tasks by executing an 'squeue' command using the SSH connection
    provided by client object. If the command executes with no error (prints the exit status and)
    it parses the stdout, structures it into a dictionary and returns it otherwise, it prints
    the stderr (and the exit status).

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to execute the shell command.
        exit_status: A boolean representing a flag when true the function prints the exit status of the executed shell command (default True).

    Returns:
        A dictionary mapping each job_id to a dictionary that maps task information headers to their
        respective values if the command executes with no error otherwise None. Example of 
        returned dictionary:

        {'40749727': 
                    {
                        'USER': 'johndoe',
                        'ACCOUNT': 'def-joh_C',
                        'NAME': '4carm0.1.24',
                        'ST': 'R',
                        'TIME_LEFT': '1:26',
                        'NODES': '3',
                        'CPUS': '4',
                        'TRES_PER_N': 'N/A',
                        'MIN_MEM': '16',
                        'NODELIST (REASON)': 'cdr[1705-1706,1712] (None)'
                    }

        }

        Returned keys and values are all strings.
    """

    stdin, stdout, stderr = client.exec_command('squeue')
    out, err = ''.join(stdout.readlines())[:-1], ''.join(stderr.readlines())[:-1]

    if exit_status:
        if err:
            print(err, '\nExit code for \"squeue\" command in get_all_tasks:', str(stderr.channel.recv_exit_status()))
        else:
            print('Exit code for \"squeue\" command in get_all_tasks:', str(stdout.channel.recv_exit_status()))
            out = [i.split()[:10] + [' '.join(i.split()[10:])] for i in out.split('\n')]
            return dict(zip([i[0] for i in out[1:]], (map(lambda x: dict(zip(out[0][1:], x[1:])), out[1:]))))
    else:
        if err:
            print(err)
        else:
            out = [i.split()[:10] + [' '.join(i.split()[10:])] for i in out.split('\n')]
            return dict(zip([i[0] for i in out[1:]], (map(lambda x: dict(zip(out[0][1:], x[1:])), out[1:]))))


def upload(client, local_path, remote_path, recursive=False):
    """Transfers files and directoreis from local machine to remote machine.

    Uploads files and directories using scp module and client object.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to get the transport from and initialize the SCPClient object.
        local_path: A string representing a single path, or a list of strings representing paths on local machine to be transferred.
                    recursive must be True to transfer directories.
        remote_path: A string representing the path on the remote machine where the files will be transferred to.
        recursive: A boolean to allow the transfer of files and directories recursively (default False).
    
    Returns:
        None.
    """

    scp = SCPClient(client.get_transport())
    scp.put(local_path, remote_path, recursive)
    scp.close()


def download(client, remote_path, local_path, recursive=False):
    """Transfers files and directories from remote machine to local machine.

    Downloads files and directories using scp module and client object.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to get the transport from and initialize the SCPClient object.
        remote_path: A string representing a single path on remote machine to be transferred.
                     recursive must be True to transfer directories.
        local_path: A string representing the path on the local machine where the files will be transferred to.
        recursive: A boolean to allow the transfer of files and directories recursively (default False).

    Returns:
        None.
    """

    scp = SCPClient(client.get_transport())
    scp.get(remote_path, local_path, recursive)
    scp.close()


def execute_command(client, cmd, exit_status=True):
    """Executes a given shell command.

    Executes a given shell command using the SSH connection provided by client object
    and prints the stdout if the command executes with no error otherwise prints stderr.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to execute the given shell command.
        cmd: A string representing the shell command to be executed.
        exit_status: A boolean allows the function to print the exit status of the executed shell command (default True).

    Returns:
        None.
    """

    stdin, stdout, stderr = client.exec_command(cmd)
    out, err = ''.join(stdout.readlines())[:-1], ''.join(stderr.readlines())[:-1]

    
    if exit_status:
        if err:
            print(err, '\nExit code for ' + "\"" + cmd + "\"" + ' command in execute_command:', str(stderr.channel.recv_exit_status()))
        else:
            print(out, '\nExit code for ' + "\"" + cmd + "\"" + ' command in execute_command:', str(stdout.channel.recv_exit_status()))
    else:
        print(err) if err else print(out)
