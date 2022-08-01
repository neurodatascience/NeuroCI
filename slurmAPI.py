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
    global host
    host = hostname
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # Vulnerability to possible MITM attack
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
    print('Connection to', host, 'was closed.')


def list_data_provider(client, path):
    """Lists existing files in the given path.

    Generates a list of the available files in the input path using SSH connection
    provided by client object to execute 'ls -r' command. If the command executes
    with no error, the stdout is printed and the list is returned otherwise the 
    stderr is printed and None is returned.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to execute the shell command.
        path: A string representing the path of the designated location.

    Returns:
        A list of strings where each string represents the available files in the given path
        if the command executes with no error otherwise None.
    """

    stdin, stdout, stderr = client.exec_command('ls -r ' + path)
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    if err:
        print(err[:-1])
    else:
        print(out[:-1])
        return out.split('\n')


def post_task(client, path):
    """Posts a task located in the given path to Slurm.

    Submits a task located in the input path to Slurm using the SSH connection 
    provided by client object to execute 'cd' command to first change the 
    current working directory to where the task file is located (cd path),
    then submit the task (task_file) by executing 'sbatch' command, and 
    prints the stdout if the command executes with no error otherwise
    prints the error.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to execute shell commands.
        path: A string representing the path to the task file's location.
    
    Returns:
        None.
    """

    cd_path = '/'.join(path.split('/')[:-1])
    task_file = path.split('/')[-1]
    stdin, stdout, stderr = client.exec_command('cd ' + cd_path + ';sbatch ' + task_file)
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    print(err[:-1]) if err else print(out[:-1])


def get_all_tasks(client):
    """Gets all tasks in queue.

    Retrieves all tasks by executing a 'squeue' command using the SSH connection
    provided by client object. If the command executes with no error it parses
    the stdout, structures it into a dictionary and returns it otherwise, it prints
    the stderr.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to execute the shell command.

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
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    if err:
        print(err[:-1])
    else:
        out = [i.split()[:10] + [' '.join(i.split()[10:])] for i in out.split('\n')]
        return dict(zip([i[0] for i in out[1:-1]], (map(lambda x: dict(zip(out[0][1:], x[1:])), out[1:-1]))))


def upload(client, local_path, remote_path, recursive=False):
    """Transfers files and directoreis from local machine to remote machine.

    Uploads files and directories using scp module and client object.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to get the transport from and initialize the SCPClient object.
        local_path: A string representing a single path, or a list of strings representing paths on local machine to be transferred.
                    recursive must be True to transfer directories.
        remote_path: A string representing the path on the remote machine where the files will be transferred to.
        recursive: A boolean to allow the transfer of files and directories recursively.
    
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
        recursive: A boolean to allow the transfer of files and directories recursively.

    Returns:
        None.
    """

    scp = SCPClient(client.get_transport())
    scp.get(remote_path, local_path, recursive)
    scp.close()


def execute_command(client, cmd):
    """Executes a given shell command.

    Executes a given shell command using the SSH connection provided by client object
    and prints the stdout if the command executes with no error otherwise prints stderr.

    Args:
        client: A paramiko.client.SSHClient object with SSH connection to execute the given shell command.
        cmd: A string representing the shell command to be executed.

    Returns:
        None.
    """

    stdin, stdout, stderr = client.exec_command(cmd)
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    print(err[:-1]) if err else print(out[:-1])
        

