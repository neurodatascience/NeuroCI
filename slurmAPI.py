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
        client: A paramiko.client.SSHClient object with SSH connection to execute command.
        path: A string representing the path of the designated location.

    Returns:
        A list of strings where each string represents the available files in the given path
        if the command executes with no error otherwise None.
    """

    stdin, stdout, stderr = client.exec_command('ls -r ' + path)
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    if err:
        print(err)
    else:
        print(out)
    
    return out.split('\n')


def post_task(client, path):

    cd_path = '/'.join(remote_path.split('/')[:-1])
    shell_file = remote_path.split('/')[-1]
    stdin, stdout, stderr = client.exec_command('cd ' + cd_path + ';sbatch ' + shell_file)
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    if err:
        print(err)
    else:
        print(out)


def get_all_tasks(client):

    stdin, stdout, stderr = client.exec_command('squeue')
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    if err:
        print(err)
        return None
    else:
        out = [i.split()[:10] + [' '.join(i.split()[10:])] for i in out.split('\n')]
        return dict(zip([i[0] for i in out[1:-1]], (map(lambda x: dict(zip(out[0][1:], x[1:])), out[1:-1]))))


def upload(client, local_path, remote_path, recursive=False):

    scp = SCPClient(client.get_transport())
    scp.put(local_path, remote_path, recursive)
    scp.close()


def download(client, remote_path, local_path, recursive=False):

    scp = SCPClient(client.get_transport())
    scp.get(remote_path, local_path, recursive)
    scp.close()


def execute_command(client, cmd):

    stdin, stdout, stderr = client.exec_command(cmd)
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    if err:
        print(err)
    else:
        print(out)

