import paramiko
from scp import SCPClient

##################################################################################

def login(hostname, username, password, port=22):

    global host
    host = hostname
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # Vulnerability to possible MITM attack

    try:
        client.connect(hostname, port, username, password)
    except paramiko.BadHostKeyException:
        print('Server’s host key could not be verified.')
    except paramiko.AuthenticationException:
        print('Authentication failed.')
    except paramiko.SSHException:
        print('An error occured while connecting\n An SSH session could not be established.')
    else:
        print('Connection to', host, 'established.')

        return client

def logout(client):

    client.close()
    print('Connection to', host, 'was closed.')

def list_data_provider(client, path):

    stdin, stdout, stderr = client.exec_command('ls -r ' + path)
    out, err = ''.join(stdout.readlines()), ''.join(stderr.readlines())

    if err:
        print(err)
    else:
        print(out)
    
    return out.split('\n')

def post_task(client, remote_path):

    cd_path = '/'.join(remote_path.split('/')[:-1])
    shell_name = remote_path.split('/')[-1]
    stdin, stdout, stderr = client.exec_command('cd ' + cd_path + ';sbatch ' + shell_name)
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

def download():
    pass
