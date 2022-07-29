import paramiko

##################################################################################

def login(hostname, username, password, port=22):
    global host
    host = hostname
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Vulnerability to possible MITM attack
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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

def logout():
    pass

def list_data_provider():
    pass

def upload():
    pass

def post_task():
    pass

def get_all_tasks():
    pass

def download():
    pass
