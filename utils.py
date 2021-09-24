from contextlib import contextmanager
import datetime
import time
import yaml


@contextmanager
def measure_time():
    start = time.time()
    yield lambda: datetime.timedelta(seconds=(time.time() - start))


def get_yaml_file(file, name):
    with open(file) as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exception:  # yaml file not valid
            print(f'The {name} file ({file.name}) is not valid')
            print(exception)
