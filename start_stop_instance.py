import dotenv
import os
import time
import socket
import tqdm

from FluidStack.client import FluidStack

dotenv.load_dotenv()


client = FluidStack(
    api_key = os.getenv('FLUIDSTACK_APIKEY')
)

def print_all_status():
    for instance in client.instances.list():
        print(f"{instance.name}: {instance.status}")

def get_instance(name):
    return [x for x in client.instances.list() if x.name == name][0]

def try_start(name, tries=60, secs=10):
    instance = get_instance(name)
    for i in range(tries):
        instance = get_instance(name)
        if instance.status == 'running':
            break
        try:
            client.instances.start(instance.id)
        except Exception as e:
            print(f"starting {name}, {i}/{tries}")
        time.sleep(secs)

def try_stop(name, tries=60, secs=10):
    instance = get_instance(name)
    for i in range(tries):
        instance = get_instance(name)
        if instance.status == 'stopped':
            break
        try:
            client.instances.stop(instance.id)
        except Exception as e:
            print(f"stopping {name}, {i}/{tries}")
        time.sleep(secs)

print_all_status()

hostname = socket.gethostname()
if hostname == '5CD2373R4X':
    try_start('jaime2-a100')
elif hostname == 'fs-api-c2d4c704-389b-4750-9c0c-1a2b49da6e1a': #hostname for a100 machine
    try_stop('jaime2-a100')
print_all_status()


