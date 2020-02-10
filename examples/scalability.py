import tempfile
import subprocess
from time import sleep
from tiresias.server import remote

server = "http://localhost:3000/"

# Launch a server + 20 clients
processes = []
processes.append(subprocess.Popen(['tiresias-server'], stderr=subprocess.DEVNULL))
for i in range(20):
    storage_dir = tempfile.mkdtemp()
    storage_port = str(8000 + i)
    processes.append(subprocess.Popen([
        'tiresias', 
        "--storage_dir", storage_dir, 
        "--storage_port", storage_port
    ]))
sleep(5.0)

# Send a median query on the dummy dataset
query_id = remote.create_task(server, {
    "type": "basic",
    "epsilon": 16.0,
    "delta": 1e-5,
    "min_count": 10,
    "featurizer": "SELECT malic_acid FROM dummy.wine",
    "aggregator": "median"
})
sleep(1.0)
print(remote.fetch_task(server, query_id))

# Send a regression query on the dummy dataset
query_id = remote.create_task(server, {
    "type": "integrated",
    "epsilon": 16.0,
    "delta": 1e-5,
    "min_count": 10,
    "featurizer": "SELECT alcohol, malic_acid, ash FROM dummy.wine",
    "model": "LinearRegression",
    "inputs": ["alcohol", "malic_acid"],
    "output": "ash"
})
sleep(1.0)
print(remote.fetch_task(server, query_id))

# Shut down the server + clients
for process in processes:
    process.terminate()
