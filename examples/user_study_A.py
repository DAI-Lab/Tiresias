"""
# User Study A: Data Contributors
1. Launch a server.
2. Submit a bunch of tasks.
3. Launch the user client.
4. Open the Google Form which will have more instructions.
"""
import tempfile
import subprocess
import webbrowser
from time import sleep
from tiresias.server import remote

server = "http://localhost:3000/"

processes = []
processes.append(subprocess.Popen(['tiresias-server'], stderr=subprocess.DEVNULL))
sleep(1.0)
processes.append(subprocess.Popen(['tiresias', '--synthetic', '--storage_dir', tempfile.mkdtemp()]))
for i in range(10):
    processes.append(subprocess.Popen([
        'tiresias', 
        '--synthetic',
        '--accept_all',
        '--headless',
        "--storage_dir", tempfile.mkdtemp(), 
        "--storage_port", str(8001 + i)
    ], stderr=subprocess.DEVNULL))
sleep(1.0)

remote.create_task(server, {
    "type": "basic",
    "epsilon": 16.0,
    "delta": 1e-5,
    "min_count": 20,
    "featurizer": "SELECT age FROM profile.demographics",
    "aggregator": "median"
})

remote.create_task(server, {
    "type": "integrated",
    "epsilon": 16.0,
    "delta": 1e-5,
    "min_count": 20,
    "featurizer": """
        SELECT 
            income, 
            gender == 'Male' as male, 
            age 
        FROM profile.demographics
    """,
    "model": "LinearRegression",
    "inputs": ["income", "male"],
    "output": "age"
})

while True:
    try:
        sleep(1.0)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        break
