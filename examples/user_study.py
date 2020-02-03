"""
# User Study A: Data Contributors
1. Launch a server.
2. Submit a bunch of tasks.
3. Launch the user client.
4. Open the Google Form which will have more instructions.
"""
import argparse
import tempfile
import subprocess
import webbrowser
from time import sleep
from tiresias.server import remote

server = "http://localhost:3000/"

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=8000, help="The port to listen on.")
args = parser.parse_args()

processes = []
processes.append(subprocess.Popen(['tiresias-server'], stderr=subprocess.DEVNULL))
print("Launching the server...")
sleep(1.0)
processes.append(subprocess.Popen(['tiresias', '--synthetic', '--storage_dir', tempfile.mkdtemp(), "--storage_port", str(args.port)]))
print("Launching the client...")
for i in range(1, 9):
    print("Launching more clients...")
    processes.append(subprocess.Popen([
        'tiresias', 
        '--synthetic',
        '--accept_all',
        '--headless',
        "--storage_dir", tempfile.mkdtemp(), 
        "--storage_port", str(args.port + i)
    ], stderr=subprocess.DEVNULL))
sleep(1.0)

remote.create_task(server, {
    "name": "Median Age",
    "type": "basic",
    "epsilon": 2.0,
    "delta": 1e-5,
    "min_count": 10,
    "featurizer": "SELECT age FROM profile.demographics",
    "aggregator": "median"
})

remote.create_task(server, {
    "name": "Age Prediction",
    "type": "integrated",
    "epsilon": 4.0,
    "delta": 1e-5,
    "min_count": 128,
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

remote.create_task(server, {
    "name": "Screen Time",
    "type": "bounded",
    "epsilon": 8.0,
    "delta": 1e-5,
    "min_count": 52,
    "featurizer": """
        SELECT application_type, COUNT(application_type) as frequency
        FROM screen_time.events 
        JOIN screen_time.types ON screen_time.events.application_name = screen_time.types.application_name
        WHERE screen_time.events.event_type == 'open'
        GROUP BY application_type
    """,
    "bounds": {
        "application_type": {
            "type": "set",
            "default": "other",
            "values": ["browser", "development", "communication"]
        },
        'frequency': {
            "type": "range",
            "low": 0,
            "high": 100,
        }
    }
})

while True:
    try:
        sleep(1.0)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        break
