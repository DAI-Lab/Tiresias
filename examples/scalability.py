"""
1. Launch a t2.large instance and run

    python examples/scalability.py --launch_server

2. Launch N t2.medium instances and run

    python examples/scalability.py --launch_clients

so each instance will have 20 clients for a total of 20*N clients.

3. On your own computer, run

    python examples/scalability.py --run_scalability

to submit tasks and measure completion time.
"""
import argparse
import tempfile
import subprocess
import pandas as pd
from tqdm import tqdm
from time import sleep
from tiresias.server import remote

server = "http://3.92.204.185:3000/" # replace with server ip

def launch_server():
    subprocess.Popen(['tiresias-server'], stderr=subprocess.DEVNULL)

def launch_clients(nb_clients):
    for i in range(nb_clients):
        storage_dir = tempfile.mkdtemp()
        storage_port = str(8000 + i)
        subprocess.Popen([
            'tiresias', 
            "--storage_dir", storage_dir, 
            "--storage_port", storage_port,
            "--server_url", server,
            "--accept_all",
            "--headless",
            "--synthetic",
        ])

def run_scalability(num_trials=20):
    results = []

    for _ in range(num_trials):
        for sample_size in tqdm([20, 40, 60, 80, 100, 120, 140, 160, 180, 200]):
            query_id = remote.create_task(server, {
                "name": "Median Age",
                "type": "basic",
                "epsilon": 2.0,
                "delta": 1e-5,
                "min_count": sample_size,
                "featurizer": "SELECT age FROM profile.demographics",
                "aggregator": "median"
            })
            for i in range(5):
                sleep(1.0 + i)
                query = remote.fetch_task(server, query_id)
                if "end" in query:
                    break
            results.append({
                "task": "median",
                "nb_users": sample_size,
                "running_time": query["end"] - query["start"]
            })
            sleep(1.0)

            query_id = remote.create_task(server, {
                "name": "Age Prediction",
                "type": "integrated",
                "epsilon": 4.0,
                "delta": 1e-5,
                "min_count": sample_size,
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
            for i in range(5):
                sleep(1.0 + i)
                query = remote.fetch_task(server, query_id)
                if "end" in query:
                    break
            results.append({
                "task": "regression",
                "nb_users": sample_size,
                "running_time": query["end"] - query["start"]
            })

    df = pd.DataFrame(results)
    df = df.groupby(["nb_users", "task"]).agg("mean").reset_index()
    df[df["task"]=="median"].to_csv("median.csv", index=False)
    df[df["task"]=="regression"].to_csv("regression.csv", index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch_server', action='store_true')
    parser.add_argument('--launch_clients', action='store_true')
    parser.add_argument('--run_scalability', action='store_true')
    args = parser.parse_args()

    if args.launch_server:
        launch_server()
        sleep(1.0)
    if args.launch_clients:
        launch_clients(nb_clients=20)
        sleep(10.0)
    if args.run_scalability:
        results = run_scalability()
        results.to_csv("scalability.csv", index=False)
        quit()

    while True:
        sleep(1.0)
