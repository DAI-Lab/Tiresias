import psutil
import requests
import argparse
from json import dumps
from time import sleep, time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help="The port to listen on.")
    args = parser.parse_args()

    requests.get("http://localhost:%s/app/device_info/register" % args.port, params={"schema": dumps({
        "cpu_usage": {
            "description": "This table contains information about the CPU usage.",
            "columns": {
                "timestamp": {
                    "type": "float",
                    "description": "Unix timestamp."
                },
                "cpu_count": {
                    "type": "float",
                    "description": "Number of CPUs (including virtual)"
                },
                "cpu_percent": {
                    "type": "float",
                    "description": "Total cpu utilization."
                },
                "cpu_physical": {
                    "type": "float",
                    "description": "Total cpu utilization."
                },
            }
        },
        "disk_usage": {
            "description": "This table contains information about the disk usage.",
            "columns": {
                "timestamp": {
                    "type": "float",
                    "description": "Unix timestamp."
                },
                "disk_space_total": {
                    "type": "float",
                    "description": "Total disk space."
                },
                "disk_space_used": {
                    "type": "float",
                    "description": "Used disk space."
                },
                "disk_space_free": {
                    "type": "float",
                    "description": "Free disk space."
                },
            }
        }
    })})

    while True:
        requests.get("http://localhost:%s/app/device_info/insert" % args.port, params={"update": dumps({
            "cpu_usage": [{
                "timestamp": time(),
                "cpu_count": psutil.cpu_count(), 
                "cpu_percent": psutil.cpu_percent(interval=1),
                'cpu_physical': psutil.cpu_count(logical=False),
            }],
            "disk_usage": [{
                "timestamp": time(),
                "disk_space_total": psutil.disk_usage('/').total,
                "disk_space_used": psutil.disk_usage('/').used,
                "disk_space_free": psutil.disk_usage('/').free,
            }]
        })})
        sleep(10)
