An open source system for privacy preserving machine learning.

## Usage
To get started with `Tiresias`, run the following commands to install the library.

```
pip install -e .
pytest
```

### Platform Maintainer
To launch the *platform server*, run the below command and note your server's IP address.

> tiresias-server --port 3000

If you navigate to `http://<SERVER_IP>:3000/` in your web browser, you'll see a list of open 
tasks. This list will initially be empty - we will demonstrate how you can submit tasks to 
the platform in a later section.

### Data Contributor
Now that your platform server is up and running, you can ask your *data contributors* to 
launch the *user client* and point it at your server with the below command.

> tiresias --server http://localhost:3000/

The *user client* comes with a few built-in dummy datasets that you can play with; however, 
it's up to the *data contributors* to choose what additional data collection applications 
they want to install (i.e. an app that tracks screen time).

The *user client* automatically opens the user interface in your default web browser. The 
user interface will show you any open tasks that you can choose to contribute to, as well 
as a list of the columns that are being collected in your personal data store.

### Data Requestor
Now that you have both the *platform server* and one or more *user clients* up and running, 
we can start writing tasks using the Tiresias Python API. Here's an example of a basic task 
which estimates the median values of the `malic_acid` field in the included `wine` dataset.

```
from tiresias.server import remote
query_id = remote.create_task("http://localhost:3000/", {
    "type": "basic",
    "epsilon": 16.0,
    "delta": 1e-5,
    "min_count": 10,
    "featurizer": "SELECT malic_acid FROM dummy.wine",
    "aggregator": "median"
})
print(remote.fetch_task(server, query_id))
```

Note that this task requires a minimum of 10 users to complete; however, even if you don't 
have that many users, you can still track the progress of the task using either the Python
API or the web application.
