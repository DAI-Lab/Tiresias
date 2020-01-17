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
print(remote.fetch_task("http://localhost:3000/", query_id))
```

Note that this task requires a minimum of 10 users to complete; however, even if you don't 
have that many users, you can still track the progress of the task using either the Python
API or the web application.

## Tasks

### Basic Tasks
This task would like to access your data by running <SQL>. Your data will be sent to the 
Tiresias server, where it will be combined with at least <COUNT> other users data and 
aggregated into a single <AGGREGATOR> value which will have noise added to make it 
(<EPSILON>, <DELTA>) differentially private to reduce the risk that anyone can figure out
your contribution to this task. Only this differentially private value will be released to 
the data requester.

```
{
    "type": "basic",
    "epsilon": <FLOAT>,
    "delta": <FLOAT>,
    "min_count": <INT>,
    "featurizer": <SQL>,
    "aggregator": <AGGREGATOR>
}
```

### Integrated Task
This task would like to access your data by running <SQL>. Your data will be sent to the 
Tiresias server, where it will be combined with at least <COUNT> other users data and 
used to train a <MODEL> model to predict <OUTPUT>. This model will have noise added to it 
to make it (<EPSILON>, <DELTA>) differentially private to reduce the risk that anyone can 
figure out your contribution to this task. Only this differentially private model will be 
released to the data requester.

```
{
    "type": "integrated",
    "epsilon": <FLOAT>,
    "delta": <FLOAT>,
    "featurizer": <SQL>,
    "model": <MODEL>,
    "inputs": [<VARNAME>, <VARNAME>, ...],
    "output": <VARNAME>
}
```

### Bounded Task
This task would like to access your data by running <SQL>. Before sending your data to 
the server, we will add noise to make it (<EPSILON>, <DELTA>) differentially private to 
reduce the risk that anyone can figure out your contribution to this task.

```
{
    "type": "bounded",
    "epsilon": <FLOAT>,
    "delta": <FLOAT>,
    "featurizer": <SQL>,
    "bounds": {
        "<VARNAME>": {
            "type": "<BOUND_TYPE>",
            "default": <FINITE_VALUE>,
            "values": [<FINITE_VALUE>, <FINITE_VALUE>],
        }
    }
}
```

### Gradient Task
This task would like to access your data by running <SQL> and the computing the gradients
for a model. Note that your actual data will not be sent to the server, only the gradients, 
and even then, we add noise to the gradients before sending it to make it (<EPSILON>, <DELTA>) 
differentially private to reduce the risk that anyone can figure out your contribution to 
this task.

```
{
    "type": "gradient",
    "epsilon": 10.0,
    "delta": 1e-5,
    "lr": 0.01,
    "featurizer": "SELECT profile_picture, y FROM profile.example",
    "model": ...,
    "loss": b64_encode(torch.nn.functional.mse_loss),
    "inputs": "profile_picture",
    "input_type": "image",
    "output": ["y"],
}
```

```
{
    "type": "gradient",
    "epsilon": 10.0,
    "delta": 1e-5,
    "lr": 0.01,
    "featurizer": "SELECT x1, x2, y FROM profile.example",
    "model": ...,
    "loss": b64_encode(torch.nn.functional.cross_entropy_loss),
    "inputs": ["x1", "x2"],
    "input_type": "vector",
    "output": ["y"],
}
```
