The `Tiresias` platform is a differentially private data marketplace which 
helps connect users and researchers, enabling them to collaboratively assemble 
diverse datasets and develop novel machine learning methods and applications.

To install the `tiresias` library in development mode, run the following:

    git clone http://github.com/DAI-Lab/p2p-data
    pip install -e .
    pytest

## Usage
In actual usage scenarios, you should primarily interact with the `tiresias` 
platform through the `tiresias-server` and `tiresias-client` binaries. For a 
typical deployment, you'll want to launch `tiresias-server` on a remote machine 
and ask each user to install the `tiresias-client` and point it at the server.

    server$ tiresias-server --port 3000
    user1$ tiresias-client --server http://server:3000/
    user2$ tiresias-client --server http://server:3000/
    user3$ tiresias-client --server http://server:3000/

### Data Collection
The users can then install data collection applications which will interact
locally with the client and insert rows into their personal data store. For 
example, suppose we wrote a data collection application which collects 
information about CPU usage:

```
from time import sleep
import tiresias.client.api as api

api.register_app(8000, "device_info", {
    "cpu_usage": {
        "description": "Real time CPU usage statistics.",
        "columns": {
            "percent_used": {
                "type": "float", 
                "description": "What percent of CPU is used?"
            },
            "number_of_cores": {
                "type": "float", 
                "description": "How many logical cores are there?"
            }
        }
    }
})

while True:
    api.insert_payload(8000, "device_info", {
        "cpu_usage": [{
            "percent_used": get_cpu_usage(),
            "number_of_cores": get_num_cores(),
        }]
    })
    sleep(1)
```

If every user runs the above application, then each of their personal data 
stores would begin to populate with information about their CPU usage history.
Note that at this point, no one has access to a user's personal data store 
except for the user's themselves.

### Example Query
If the user wants to contribute their data to a query, then they can check 
with the server and see if there are any pending queries. For example, suppose
a researcher wanted to estimate the average CPU usage in a differentially 
private manner. They could submit the following query to the server, allowing
users to see it and potentially contribute their data:

```
import tiresias.server.api as api

query_id = api.create_query("http://<server_ip>:3000/", {
    "type": "basic",
    "epsilon": 1.0,
    "featurizer": "SELECT avg(percent_used) FROM device_info.cpu_usage",
    "aggregator": "mean"
})
```

The SQL-based `featurizer` field provides a rich syntax for expressing 
complicated relationships (e.g. joining data from multiple applications - 
for example, examining the relationship between CPU usage and browsing 
history), providing numerous dimensions to explore. In this example, however,
we're simply computing the average CPU usage for this specific user and
using that as the feature.

After giving the users some time to review the query and contribute their
data, the researcher can then check on the status of the query:

```
query = api.fetch_query("http://localhost:3000/", query_id)
print("The average CPU usage was: %s" % query["result"])
```

This will return a value that is differentially private with respect to
each individual user who chose to contribute their data to the query. Note
that this is only a very simple example of a query; examples of more complex 
queries such as bounded queries, machine learning queries, and federated 
learning queries are shown below.

## Queries and Mechanisms
This section describes the four primary types of queries the `tiresias` 
platform supports.

### Basic Queries
Basic queries take the below form where the featurizer is a SQL script that 
returns a single scalar value and the aggregator is one of: `mean`, `median`, 
`count`, or `sum`.

```
query_id = api.create_query("http://<server_ip>:3000/", {
    "type": "basic",
    "epsilon": 1.0,
    "featurizer": <SQL>,
    "aggregator": <AGGREGATOR>
})
```

This type of query returns a single value correponding to the aggregation 
of all the features.

### Bounded Queries
Bounded queries take the below form where the featurizer is a SQL script that 
returns a single row (e.g. a dictionary). In addition, bounded queries must 
specify the bounds for each value in the dictionary. Bounds can be either a 
`set` of values from a finite domain or a `range` of values with a finite 
minimum or maximum.

```
query_id = api.create_query("http://<server_ip>:3000/", {
    "type": "bounded",
    "epsilon": 1.0,
    "featurizer": "SELECT species, age FROM profile.pets",
    "bounds": {
        "species": {
            "type": "set", 
            "default": "dog",
            "values": ["cat", "dog"], 
        },
        "age": {
            "type": "range", 
            "low": 0.0, 
            "high": 100.0
        },
    }
})
```

This type of query returns a list of dictionaries, where each dictionary 
corresponds to the features for a particular user.

### Machine Learning Queries
Machine learning queries take the below form where the featurizer is a SQL 
script that returns a single row (e.g. a dictionaries). In addition, machine 
learning queries specify an aggregator which corresponds to a specific model.

For example, consider a simple Gaussian Naive Bayes query where the inputs and
outputs are `[x1, x2]` and `y` respectively.
```
query_id = api.create_query("http://<server_ip>:3000/", {
    "type": "machine_learning",
    "epsilon": 10.0,
    "featurizer": "SELECT x1, x2, y FROM example_app.tableA",
    "aggregator": {
        "model": "GaussianNB",
        "inputs": ["x0", "x1"],
        "output": "y",
        "bounds": [(0.0, 1.0), (0.0, 1.0)]
    }
})
```
Note that the model specification is provided in the aggregator field and that
we specify the bound for each input variable.

Here's another example of a logistic regression query where the inputs and
outputs are `[x1, x2]` and `y` respectively.
```
query_id = api.create_query("http://<server_ip>:3000/", {
    "type": "federated_learning",
    "epsilon": 10.0,
    "featurizer": "SELECT x1, x2, y FROM example_app.tableA",
    "aggregator": {
        "model": "LogisticRegression",
        "inputs": ["x0", "x1"],
        "output": "y",
        "data_norm": 2.0
    }
})
```
Note that in this case, while we don't need to specify bounds, we do need to 
specify the maximum norm of any particular row in the data.

### Federated Learning Queries
Federated learning queries take the below form where the featurizer is a SQL 
script that returns a list of rows (e.g. a list of dictionaries). In addition, 
federated learning queries specify an aggregator which corresponds to a deep
learning model.

For example, consider a simple linear regression query where the inputs and
outputs are `[x1, x2]` and `y` respectively.
```
query_id = api.create_query("http://<server_ip>:3000/", {
    "type": "federated_learning",
    "epsilon": 10.0,
    "featurizer": "SELECT x1, x2, y FROM example_app.tableA",
    "aggregator": {
        "lr": 1e-4,
        "loss": "MSE",
        "model": "Linear",
        "inputs": ["x1", "x2"],
        "outputs": ["y"],
    }
})
```

We can easily switch to a slightly more complex model with multiple hidden 
layers (e.g. a feedforward neural network with two hidden layers and ReLU
activation functions) by slightly modifying our query.
```
query_id = api.create_query("http://<server_ip>:3000/", {
    "type": "federated_learning",
    "epsilon": 10.0,
    "featurizer": "SELECT x1, x2, y FROM example_app.tableA",
    "aggregator": {
        "lr": 1e-4,
        "loss": "MSE",
        "model": "MultilayerPerceptron",
        "shape": [10, 10],
        "activation": "ReLU",
        "inputs": ["x0", "x1"],
        "outputs": ["y"],
    }
})
```

The federated learning framework is also theoretically capable of supporting 
more complex types of models such as recurrent neural networks for natural 
language processing and convolutional neural networks for computer vision; we
will implement support for these additional features in the near future.
