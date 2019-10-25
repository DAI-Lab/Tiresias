"""
The `tiresias.worker` module is responsible for providing the server-side
computations needed to handle a query. This module is currently tightly 
integrated with the `tiresias.server` module but will eventually be split
out to allow for single-server multiple-worker computation models.
"""
from .handler import handle