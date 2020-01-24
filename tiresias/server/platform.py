import uuid
import threading
from time import time
from enum import Enum
from tiresias.core import b64_encode
from tiresias.server.handler import handle_task

class State:
    ERROR = 'ERROR'
    RUNNING = 'RUNNING'
    PENDING = 'PENDING'
    COMPLETE = 'COMPLETE'

class Platform(object):

    def __init__(self):
        """
        The Platform object is responsible for managing and executing tasks. It's designed to work
        with a multi-threaded web server and stores everything in-memory.
        """
        self._lock = threading.RLock()
        self._tasks = {}
        self._payloads = {}

    def gc(self, timeout=60):
        """
        Delete the data for any completed tasks and delete any completed tasks that have passed the
        timeout window.
        """
        with self._lock:
            for tid in set(self._tasks.keys()):
                if self._tasks[tid]["status"] == State.COMPLETE:
                    del self._payloads[tid]
                    if time() - self._tasks[tid]["start"] > timeout:
                        del self._tasks[tid]
    
    def run(self):
        """
        Examing the pending tasks and, if there's enough data, start processing the task. Note that
        the task handler will block the platform accepting new tasks and/or data; at some point, we 
        will need to move the task handler into its own process.
        """
        with self._lock:
            for tid in self.tasks(only_pending=True):
                if self._tasks[tid]["count"] < self._tasks[tid]["min_count"]:
                    continue
                self._tasks[tid]["status"] = State.RUNNING
                result, err = handle_task(self._tasks[tid], self._payloads[tid])
                if err:
                    self._tasks[tid]["status"] = State.ERROR
                    self._tasks[tid]["result"] = repr(err)
                else:
                    self._tasks[tid]["result"] = b64_encode(result)
                    self._tasks[tid]["status"] = State.COMPLETE
                    self._tasks[tid]["end"] = time()
    
    def tasks(self, only_pending=False):
        """
        Return a list of tasks.
        """
        with self._lock:
            if not only_pending:
                return self._tasks
            return {k: v for k, v in self._tasks.items() if v["status"] == State.PENDING}
    
    def create(self, task):
        """
        Create a new task.
        """
        with self._lock:
            task["id"] = str(uuid.uuid4())
            task["status"] = State.PENDING
            task["start"] = time()
            task["count"] = 0
            self._tasks[task["id"]] = task
            self._payloads[task["id"]] = []
            return task["id"]
    
    def fetch(self, task_id):
        """
        Fetch a specific task.
        """
        with self._lock:
            return self._tasks[task_id]

    def submit(self, task_id, payload):
        """
        Store the data for a specific task in-memory and update the counters.
        """
        with self._lock:
            task = self._tasks[task_id]
            if task["status"] != State.PENDING:
                return False
            self._payloads[task_id].append(payload)
            self._tasks[task_id]["count"] = len(self._payloads[task_id])
            return True
