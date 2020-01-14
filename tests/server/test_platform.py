import threading
from time import sleep
from random import random
from pytest import approx
from tiresias.core import b64_decode
from tiresias.server.platform import Platform, State

def test_platform_basic():
    platform = Platform()
    assert len(platform.tasks()) == 0

    task_id = platform.create({
        "type": "basic",
        "epsilon": 16.0,
        "delta": 1e-5,
        "min_count": 20,
        "featurizer": "SELECT * FROM dummy",
        "aggregator": "mean"
    })
    assert len(platform.tasks()) == 1

    for _ in range(100):
        assert platform.submit(task_id, [random()])

    platform.run()
    task = platform.fetch(task_id)
    assert task["status"] == State.COMPLETE
    assert b64_decode(task["result"]) == approx(0.5, abs=0.1)

def test_platform_basic_multithreaded():
    platform = Platform()
    assert len(platform.tasks()) == 0

    def _run():
        for _ in range(10):
            sleep(0.05)
            platform.run()
    runner = threading.Thread(target=_run)
    runner.start()

    try:
        task_id = platform.create({
            "type": "basic",
            "epsilon": 16.0,
            "delta": 1e-5,
            "min_count": 20,
            "featurizer": "SELECT * FROM dummy",
            "aggregator": "mean"
        })
        assert len(platform.tasks()) == 1
        
        threads = []
        for _ in range(100):
            threads.append(threading.Thread(target=platform.submit, args=(task_id, [random()])))
        for t in threads:
            t.start()

        sleep(0.1)
        task = platform.fetch(task_id)
        assert task["status"] == State.COMPLETE
        assert b64_decode(task["result"]) == approx(0.5, abs=0.1)

    finally:
        runner.join()

def test_platform_integrated():
    platform = Platform()
    assert len(platform.tasks()) == 0

    task_id = platform.create({
        "type": "integrated",
        "epsilon": 10.0,
        "featurizer": "SELECT x1, x2, y FROM profile.example",
        "model": "LinearRegression",
        "inputs": ["x0", "x1"],
        "output": "y",
        "min_count": 20
    })
    assert len(platform.tasks()) == 1

    for _ in range(100):
        assert platform.submit(task_id, [{
            "x0": random(),
            "x1": random(),
            "y": random(),
        }])

    platform.run()
    task = platform.fetch(task_id)
    assert task["status"] == State.COMPLETE
    assert b64_decode(task["result"]).predict

