from tiresias.client.storage import execute_sql
from tiresias.client.handler.basic import handle_basic
from tiresias.client.handler.bounded import handle_bounded
from tiresias.client.handler.integrated import handle_integrated
from tiresias.client.handler.gradient import handle_gradient

def handle_task(storage_dir, task):
    dispatcher = {
        "basic": handle_basic,
        "bounded": handle_bounded,
        "integrated": handle_integrated,
        "gradient": handle_gradient,
    }
    if task["type"] not in dispatcher:
        return None, ValueError("Unknown query type.")
    func = dispatcher[task["type"]]

    try:
        data = execute_sql(storage_dir, task["featurizer"])
        return func(task, data), None
    except Exception as e:
        return None, e
