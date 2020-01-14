from tiresias.core import b64_encode, b64_decode
from tiresias.server.handler.basic import handle_basic
from tiresias.server.handler.bounded import handle_bounded
from tiresias.server.handler.integrated import handle_integrated
from tiresias.server.handler.gradient import handle_gradient

def handle_task(task, data):
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
        return func(task, data), None
    except Exception as e:
        return None, e
