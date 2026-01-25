import traceback

def safe_execute(code: str):
    local_env = {}

    try:
        exec(code, {}, local_env)
        return {
            "success": True,
            "output": local_env
        }
    except Exception:
        return {
            "success": False,
            "error": traceback.format_exc()
        }
