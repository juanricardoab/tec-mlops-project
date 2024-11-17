from distutils.util import strtobool
from os import getenv


def initialize_fastapi_server_debugger_if_needed():
    if strtobool(getenv("DEBUG", "False")):
        import multiprocessing

        if multiprocessing.current_process().pid > 1:
            import debugpy

            debugpy.listen(("0.0.0.0", 10019))
            print(
                "⏳ VS Code debugger can now be attached, press F5 in VS Code ⏳",
                flush=True,
            )
            debugpy.wait_for_client()
            print("🎉 VS Code debugger attached, enjoy debugging 🎉", flush=True)
