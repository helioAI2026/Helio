import sys

from src.app import App
from src.eye_state import ModelMissingError

if __name__ == "__main__":
    try:
        app = App()
    except ModelMissingError as e:
        sys.exit(str(e))
    app.run()
