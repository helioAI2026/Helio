import json
from datetime import datetime
from pathlib import Path


class EventLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"sleepiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def log_alert(self, result: dict):
        event = {
            'timestamp': datetime.now().isoformat(),
            'score': float(round(result['score'], 3)),
            'ear': float(round(result['ear'], 3)),
            'perclos': float(round(result['perclos'], 3)),
            'status': result['status'],
        }
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception:
            pass

    def get_session_file(self) -> Path:
        return self.log_file
