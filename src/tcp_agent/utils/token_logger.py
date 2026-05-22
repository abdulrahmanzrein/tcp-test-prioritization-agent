
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import deque

class TokenUsageLogger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TokenUsageLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, log_dir: str = "logs"):
        if self._initialized:
            return
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "token_usage.log"
        self.summary_file = self.log_dir / "token_usage_summary.json"
        
        # Thread safety
        self.write_lock = threading.Lock()
        
        # In-memory history for "per minute" calculation (last 60 seconds)
        self.history = deque() 
        self.history_lock = threading.Lock()
        
        self._initialized = True

    def log_request(self, model_name: str, input_tokens: int, output_tokens: int, status: str = "success"):
        now = time.time()
        total_tokens = input_tokens + output_tokens
        
        # 1. Update in-memory history
        with self.history_lock:
            self.history.append({
                "timestamp": now,
                "tokens": total_tokens,
                "input": input_tokens,
                "output": output_tokens
            })
            # Clean up old entries (> 60s)
            while self.history and self.history[0]["timestamp"] < now - 60:
                self.history.popleft()
            
            # Calculate metrics
            tpm = sum(entry["tokens"] for entry in self.history)
            rpm = len(self.history)
            ipm = sum(entry["input"] for entry in self.history)
            opm = sum(entry["output"] for entry in self.history)

        # 2. Write to log file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "status": status,
            "tpm_window": tpm,
            "rpm_window": rpm,
            "ipm_window": ipm,
            "opm_window": opm
        }
        
        with self.write_lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

    def get_current_metrics(self):
        now = time.time()
        with self.history_lock:
            while self.history and self.history[0]["timestamp"] < now - 60:
                self.history.popleft()
            
            return {
                "tpm": sum(entry["tokens"] for entry in self.history),
                "rpm": len(self.history),
                "ipm": sum(entry["input"] for entry in self.history),
                "opm": sum(entry["output"] for entry in self.history)
            }

# Global instance
token_logger = TokenUsageLogger()
