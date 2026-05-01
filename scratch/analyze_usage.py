
import json
from pathlib import Path

def analyze_logs(log_path):
    input_tokens = 0
    output_tokens = 0
    requests = 0
    success_requests = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                requests += 1
                if data['status'] == 'success':
                    success_requests += 1
                    input_tokens += data['input_tokens']
                    output_tokens += data['output_tokens']
            except:
                continue
                
    # o3-mini pricing: $1.10 / 1M input, $4.40 / 1M output
    cost = (input_tokens / 1_000_000) * 1.10 + (output_tokens / 1_000_000) * 4.40
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "requests": requests,
        "success_requests": success_requests,
        "cost": cost,
        "avg_input_per_build": input_tokens / 5,
        "avg_output_per_build": output_tokens / 5,
        "avg_cost_per_build": cost / 5
    }

if __name__ == "__main__":
    result = analyze_logs("logs/token_usage.log")
    print(json.dumps(result, indent=2))
