import json
import re
from datetime import datetime

input_file = "agentcloud/data/raw/hdfs.log"
output_file = "agentcloud/data/train.jsonl"


def classify_event(message: str):
    msg = message.lower()

    # crash
    if any(x in msg for x in [
        "segfault",
        "core dump",
        "crash",
        "fatal",
        "terminating",
        "shutdown",
        "dead",
        "exception"
    ]):
        return "crash"

    # intrusion
    if any(x in msg for x in [
        "failed password",
        "authentication failure",
        "brute force",
        "port scan",
        "exploit",
        "malicious",
        "unauthorized"
    ]):
        return "intrusion"

    # overload
    if any(x in msg for x in [
        "high cpu",
        "memory pressure",
        "oom",
        "throttle",
        "maxrequestworkers",
        "swap usage"
    ]):
        return "overload"

    return "normal"


def extract_level(line: str):
    match = re.search(r"\[(INFO|WARN|ERROR|CRITICAL|DEBUG)\]", line)
    if match:
        return match.group(1)
    return "INFO"


def extract_service(message: str):
    services = [
        "HDFS",
        "kubelet",
        "kernel",
        "sshd",
        "nginx",
        "systemd",
        "Apache",
        "CloudWatch",
        "Prometheus"
    ]

    for s in services:
        if s.lower() in message.lower():
            return s

    return "unknown"


with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:

        line = line.strip()

        if not line:
            continue

        level = extract_level(line)

        # remove timestamp prefix
        message = re.sub(r"^\[.*?\]\s*", "", line)

        label = classify_event(message)

        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": extract_service(message),
            "level": level,
            "message": message,
            "label": label
        }

        fout.write(json.dumps(log) + "\n")

print("Dataset conversion complete.")