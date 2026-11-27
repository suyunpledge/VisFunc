"""Run an individual measurement inside the disposable, networkless container."""

import base64
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

job = json.load(sys.stdin)
Path("clip.mp4").write_bytes(base64.b64decode(job["video"], validate=True))
Path("measure.py").write_text(job["code"], encoding="utf-8")
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
with open("log.txt", "w+b") as log:
    child = subprocess.Popen(
        [sys.executable, "-I", "measure.py"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        status = child.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, 9)
        child.wait()
        status = 124
    log.seek(0)
    output = log.read(4096).decode("utf8", "replace")
print(json.dumps({"status": status, "output": output}))
