"""Reserve a desktop operation before dispatch. Only its digest and status go to disk."""

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from uuid import UUID

RETENTION_SECONDS = 86400
MAX_RECEIPTS = 50000


def receipt_path():
    setting = os.environ.get("VISFUNC_ACTION_DB")
    if setting:
        return setting
    root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "state")
    return str(root / "VisFunc" / "action-receipts.sqlite3")


def operation_key(scope, request_id, operation_id):
    if operation_id is not None:
        try:
            key = UUID(operation_id)
        except (TypeError, ValueError, AttributeError):
            raise ValueError("operation ID must be a UUID") from None
        return "operation:" + key.hex
    if not scope:
        raise ValueError(
            "initialize an MCP session or supply an Idempotency-Key UUID before acting"
        )
    if type(request_id) not in (str, int):
        raise ValueError("action request ID must be a string or integer")
    identity = json.dumps([scope, request_id], separators=(",", ":")).encode()
    return "rpc:" + hashlib.sha256(identity).hexdigest()


class ActionReceipts:
    def __init__(self, path=":memory:"):
        self.path = path
        self.lock = threading.Lock()
        self.db = None
        self.counts = {"reserved": 0, "replays": 0, "conflicts": 0}

    def _connection(self):
        if self.db is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, check_same_thread=False)
            try:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS receipts "
                    "(key TEXT PRIMARY KEY, fingerprint TEXT, expires INTEGER, receipt TEXT)"
                )
                db.execute("CREATE INDEX IF NOT EXISTS expiry ON receipts(expires)")
            except BaseException:
                db.close()
                raise
            self.db = db
        return self.db

    def reserve(self, key, name, args):
        canonical = json.dumps([name, args], sort_keys=True, separators=(",", ":"), allow_nan=False)
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        with self.lock:
            db = self._connection()
            with db:
                db.execute("BEGIN IMMEDIATE")
                now = int(time.time())
                db.execute("DELETE FROM receipts WHERE expires <= ?", (now,))
                saved = db.execute(
                    "SELECT fingerprint, receipt FROM receipts WHERE key=?", (key,)
                ).fetchone()
                if saved:
                    if saved[0] != fingerprint:
                        self.counts["conflicts"] += 1
                        raise ValueError("operation ID already used with different arguments")
                    receipt = (
                        json.loads(saved[1])
                        if saved[1]
                        else {
                            "execution_state": "unknown",
                            "next_step": "Inspect current state. Do not repeat the action.",
                        }
                    )
                    self.counts["replays"] += 1
                    return {**receipt, "replayed": True}
                if db.execute("SELECT count(*) FROM receipts").fetchone()[0] >= MAX_RECEIPTS:
                    raise ValueError("action receipt store full; action not dispatched")
                db.execute(
                    "INSERT INTO receipts VALUES (?, ?, ?, NULL)",
                    (key, fingerprint, now + RETENTION_SECONDS),
                )
            self.counts["reserved"] += 1
        return None

    def stats(self):
        with self.lock:
            return dict(self.counts)

    def finish(self, key, receipt):
        with self.lock:
            db = self._connection()
            with db:
                db.execute("UPDATE receipts SET receipt=? WHERE key=?", (json.dumps(receipt), key))

    def close(self):
        with self.lock:
            if self.db is not None:
                self.db.close()
                self.db = None
