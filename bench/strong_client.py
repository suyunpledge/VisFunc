"""Meter paid calls before dispatch and preserve each tool turn's provider usage."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from motion_corpus import TOOLS

sys.path.insert(0, str(TOOLS.parent / "bench"))
import client
from providers import find

MODELS = {"gpt": "gpt-6.1-sol", "mimo": "mimo-v2.6-flash", "seed": "doubao-seed-2-1-pro-260915"}
SLOTS = {"gpt": "gpt-6.1-sol", "mimo": "mimo-v2.6-flash", "seed": "seed-2.1-pro"}


def metered_usd(model, usage):
    if model != "gpt":
        return 0.0  # Subscription quota; record tokens separately.
    details = usage.get("input_tokens_details", {})
    read = details.get("cached_tokens", 0)
    written = details.get("cache_write_tokens", 0)
    long_write = details.get("cache_write_tokens_details", {}).get("ttl_12h_tokens", 0)
    ordinary = usage.get("input_tokens", 0) - read - written
    amount = (
        2 * ordinary
        + 0.1 * read
        + 2.5 * (written - long_write)
        + 5 * long_write
        + 10 * usage.get("output_tokens", 0)
    )
    return amount / 1e6


class Ledger:
    def __init__(self, path, limit):
        self.path, self.limit = path, limit
        self.lock = threading.Lock()
        with self.database() as db:
            db.execute(
                "create table if not exists charge("
                "id text primary key, task text, usd real, done integer)"
            )

    def database(self):
        return sqlite3.connect(self.path, timeout=30)

    def reserve(self, task, worst):
        with self.lock, self.database() as db:
            db.execute("begin immediate")
            total = db.execute("select coalesce(sum(usd),0) from charge").fetchone()[0]
            if total + worst > self.limit:
                raise RuntimeError("budget_reservation_unavailable")
            key = uuid4().hex
            db.execute("insert into charge values(?,?,?,0)", (key, task, worst))
            return key

    def settle(self, key, amount):
        with self.database() as db:
            db.execute("update charge set usd=?, done=1 where id=?", (amount, key))


class PaidModels:
    def __init__(self, slots, ledger, models=None):
        config = json.loads(Path(slots).read_text(encoding="utf8"))
        self.slots = {key: find(config[SLOTS[key]]) for key in models or MODELS}
        self.ledger = ledger
        for key, provider in self.slots.items():
            url = urlsplit(provider.base_url)
            if url.scheme != "https" or not (
                (key == "gpt" and url.hostname == "api.openai.com")
                or (
                    key == "mimo"
                    and url.hostname
                    in ("token-plan-cn.xiaomimimo.com", "token-plan.xiaomimimo.com")
                )
                or (
                    key == "seed"
                    and url.hostname == "ark.cn-beijing.volces.com"
                    and url.path == "/api/plan/v3"
                    and provider.label == "豆包多模态测试"
                )
            ):
                raise ValueError(f"{key}: pricing/plan endpoint does not match the approved budget")

    def call(self, model, messages, task, max_output=3000):
        provider = self.slots[model]
        if model == "gpt":
            url = "https://api.openai.com/v1/responses"
            wire = []
            for message in messages:
                content = message["content"]
                if isinstance(content, list):
                    content = [
                        {"type": "input_text", "text": p["text"]}
                        if p["type"] == "text"
                        else {"type": "input_image", "image_url": p["image_url"]["url"]}
                        for p in content
                    ]
                wire.append({"role": message["role"], "content": content})
            body = {
                "model": MODELS[model],
                "input": wire,
                "max_output_tokens": max_output,
                "reasoning": {"effort": "low"},
                "service_tier": "default",
                "store": False,
            }
        else:
            url = provider.base_url + "/chat/completions"
            body = {"model": MODELS[model], "messages": messages, "max_tokens": max_output}
            if model == "seed":
                body["thinking"] = {"type": "disabled"}
        # Price the exact wire input before generation. A failed preflight sends no
        # generation request. Leave headroom for metadata and cache-write pricing.
        counted = None
        if model == "gpt":
            status, data, seconds = client.post(
                url + "/input_tokens",
                provider.key,
                {"model": MODELS[model], "input": wire},
                timeout=45,
            )
            counted = data.get("input_tokens") if isinstance(data, dict) else None
            if status != 200 or type(counted) is not int or not 0 < counted < 260000:
                return {
                    "status": status if status != 200 else 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error_code": "count_preflight",
                    "budget_usd": 0,
                    "usage_known": True,
                    "seconds": round(seconds, 3),
                    "answer": "",
                }
        worst = ((counted + 256) * 5 + max_output * 10) / 1e6 if counted is not None else 0.0
        charge = self.ledger.reserve(task, worst)
        status, payload, seconds = client.post(url, provider.key, body, timeout=140)
        data = payload if isinstance(payload, dict) else {}
        usage = data.get("usage") or {}
        inputs = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        outputs = usage.get("output_tokens", usage.get("completion_tokens", 0))
        usd = metered_usd(model, usage)
        # Unknown billing after a lost response retains the full reservation.
        known = bool(usage) or 400 <= status < 500
        charge_usd = usd if known else worst
        self.ledger.settle(charge, charge_usd)
        if model == "gpt":
            answer = "".join(
                c.get("text", "")
                for item in data.get("output", [])
                for c in item.get("content", [])
                if c.get("type") == "output_text"
            )
            finish = data.get("status")
            incomplete = data.get("incomplete_details") or {}
            finish = incomplete.get("reason") or finish
        else:
            choice = (data.get("choices") or [{}])[0]
            answer = (choice.get("message") or {}).get("content") or ""
            finish = choice.get("finish_reason")
        record = {
            "status": status,
            "served_model": data.get("model"),
            "response_id": data.get("id"),
            "provider_endpoint": url,
            "finish_reason": finish,
            "counted_input": counted,
            "input_tokens": inputs,
            "output_tokens": outputs,
            "usage": usage,
            "seconds": round(seconds, 3),
            "budget_usd": round(charge_usd, 7),
            "usage_known": known,
            "answer": answer,
        }
        if status != 200:
            # Store the provider error code only; omit remote echoes of request bodies.
            try:
                error = (json.loads(payload) if isinstance(payload, str) else data).get("error", {})
                record["error_code"] = str(error.get("code", ""))[:90]
            except (ValueError, AttributeError):
                record["error_code"] = "transport" if not status else "request_failed"
        return record
