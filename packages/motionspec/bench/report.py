"""Item-weighted diagnostic and delivery metrics, with explicit denominators."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def fraction(pair):
    return f"{pair[0]}/{pair[1]} ({100 * pair[0] / pair[1]:.1f}%)" if pair[1] else "n/a"


def aggregate(rows):
    groups = defaultdict(
        lambda: {
            "score": [0, 0],
            "delivery_score": [0, 0],
            "strict": [0, 0],
            "accepted": [0, 0],
            "wrong": [0, 0],
            "attempts": 0,
            "known_tokens": 0,
            "unknown_usage": 0,
            "request_seconds": 0.0,
            "wall_seconds": 0.0,
        }
    )
    for row in rows:
        for label in (row["model"], "ALL"):
            group = groups[label, row["condition"]]
            for name, source in (
                ("score", row["score"]),
                ("delivery_score", row.get("end_to_end_score")),
            ):
                if source is not None:
                    for correct, total in source.values():
                        group[name][0] += correct
                        group[name][1] += total
            delivery = row.get("delivery")
            if delivery is not None:
                for name, field in (("strict", "strict_format_pass"), ("accepted", "accepted")):
                    if name == "strict" and not row.get("ok", True):
                        continue
                    group[name][0] += int(delivery[field])
                    group[name][1] += 1
            if row.get("answer_wrong") is not None:
                group["wrong"][0] += int(row["answer_wrong"])
                group["wrong"][1] += 1
            attempts = row.get("attempts")
            group["attempts"] += len(attempts) if attempts is not None else 1
            group["unknown_usage"] += row.get("usage_unknown_attempts", 1)
            group["known_tokens"] += row.get("usage", {}).get("total_tokens", 0)
            group["request_seconds"] += row.get("request_seconds", row["seconds"])
            group["wall_seconds"] += row["seconds"]
    return groups


def render(rows):
    groups = aggregate(rows)
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    models = list(dict.fromkeys(row["model"] for row in rows))
    scenes = {row["scene"] for row in rows}
    lines = [
        "# MotionSpec 评测",
        "",
        f"{len(rows)} 个任务，{len(scenes)} 个独立场景。"
        "分母为每个动效的六个字段，加每段一次数量判断。",
        "合计为逐项加权 sum(correct)/sum(total)，不是模型百分比的简单平均。",
        "同一视频内字段相关；这些比率不是独立伯努利样本，不据此计算窄置信区间。",
        "",
        "字段诊断允许读取不完整对象的已有字段；端到端指标要求整条响应通过交付 schema。",
        "严格格式通过率把裸数组/Markdown 外壳记为不合规，交付通过率允许这两种显式外壳归一化。",
        "严格格式通过率只统计收到响应的任务；交付通过率包括传输失败的任务。",
        "无法提取对象的响应没有可判断的 answer_wrong，单独体现在格式和交付失败中。",
        "旧记录缺失的交付/重试元数据标为 n/a 或未知，不追溯虚构用量。",
        "",
        "| 模型 | 条件 | 字段正确 | 端到端字段正确 | 严格格式通过 | 交付通过 | 可读回答中有错 |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in [*models, "ALL"]:
        for condition in conditions:
            group = groups[model, condition]
            lines.append(
                f"| {model} | {condition} | "
                + " | ".join(
                    fraction(group[key])
                    for key in ("score", "delivery_score", "strict", "accepted", "wrong")
                )
                + " |"
            )
    lines += [
        "",
        "用量按所有 attempt 累加；未知用量不按免费或零 token 处理。"
        "耗时为任务累计，不能当并行运行总墙钟时间。",
        "",
        "| 条件 | attempts | 已知 total tokens | 用量未知 attempts/旧任务 |"
        " 请求耗时总秒 | 含等待任务耗时总秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        g = groups["ALL", condition]
        lines.append(
            f"| {condition} | {g['attempts']} | {g['known_tokens']} | {g['unknown_usage']} | "
            f"{g['request_seconds']:.1f} | {g['wall_seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    folder = Path(sys.argv[1])
    rows = [
        json.loads(line)
        for line in (folder / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = render(rows)
    (folder / "report.md").write_text(output, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(output)


if __name__ == "__main__":
    main()
