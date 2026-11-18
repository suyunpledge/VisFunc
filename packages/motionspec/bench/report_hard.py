"""Report item-weighted results; pair only uniquely identified repeated tasks."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from report import aggregate, fraction
from run_hard import CONDITIONS, FIELDS


def paired_rows(rows, conditions):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.get("run_id") or type(row.get("repeat")) is not int:
            continue
        key = (row["run_id"], row["model"], row["scene"], row["repeat"], row.get("provider_slot"))
        groups[key][row["condition"]].append(row)
    paired = []
    for group in groups.values():
        if all(len(group[c]) == 1 and group[c][0].get("ok")
               and group[c][0].get("delivery", {}).get("accepted") for c in conditions):
            paired.extend(group[c][0] for c in conditions)
    return paired


def metrics(rows):
    adapted = []
    for row in rows:
        copy = {**row, "score": row["score"]["fields"]}
        if "request_seconds" not in row and row.get("attempts"):
            copy["request_seconds"] = sum(attempt["seconds"] for attempt in row["attempts"])
        if "usage_unknown_attempts" not in row:
            copy["usage_unknown_attempts"] = max(1, len(row.get("attempts", []))) - int(
                type((row.get("usage") or {}).get("total_tokens")) is int
            )
        if "end_to_end_score" in row:
            copy["end_to_end_score"] = row["end_to_end_score"]["fields"]
        adapted.append(copy)
    return aggregate(adapted)


def render(rows, conditions=None):
    conditions = conditions or [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    models = list(dict.fromkeys(r["model"] for r in rows))
    all_metrics = metrics(rows)
    paired = paired_rows(rows, conditions)
    paired_metrics = metrics(paired)
    lines = ["# 第二批场景模型实测", "",
             f"{len(rows)} 个记录，{len({r['scene'] for r in rows})} 个场景。",
             "字段合计按 sum(correct)/sum(total) 加权；数量判断与多报另列。",
             "字段诊断允许不完整对象；端到端得分要求交付 schema 通过。", "",
             "| 模型 | 条件 | 字段诊断 | 端到端字段 | 严格格式通过 | 交付通过 | 可读回答中有错 |",
             "|---|---|---|---|---|---|---|"]
    for model in [*models, "ALL"]:
        for condition in conditions:
            g = all_metrics[model, condition]
            lines.append(f"| {model} | {condition} | " + " | ".join(fraction(g[k]) for k in
                         ("score", "delivery_score", "strict", "accepted", "wrong")) + " |")
    lines += ["", "## 配对比较", "",
              "只纳入同一 run_id、模型、场景、repeat、提供方的完整条件组；重复记录不参与配对。",
              "缺少轮次标识的历史记录无法可靠配对，不按出现顺序猜测。配对仅描述所有条件交付成功的子集。",
              f"纳入 {len(paired)} 条记录；条件集合：{', '.join(conditions)}。", "",
              "| 模型 | 条件 | 配对字段正确 |", "|---|---|---|"]
    for model in [*models, "ALL"]:
        for condition in conditions:
            lines.append(f"| {model} | {condition} | {fraction(paired_metrics[model, condition]['score'])} |")
    lines += ["", "## 数量、错误和成本", "",
              "tokens 与请求耗时累计所有 attempt；未知用量不按零成本处理。传输失败与格式失败分开统计。",
              "旧记录缺少的元数据为未知；独立补跑目录的成本另计，不能把原失败调用当作免费。", "",
              "| 条件 | 数量正确 | 多报 | 传输失败 | attempts | 已知 tokens | 用量未知 | 请求总秒 |",
              "|---|---|---|---|---|---|---|---|"]
    for condition in conditions:
        selected = [r for r in rows if r["condition"] == condition]
        count = [sum(bool(r["score"]["count_ok"]) for r in selected), len(selected)]
        extra = sum(r["score"]["extra"] for r in selected)
        failed = sum(not r.get("ok", False) for r in selected)
        g = all_metrics["ALL", condition]
        lines.append(f"| {condition} | {fraction(count)} | {extra} | {failed} | {g['attempts']} | "
                     f"{g['known_tokens']} | {g['unknown_usage']} | {g['request_seconds']:.1f} |")
    lines += ["", "## 按测量项", "", "| 字段 | 条件 | 正确 |", "|---|---|---|"]
    for field in FIELDS:
        for condition in conditions:
            pairs = [r["score"]["fields"][field] for r in rows if r["condition"] == condition]
            lines.append(f"| {field} | {condition} | {fraction([sum(p[i] for p in pairs) for i in (0, 1)])} |")
    lines += ["", "## 按场景", "", "| 场景 | 条件 | 字段诊断 |", "|---|---|---|"]
    for scene in dict.fromkeys(row["scene"] for row in rows):
        for condition in conditions:
            pairs = [pair for row in rows if row["scene"] == scene and row["condition"] == condition
                     for pair in row["score"]["fields"].values()]
            lines.append(f"| {scene} | {condition} | {fraction([sum(pair[i] for pair in pairs) for i in (0, 1)])} |")
    return "\n".join(lines) + "\n"


def main():
    folder = Path(sys.argv[1])
    rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    protocol = folder / "protocol.json"
    conditions = json.loads(protocol.read_text(encoding="utf-8")).get("conditions") if protocol.exists() else None
    text = render(rows, conditions)
    (folder / "report.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    main()
