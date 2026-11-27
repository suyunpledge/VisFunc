# 桌面 Agent 动作回执与独立会话复测

在 Windows 交互桌面上分别启动三个 Qt 测试窗口、桌面 MCP 服务和全新的本机 Keystone 会话。模型为 GPT-6.1 Sol，每个会话只可操作本次窗口。以窗口实际记录的计数、左右字段以及最终状态为准。下表统计该会话全部工具调用。

| 场景 | 注入或设置 | 最终状态与调用行为 | 调用数 | 耗时 |
|---|---|---|---:|---:|
| 点击后的 OCR 异常 | 仅使动作完成后的 OCR 抛出异常 | Add one 执行 1 次；回执区分已执行和观察失败；Agent 用只读控件树核对 Count 1 | 2 | 13.5 s |
| 同名 Seats 输入框 | 东西两栏同名，初始值均为 2 | Agent 按控件引用定位东栏并保存一次，西栏仍为 2、东栏为 7 | 4 | 18.7 s |
| 显式等待 | 保持像素帧冻结，应用延迟更新控件树 | 点击 1 次后调用 wait_for_settle，返回 ui_automation，读到 Batch verified 731 | 2 | 11.5 s |

三个场景各运行一次，用于验证故障路径能否走通。重复操作发生率、跨应用兼容性和识别准确率需要另行采样。原始证据包括工具调用顺序、MCP 返回值和 Qt 窗口状态。仓库回归用例位于 `packages/motionspec/tests/test_agent.py` 和 `tests/test_ui_elements.py`，涵盖真实动作后的观察异常、同名项与陈旧引用，以及 UIA 等待参数、窗口丢失和同名状态差异。运行 `python -m pytest`，本轮 492 项通过，1 条依赖弃用警告。

复测提示明确要求只点一次、通过名称工具操作右侧字段（必要时提供引用）、以及显式调用等待并开启读文本。每个会话只能调用当前窗口的 MCP 工具。表中结果衡量受控条件下的执行与复核能力。[各会话的工具调用、回执及窗口终态](../bench/results/20261122-agent-receipts-keystone/agent-receipts-keystone-20261122.json)

## Windows 便携包验收

解压后又开了第四个独立 Keystone 会话，使用包内运行时复测点击后的 OCR 异常。Agent 先读取控件，点击一次，再读取控件确认 `Count 1`。窗口记录的点击次数为 1，整个会话耗时 12.5 秒。[成品校验值、工具调用顺序和故障回执](../bench/results/20261122-agent-release-verification/agent-release-verification-20261122.json)

在同一台 Windows x64 机器将 ZIP 解压到另一个含空格和中文的目录，移除测试进程继承的 Python 搜索路径，并将 PATH 限定为包内工具和 Windows 系统目录。启动器连续两次通过 API 健康检查和带 token 的 MCP 初始化与工具发现，第二次保留已有配置。通过只读 MCP 从实际 Qt 窗口枚举到两个同名控件的不同引用。HTTP OCR 在 1920 × 1080 设置页上读出 18 行。包内 FFmpeg 生成的 1 秒样片经 CLI 导出 spec.json、summary.md 和 functions.json；YOLO11n 和 CrowdHuman 均执行了一次本地 CPU 推理。模型自检使用空白输入，只验证模型能加载并完成推理。
