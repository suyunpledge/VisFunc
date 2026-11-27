# 四层图片处理模块调用核查

审查基于 `3bdea85` 的运行时代码。沿 HTTP 两个分析入口、命令行、图片函数编译、视频候选生成、示例和测试追踪模块导入及实际调用，排除了归档备份。四个包都进入实际运行路径。

| 模块 | 到达它的入口与实际工作 |
| --- | --- |
| `svsg/l1_compiler` | 图片上传经 `svsg/api.py` 选择检测器并编译。图片函数编译器使用 `ImageMeta`，视频候选生成器也使用检测适配器。L1 编译器调用几何引擎和不确定度估计。 |
| `svsg/l15_verifier` | API 按配置注入 `L15Verifier` 和 OpenAI 视觉适配器。L3 在装配时校验超时预算，运行中按需要调用验证服务。 |
| `svsg/l2_runtime` | L3 每次编排都建立状态机、校验 IR 与依赖，交付前调用证据锚定，在复核分歧时调用冲突解决器。意图路由见下。 |
| `svsg/l3_orchestrator` | HTTP 分析入口调用 `Orchestrator`，由其组装提示词、调用已配置的模型适配器、生成证据约束后的输出。示例也调用渲染模块。 |

待确认清单中有一个模块。`l2_runtime/intent_router.py` 的 `route_intent` 由包导出，仓内调用只见于该模块的便捷函数和 `tests/test_l2_runtime.py`。另有一个函数 `l15_verifier/fallback.py::silent_degrade` 只有导出和测试调用，但同文件的超时校验在 L3 装配时实际使用。此次保留两者，外部使用方不在静态扫描范围内。
