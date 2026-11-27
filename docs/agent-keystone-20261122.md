# 在 Pledge Keystone 里实测：GPT-6.1 通过 MCP 操作桌面窗口

目的：确认 `motionspec agent-mcp` 能否作为 MCP 服务接入 Pledge Keystone，让 OpenAI 模型真正完成一个桌面操作任务，并找出接入中的问题。

## 怎么测的

Keystone 取 `feature/ui-redesign` 分支（95da1dc），组件原样调用：

| 环节 | 用的是 |
|---|---|
| 对话循环 | 桌面端 Work 模式的 `qt_frontend.conversation.run`，回合上限按本机配置读出（12） |
| MCP 客户端 | 桌面端的 `qt_frontend.mcp_http.McpHttp`（Streamable HTTP） |
| 模型 | `run.py gateway`，上游 `api.openai.com`；GPT-6.1 带工具的请求由网关自动转成 Responses 接口 |
| 屏幕工具 | `motionspec agent-mcp --http 127.0.0.1:端口 --no-images --allow-window "Keystone agent task"` |

只有 `McpCatalog` 把发现的工具包成对话工具的那 20 来行（`mcp_名字_哈希` 的命名、64 KiB 参数和 128 KiB 结果上限、回执格式）是照抄的，因为它依赖 Qt 事件循环。权限确认自动通过，相当于用户逐项点了"允许"。脚本见 [agent_keystone_live.py](../packages/motionspec/bench/agent_keystone_live.py)。

任务窗口是一个 Qt 写的"同步设置"小应用（[agent_task_app.py](../packages/motionspec/bench/agent_task_app.py)），置顶、不抢焦点：服务器名称和端口输入框，两个复选框，四个只有图标的工具栏按钮（Refresh、Save、Delete、Help），改动后才可用的 Apply，以及 Start sync。开始同步会弹出确认框，确认后依次显示连接、进度、完成，约 2.2 秒。Delete 会弹出删除确认框，确认即算失败。任务是否完成由应用自己写出的状态文件判定，不看模型怎么说。

给模型的任务（原文）：

> 请在「Keystone agent task」窗口里完成：把服务器名称改成 archive-02，打开 Encrypt transfers，保存设置，然后开始同步。等同步完成后，告诉我同步了多少个文件、同步到了哪台服务器、是否加密。不要删除任何数据。

成功条件：同步到 archive-02、加密、完成、没有删除。

## 结果

| 轮次 | 模型 | 结果 | 用时 | 回合 | 工具调用 | token |
|---|---|---|---:|---:|---:|---:|
| 第 1 轮：操作后要另调 wait_for_settle | sol | 未完成，用完 12 次调用，停在确认框前 | 59 s | 13 | 12 | 58,942 |
| 第 2 轮：操作自己等待并报告效果 | sol | 完成 | 40 s | 9 | 8 | 35,576 |
| | terra | 完成 | 56 s | 10 | 11 | 55,859 |
| 第 3 轮：再附上控件状态变化、精简控件列表 | sol | 完成 | 38 s | 9 | 8 | 24,905 |
| | terra | 完成 | 50 s | 9 | 9 | 37,606 |
| | sol | 完成 | 39 s | 8 | 7 | 21,076 |
| | terra | 中断：OpenAI 上游断开，网关返回 502 | 24 s | 5 | 6 | 15,369 |
| | terra（重跑） | 完成 | 43 s | 8 | 8 | 30,554 |

所有完成的运行，回答里的文件数（48）、服务器和加密状态都正确，都没有点 Delete，也没有一次工具报错。每次运行时间大部分花在模型上：第 3 轮 sol 模型合计约 32 s，工具合计 5～6 s；单次工具调用多在 0.1～1.4 s。

sol 第 3 轮的最短路径：列控件 → 填服务器名称 → 勾选加密 → 点 Save → 点 Start sync（结果里带回确认框和它的按钮）→ 在确认框点 Start → 等同步完成并读出状态 → 回答。terra 习惯多读一次屏幕文字再核对。两个模型都选择了工具栏上的 Save 图标而不是 Apply，在这个应用里效果相同。

## 两轮改进来自哪里

第 1 轮里模型每次操作后都调一次 `wait_for_settle` 看效果，开头还为留基准帧先调一次，12 次调用里 5 次花在这上面，任务没做完。第 2 轮起，`click_element` 和 `set_field_text` 自己取基准帧、执行、等界面稳定，直接返回变化区域里的文字和新弹出的窗口（含它的按钮），调用数从 12 降到 8。

第 2 轮里两个模型都在勾选复选框后又列了一遍控件，为了看 Apply 能不能点了；控件列表每次约 4.5 KB，之后每回合都要带着。第 3 轮的操作结果附上了 `controls_changed`（哪些控件变成可用、勾选、改了值、出现或消失），控件列表也只留名字、类型、是否可用、值或勾选状态和点击点，约 2.4 KB。sol 的 token 从约 3.6 万降到 2.1～2.5 万，terra 从约 5.6 万降到 3.0～3.8 万。

## 第 4、5 轮：取消回合上限，加三类任务

脚本加了 `--max-turns`（默认 100，相当于不设上限），本机 Keystone 配置里的 12 次上限没有改动。新增三个任务，每个任务的完成情况都由应用或页面自己的状态判定：

- recovery：同一个设置窗口，第一次同步会因端口错误失败，弹出错误框说明服务器只接受 8443 端口，需要关掉错误框、改端口、保存、再同步；
- web：Playwright 单独启动的 Chrome 窗口（不用本机浏览器配置），填注册表单：两个输入框、下拉框选 Pro、点两次只有图标的"Add seat"（aria-label）、勾选条款、提交；
- canvas：按钮全部由 paintEvent 画出的 Qt 窗口，UI Automation 只看得到标题栏，要用屏幕文字找位置，再用新加的 `click_point` 按坐标点击。

`click_point` 通过向窗口发送鼠标消息点击，不移动指针、不需要焦点；只能点允许的窗口内部，点下的位置如果被别的应用窗口挡住就拒绝。

| 任务 | 模型 | 第 4 轮 | 第 5 轮（优化后） |
|---|---|---|---|
| settings | sol | 成功，36 s，8 次调用，26,079 token | 成功，40 s，8 次，26,395 |
| | terra | 成功，40 s，9 次，39,563 | 成功，36 s，9 次，37,455 |
| recovery | sol | 成功，60 s，13 次，61,739 | 成功，58 s，13 次，60,118 |
| | terra | 成功，62 s，15 次，88,025 | 成功，65 s，15 次，85,844 |
| web | sol | 成功，51 s，9 次，30,381；工具合计 18.6 s | 成功，43 s，9 次，30,542；工具合计 8.8 s |
| | terra | 成功，50 s，10 次，46,379；工具合计 19.6 s | 成功，40 s，9 次，31,057；工具合计 8.1 s |
| canvas | sol | 成功，21 s，5 次，12,998 | 成功，20 s，5 次，12,295 |
| | terra | 成功，15 s，5 次，11,550 | 成功，20 s，5 次，12,324 |

两轮 16 次全部完成，回答都正确，没有一次工具报错，也没有一次碰 Delete。recovery 需要 13～15 次工具调用，按当时 Keystone 默认的 12 次上限做不完；Keystone 之后改为默认不限步数（可在"模型与配置"的高级区域设置上限）。两个模型在 recovery 里都读懂了错误框（新弹出的窗口和其中的文字直接出现在点击结果里），一次改对端口。canvas 里两个模型都是"列控件发现没有名字 → 读屏幕文字 → 三次坐标点击"，没有点空。

### 第 4 轮暴露的问题：杀毒软件拦截了浏览器截屏

web 任务里每次操作要 1～2.3 秒，比 Qt 窗口慢一倍多，等待结果总是"没有变化"。查下来，截到的 Chrome 窗口不是网页，而是 ESET Internet Security 的提示图："为保护您的隐私，在浏览器屏幕保护处于活动状态期间，不得截图、录制或分享您的浏览器屏幕"。第一次操作时提示图出现，看起来像一次大变化，之后截到的永远是同一张图。关掉 GPU、关掉 DirectComposition、换 Pillow 截屏、关掉 Chrome 遮挡检测都一样。所以在这台电脑上，浏览器窗口的截屏、等待判定和 OCR 都不可用；terra 在第 4 轮开头读到的"屏幕文字"其实是这张提示图。任务仍然完成，是因为操作和结果都来自 UI Automation。

这是本机安全软件的设置，没有改动。服务这边的处理：

- 一次操作后像素没变、但 UI Automation 看到控件变了，就把这个窗口记为"截屏不可靠"。之后对它的操作不再等像素，改为每 120 ms 读一次控件，连续 250 ms 不变即算稳定；结果里注明效果来自 UI Automation。对这个窗口的 `read_screen_text`、`find_on_screen`（OCR 部分）和 `list_ui_elements` 结果附上同样的提醒。
- 唤醒 Chrome 页面无障碍树时，原来等"元素数量变多"，页面已经就绪时会白等满 3 秒；改为看有没有 Document 节点，已有就直接返回。第一次列控件从约 3.5 s 降到 0.15～0.2 s。

第 5 轮 web 任务的工具合计时间从约 19 s 降到约 8.5 s，terra 也不再去读那张提示图，token 从 46,379 降到 31,057。

其他改动：返回的控件列表和弹窗内容去掉了标题栏的最小化、最大化、关闭和系统菜单；点了复选框之类的小控件后，同一行的文字标签会和变化一起报告（第 4 轮里点"Notify team"方框时没有带回标签）。

## 接入中发现的 Keystone 侧问题

这些都在 Keystone 里，这次没有改动它，只记录下来：

1. Keystone 自带的电脑操作驱动（`computer_win32.WindowsControlDriver`）靠枚举子窗口观察，在这个 Qt 窗口里看到 0 个元素；UI Automation 看到 20 个，包括 4 个只有图标的按钮。Qt、Chrome、Electron 这类应用的控件都不是独立子窗口，原生驱动只能退回到截图坐标点击。
2. MCP 工具结果整个序列化成文本交给模型，图片内容会变成 base64 字符串，超过 128 KiB 就被拒绝。所以接 Keystone 时要用 `--no-images` 去掉截图工具。
3. MCP HTTP 请求的超时是 10 秒。服务在 HTTP 模式下默认把每次等待限制在 8 秒内。
4. 本机 HTTP 的 MCP 服务只在 Work 模式里提供给模型；Chat 模式只接受公网 HTTPS 服务。
5. 上游一次断连（502）就让整个对话循环停止，没有重试；在桌面端需要用户点"继续"。
6. 每次调用工具前，客户端都会先请求一遍工具清单，多一次往返（本机很快，影响不大）。
7. GPT-6.1 在 Chat Completions 接口上只有关闭推理才能用工具。Keystone 网关会对官方地址自动改走 Responses，这一点工作正常；如果换成别的代理地址，要确认它也支持 Responses。

## 在桌面端接入

```bash
python -m pip install -e ".[overlays,uia]" -e "./packages/motionspec[capture]"
motionspec agent-mcp --http 127.0.0.1:8765 --no-images --allow-window "要操作的窗口标题"
```

在 Keystone 的连接器页添加自定义 MCP 接口，地址填 `http://127.0.0.1:8765/mcp`。需要令牌时，启动前设置环境变量 `VISFUNC_MCP_TOKEN`，并在连接器里填同一个值。操作类工具只对 `--allow-window` 列出的窗口生效，判断依据是实际找到的窗口标题；不加这个参数时只提供观察类工具。

## 还没做的

- 桌面端连接器页的完整点击流程（添加接口、在 Work 模式里对话）没有在界面上走一遍，这次是用同样的组件在脚本里调用的。
- 每个任务每个模型只跑了两到三次，次数不足以估计成功率。
- `click_point` 发送的是窗口消息；直接读取硬件鼠标状态的程序（部分游戏、少数自绘控件）不会响应，这时需要 Keystone 自己带确认的真实输入。
- 浏览器窗口在本机被杀毒软件挡住截屏，网页里只有图标、又没有 aria-label 的按钮，在这台电脑上无法通过截屏找到。

## 复现

```bash
python packages/motionspec/bench/agent_keystone_live.py <Keystone 仓库> <装了 PySide6 的 python> out.json gpt-6.1-sol gpt-6.1-terra
```

脚本从 `~/.forge/secrets.json` 读 OpenAI 密钥，只通过环境变量交给网关子进程；网关使用临时 home。运行时桌面上会出现任务窗口。原始记录见第 1～3 轮的 [结果文件](../bench/results/20261122-agent-keystone-live/agent-keystone-live-20261122.json) 和第 4、5 轮的 [结果文件](../bench/results/20261122-agent-keystone-rounds45/agent-keystone-rounds45-20261122.json)。第 4、5 轮用 `--tasks settings recovery web canvas` 运行。
