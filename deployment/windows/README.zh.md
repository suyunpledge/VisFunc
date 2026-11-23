# VisFunc · Windows x64 一键部署

有两种成品包：

| 包 | 内容 | 适合 |
|---|---|---|
| `SVSG-Windows-x64-Desktop` | 桌面 Agent MCP、界面文字识别、UI Automation、屏幕采集，以及使用本地像素函数的图片 API | 只让 Agent 操作电脑 |
| `SVSG-Windows-x64` | 以上全部，再加本地 YOLO 与 CrowdHuman 检测、Qt/QML、FFmpeg/FFprobe | 还要在本机做物体或行人检测 |

1. 将整个 ZIP 解压到可写的短路径，例如 `D:\VisFunc`，双击包目录里的 `Start.cmd`。首次启动会从模板创建本地 `.env`，并生成随机的 MCP 访问令牌 `VISFUNC_MCP_TOKEN`。缺少 Microsoft C++ 运行库时，启动器会调用随包的离线安装程序，Windows 可能提示授权。
2. 看到 “API ready” 和 “Desktop MCP ready” 后，打开 `http://127.0.0.1:3002/docs` 调用 API。Keystone 等支持 Streamable HTTP 的客户端连接 `http://127.0.0.1:8765/mcp`，令牌栏填 `.env` 里的 `VISFUNC_MCP_TOKEN`。没有这个令牌的程序既不能读屏幕也不能操作。
3. 默认只能读取桌面。要允许点击或填表，在 `allowed-windows.txt` 每行写一条规则，重启生效。推荐写成 `窗口标题中有区分度的文字 | 程序名.exe`，这样只有该程序的窗口能被操作，其他程序即使改成同样的标题也不行；`| 程序名.exe` 允许该程序的所有窗口；只写标题的旧写法仍然可用，启动时会提醒。保留控制台，按 Ctrl+C 停止两个服务。

## 运行条件与行为

- 需要已登录、未锁屏的 Windows 用户会话。作为 Windows 服务（会话 0）启动时，启动器直接拒绝。锁屏或出现 UAC、登录等安全桌面时，读屏和操作工具会报错并说明原因，不会截到黑屏或点到别处；动作在写入回执之前就被拒绝，解锁后可直接重试。
- 同一窗口上的调用依次执行，不同窗口的调用由 4 个工作线程同时处理，多个 Agent 各操作各的窗口时不会互相排队。
- 日志写在包目录的 `logs/`：`api.log` 和 `agent-mcp.log` 是两个服务的输出（超过 10 MB 换新文件），`agent-mcp-calls.jsonl` 每次工具调用一行，记录时间、工具、窗口标题、耗时和结果（超过 5 MB 轮换，保留 5 份）。输入的文字和查询内容不写入日志。
- 动作回执库默认在包目录的 `data/action-receipts.sqlite3`，重启后客户端仍可用原操作标识查询。
- `SVSG_HOST` 改成非本机地址时，必须先在 `.env` 设置 `SVSG_API_KEY`，否则启动器拒绝启动。

## 自检与使用

在包目录打开 PowerShell，执行 `.\Start.cmd --check`，它会真正启动服务，验证 API 健康检查、MCP 初始化和工具发现，然后退出。运行自检时需要两个空闲端口。`.env` 中的 `SVSG_PORT` 和 `VISFUNC_MCP_PORT` 可以修改，重复启动会保留配置和令牌。进程环境变量优先。

```powershell
.\MotionSpec.cmd screen-text "D:\samples\screen.png"
.\MotionSpec.cmd analyze "D:\samples\screen.mp4" --functions -o "D:\samples\analysis"
.\MotionSpec.cmd agent-mcp --help
```

API 默认启用本地像素函数编译。完整包里使用本地 YOLO 作为图片 API 的检测器时，将 `.env` 中的 `SVSG_DETECTOR` 改为 `yolo` 并重启。需要云端多模态识别或推理时，在 `.env` 填写自己的提供商配置。Keystone 客户端单独安装。

`source/` 包含项目源码、回归测试和文档。`manifest.json` 记录基线提交、包的种类、打包时各源码文件的 SHA-256、依赖版本以及模型、工具的 SHA-256。包外同名 `.zip.sha256` 可与 `Get-FileHash -Algorithm SHA256` 的结果比对。完整包的 FFmpeg 许可与构建说明在 `licenses/`，Python 和依赖的许可证随运行时保留。CrowdHuman 权重的来源见 [项目评测](../../docs/pedestrian-occlusion-evaluation-20261120.md)（在成品包内见 `source/docs/`）。YOLO11n 使用 Ultralytics 发布的权重。

## 从源码重建

构建机使用 Windows x64、CPython 3.12.10、Git、pip、wheel、setuptools。准备同版本官方 embeddable ZIP 和 Microsoft VC++ x64 离线安装包；完整包另需包含 `bin` 和 `LICENSE` 的 FFmpeg Windows 构建以及两个模型。执行以下命令，指定一个新的工作目录。默认由 pip 安装运行依赖；离线构建可重复传入 `--vendor`，指向已安装的对应平台依赖目录，桌面包只会从中拷出它的依赖闭包。最终用户只需解压成品。

```powershell
# 桌面包
python deployment/windows/build.py --profile desktop --work D:\build\desktop --output D:\releases `
  --python-embed D:\inputs\python-3.12.10-embed-amd64.zip --redist D:\inputs\vc_redist.x64.exe

# 完整包
python deployment/windows/build.py --work D:\build\full --output D:\releases `
  --python-embed D:\inputs\python-3.12.10-embed-amd64.zip `
  --ffmpeg D:\inputs\ffmpeg-full_build --redist D:\inputs\vc_redist.x64.exe `
  --yolo D:\inputs\yolo11n.pt --crowd D:\inputs\crowdhuman.onnx
```
