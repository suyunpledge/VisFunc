# VisFunc · Windows x64 一键部署

1. 将整个 ZIP 解压到可写的短路径，例如 `D:\VisFunc`。打开其中的 `SVSG-Windows-x64`，双击 `Start.cmd`。首次启动会从模板创建本地 `.env`。缺少 Microsoft C++ 运行库时，启动器会调用随包的离线安装程序，Windows 可能提示授权。
2. 看到 “API ready” 和 “Desktop MCP ready” 后，打开 `http://127.0.0.1:3002/docs` 调用 API。Keystone 等支持 Streamable HTTP 的客户端连接 `http://127.0.0.1:8765/mcp`。桌面感知需要已登录的交互式 Windows 会话。
3. 默认可以读取桌面。要允许点击或填表，在 `allowed-windows.txt` 逐行填写目标窗口标题中有区分度的文字，重启生效。保留控制台，按 Ctrl+C 停止两个服务。

## 自检与使用

在包目录打开 PowerShell，执行 `.\Start.cmd --check`，它会真正启动服务，验证 API 健康检查、MCP 初始化和工具发现，然后退出。运行自检时需要两个空闲端口。`.env` 中的 `SVSG_PORT` 和 `VISFUNC_MCP_PORT` 可以修改，重复启动会保留配置。进程环境变量优先。设置 `VISFUNC_MCP_TOKEN` 后，客户端需发送 Bearer token。

```powershell
.\MotionSpec.cmd screen-text "D:\samples\screen.png"
.\MotionSpec.cmd analyze "D:\samples\screen.mp4" --functions -o "D:\samples\analysis"
.\MotionSpec.cmd agent-mcp --help
```

API 默认启用本地像素函数编译。包内包含 CPython 3.12.10、两套 Python 入口、RapidOCR 及其模型、UI Automation、屏幕采集、Qt/QML、CPU 推理依赖、FFmpeg/FFprobe，以及 `models/yolo11n.pt` 和 `models/crowdhuman.onnx`。使用本地 YOLO 作为图片 API 的检测器时，将 `.env` 中的 `SVSG_DETECTOR` 改为 `yolo` 并重启。需要云端多模态识别或推理时，在 `.env` 填写自己的提供商配置。Keystone 客户端单独安装。

`source/` 包含项目源码、回归测试和文档。`manifest.json` 记录基线提交、打包时各源码文件的 SHA-256、依赖版本以及模型、工具的 SHA-256。包外同名 `.zip.sha256` 可与 `Get-FileHash -Algorithm SHA256` 的结果比对。FFmpeg 许可与构建说明在 `licenses/`，Python 和依赖的许可证随运行时保留。CrowdHuman 权重的来源见 [项目评测](../../docs/pedestrian-occlusion-evaluation-20261120.md)（在成品包内见 `source/docs/`）。YOLO11n 使用 Ultralytics 发布的权重。

## 从源码重建

构建机使用 Windows x64、CPython 3.12.10、Git、pip、wheel、setuptools。准备同版本官方 embeddable ZIP、包含 `bin` 和 `LICENSE` 的 FFmpeg Windows 构建、Microsoft VC++ x64 离线安装包，以及两个模型。执行以下命令，指定一个新的工作目录。默认由 pip 安装运行依赖，离线构建可重复传入 `--vendor`，指向已安装的对应平台依赖目录。最终用户只需解压成品。

```powershell
python deployment/windows/build.py --work D:\build\new --output D:\releases `
  --python-embed D:\inputs\python-3.12.10-embed-amd64.zip `
  --ffmpeg D:\inputs\ffmpeg-full_build --redist D:\inputs\vc_redist.x64.exe `
  --yolo D:\inputs\yolo11n.pt --crowd D:\inputs\crowdhuman.onnx
```
