# SVSG + MotionSpec 单仓库

SVSG 保持根 Python 项目，MotionSpec 位于 `packages/motionspec/`，保留独立的
`pyproject.toml`、包名和 `motionspec` 命令。仓库只需一个 `.git`，没有子模块或嵌套仓库。
原 SVSG 的远程地址与服务启动脚本路径保持有效。

## 安装与测试

仓库根目录，Python 3.11+：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

测试配置覆盖两套测试目录，并显式把当前仓库源码放入 Python 搜索路径，防止误用之前
安装在其他目录的旧 MotionSpec。跨项目测试使用同一份 SVSG 函数实现。
合并后在 Python 3.12 上验证：243 项测试全部通过，SVSG Ruff 与 mypy 检查通过，
从仓库外调用 MotionSpec CLI 及共享函数查询也正常加载本仓库源码。

也可按需独立安装：

```bash
python -m pip install -e ".[api,llm]"
python -m pip install -e "./packages/motionspec"
```

第二条安装视频测量包；要使用 `--functions`，SVSG 还需 Pillow（`functions` 或 `api` 扩展）。
仅做函数导出时，可用 `python -m pip install -e ".[functions]" -e "./packages/motionspec"`。

单独运行测试：

```bash
python -m pytest tests
python -m pytest packages/motionspec/tests
python -m mypy --python-version 3.12
python -m ruff check svsg tests examples scripts/evaluate_recognition.py
```

复算验证工具从根目录运行：

```bash
python packages/motionspec/tools/validate_functions.py --out out/function-checks
```

MotionSpec 的 Qt 场景工具仍需额外安装 PySide6。根 Dockerfile 当前构建图像 API 服务，
统一 Python 开发环境同时提供视频 CLI；仓库合并不自动改变现有容器的运行职责。

## 代码边界

- `svsg/image_functions/`：像素、区域、轮廓与空间测量的唯一实现。
- `svsg/video_functions/`：帧函数、差分、时间函数、严格查询和识别证据绑定。
- `svsg/recognition/`：多模态候选发现与复核。
- `packages/motionspec/motionspec/`：视频解码、时段检测和运动拟合，按需导入 SVSG 函数接口。

两个包无循环导入。独立视频测量不需要启动 HTTP 服务，也不自动发起模型请求。

## 历史保留

2026-11-16，以 SVSG 的 `d875b04` 为主仓库基点，通过不压缩历史的 Git subtree 导入
MotionSpec 的 `258003a55f98653f986926b458a77c2aefd6c2a2`。
导入提交为 `f5881888838e9d38051cf309d0667909952ae097`，双方原提交均为当前历史的祖先。
导入时子目录 Git tree 与 MotionSpec 源提交完全一致，随后只调整单仓库入口和文档。

```bash
git log --graph --oneline --all
git log -- packages/motionspec
git show 258003a
```

后续代码统一在此仓库维护，原独立检出目录可留作本地备份。原提交中的路径仍按当时的
独立仓库布局显示，文件现有位置以当前树为准。

## 多个工作区协作

修改以当前合并仓库为准；从旧独立仓库迁移改动时，逐项审查补丁，避免整目录复制覆盖
统一时间协议、输入校验和回归测试。旧仓库单独通过测试，不能替代根目录的完整测试。

若从仓库外运行 CLI 时加载了旧副本，在合并仓库根目录重新安装：

```bash
python -m pip install -e "./packages/motionspec"
```

随后在不含 `motionspec/` 源码的目录检查 `python -c "import motionspec; print(motionspec.__file__)"`，
应指向本仓库的 `packages/motionspec/motionspec/`。不要手工修改 site-packages 中的
editable finder；安装记录和导入路径应由同一次安装生成。在旧仓库内运行 Python 仍可能
优先导入当前目录源码，因此后续开发、测试和提交都从合并仓库启动。
