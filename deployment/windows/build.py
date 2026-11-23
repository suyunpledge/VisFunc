"""Build a relocatable Windows x64 ZIP from local runtime, model and wheel inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import struct
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc")
LAUNCH_FILES = ("Start.cmd", "MotionSpec.cmd", "launch.py", "allowed-windows.txt", "README.zh.md")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(args):
    if sys.platform != "win32" or struct.calcsize("P") != 8 or sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError("build with the matching CPython 3.12.10 Windows x64 installation")
    root = args.work / "SVSG-Windows-x64"
    root.mkdir(parents=True, exist_ok=False)
    runtime = root / "runtime"
    with zipfile.ZipFile(args.python_embed) as archive:
        if "python312.dll" not in archive.namelist():
            raise ValueError("expected CPython 3.12 embeddable runtime")
        archive.extractall(runtime)
    # TorchScript needs inspectable Python source for some standard library functions.
    shutil.copytree(
        sysconfig.get_path("stdlib"),
        runtime / "stdlib",
        ignore=shutil.ignore_patterns(
            "site-packages", "__pycache__", "*.pyc", "test", "tests", "idlelib", "ensurepip"
        ),
    )
    (runtime / "python312._pth").write_text(
        "stdlib\npython312.zip\n.\nLib/site-packages\n../source\n../source/packages/motionspec\n"
        "import site\n",
        encoding="ascii",
    )
    site = runtime / "Lib" / "site-packages"
    source = root / "source"
    # Copy the working tree, including current fixes and regressions. Exclude local config,
    # generated caches and historic benchmark transcripts from deployment artifacts.
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=PROJECT)
    paths = set(tracked.decode("utf-8").split("\0")) - {""}
    paths.update(f"deployment/windows/{name}" for name in (*LAUNCH_FILES, "build.py"))
    extra_paths = {
        (PROJECT / extra).resolve(strict=True).relative_to(PROJECT).as_posix()
        for extra in args.include_source
    }
    paths.update(extra_paths)
    for name in sorted(paths):
        if name.startswith("archive/") or (
            name.startswith("packages/motionspec/bench/results/")
            and not name.endswith(".md")
            and name not in extra_paths
        ):
            continue
        src = PROJECT / name
        dest = source / name
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    if args.vendor:
        for directory in args.vendor:
            # pip-generated console wrappers point to the build machine's Python.
            def wheel_files(path, names, vendor=directory):
                ignored = set(SKIP(path, names))
                if Path(path) == vendor:
                    ignored.add("bin")
                return ignored

            shutil.copytree(directory, site, dirs_exist_ok=True, ignore=wheel_files)
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-compile",
                "--target",
                str(site),
                f"{source}[api,llm,ml,crowd,overlays,uia]",
                f"{source / 'packages' / 'motionspec'}[capture]",
                "PySide6",
            ],
            check=True,
        )
    # Ensure that both imports and distribution metadata identify this exact source version.
    # Install wheels in a separate directory to avoid replacing any third-party dependency.
    own = args.work / "projects"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-compile",
            "--target",
            str(own),
            str(source),
            str(source / "packages" / "motionspec"),
        ],
        check=True,
    )
    for dist in own.glob("*.dist-info"):
        dest = site / dist.name
        shutil.copytree(dist, dest, dirs_exist_ok=True)
        (dest / "direct_url.json").unlink(missing_ok=True)
    for name in ("svsg", "motionspec"):
        # The editable source under source/ is the runtime project code.
        if (site / name).exists():
            shutil.rmtree(site / name)
    for project in [source, source / "packages" / "motionspec"]:
        for path in [project / "build", *project.glob("*.egg-info")]:
            if path.is_dir():
                shutil.rmtree(path)
    for filename in LAUNCH_FILES:
        shutil.copy2(PROJECT / "deployment" / "windows" / filename, root / filename)
    config = (source / ".env.example").read_text(encoding="utf-8")
    config = config.replace("SVSG_DETECTOR=disabled", "SVSG_DETECTOR=functions").replace(
        "SVSG_YOLO_WEIGHTS=yolov8n.pt", "SVSG_YOLO_WEIGHTS=models/yolo11n.pt"
    )
    (root / ".env.example").write_text(
        config + "\nVISFUNC_MCP_PORT=8765\n# VISFUNC_MCP_TOKEN=\n", encoding="utf-8"
    )
    binaries = {
        "bin/ffmpeg.exe": args.ffmpeg / "bin/ffmpeg.exe",
        "bin/ffprobe.exe": args.ffmpeg / "bin/ffprobe.exe",
        "licenses/ffmpeg-LICENSE": args.ffmpeg / "LICENSE",
        "licenses/ffmpeg-README.txt": args.ffmpeg / "README.txt",
        "models/yolo11n.pt": args.yolo,
        "models/crowdhuman.onnx": args.crowd,
        "prerequisites/vc_redist.x64.exe": args.redist,
    }
    for name, src in binaries.items():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    versions = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions(path=[str(site)])
        if d.metadata["Name"]
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "base_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
                ).strip(),
                "source_sha256": {
                    p.relative_to(source).as_posix(): sha256(p)
                    for p in sorted(source.rglob("*"))
                    if p.is_file()
                },
                "python": sys.version,
                "python_embed_sha256": sha256(args.python_embed),
                "bundled_files_sha256": {name: sha256(root / name) for name in binaries},
                "packages": dict(sorted(versions.items())),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def pack(root, output):
    output.mkdir(parents=True, exist_ok=True)
    target = output / (root.name + ".zip")
    # Only publish the final path after the CRC check has passed.
    temp = output / (target.name + ".partial")
    with zipfile.ZipFile(temp, "x", zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(root.parent))
    with zipfile.ZipFile(temp) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"CRC failure: {bad}")
    digest = sha256(temp)
    temp.rename(target)
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    print(target, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "output", "python-embed", "ffmpeg", "redist", "yolo", "crowd"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--vendor", action="append", type=Path, help="preinstalled wheel directory")
    parser.add_argument(
        "--include-source",
        action="append",
        default=[],
        metavar="FILE",
        help="also include this repository file, including benchmark evidence",
    )
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.vendor:
        args.vendor = [path.resolve() for path in args.vendor]
    output = args.output / "SVSG-Windows-x64.zip"
    if output.exists() or output.with_suffix(".zip.partial").exists():
        parser.error("choose an output directory without an existing bundle or partial build")
    pack(build(args), args.output)


if __name__ == "__main__":
    main()
