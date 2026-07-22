from pathlib import Path, PureWindowsPath

from langchain_core.tools import tool


WORKSPACE_ROOT = Path(__file__).resolve().parent
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}


def _resolve_workspace_path(path: str) -> Path:
    """校验并解析工作区内的相对路径。"""
    relative_path = Path(path)
    windows_path = PureWindowsPath(path)

    if (
        relative_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.root)
    ):
        raise ValueError("不允许使用绝对路径")

    if ".." in relative_path.parts or ".." in windows_path.parts:
        raise ValueError("不允许使用 '..' 进行路径穿越")

    if any(part in IGNORED_DIRECTORIES for part in relative_path.parts):
        raise ValueError("不允许访问被忽略的目录")

    resolved_path = (WORKSPACE_ROOT / relative_path).resolve()
    try:
        resolved_relative_path = resolved_path.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError("路径超出当前项目目录") from error

    if any(part in IGNORED_DIRECTORIES for part in resolved_relative_path.parts):
        raise ValueError("不允许访问被忽略的目录")

    return resolved_path


@tool
def list_files(directory: str = ".") -> str:
    """列出当前项目指定目录的直接子项；目录必须是项目根目录下的相对路径。"""
    directory_path = _resolve_workspace_path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"不是目录：{directory}")

    entries = []
    for entry in directory_path.iterdir():
        if entry.name in IGNORED_DIRECTORIES and entry.is_dir():
            continue

        resolved_entry = entry.resolve()
        try:
            relative_entry = resolved_entry.relative_to(WORKSPACE_ROOT)
        except ValueError:
            continue

        if any(part in IGNORED_DIRECTORIES for part in relative_entry.parts):
            continue

        display_path = entry.relative_to(WORKSPACE_ROOT).as_posix()
        if resolved_entry.is_dir():
            display_path += "/"
        entries.append(display_path)

    return "\n".join(sorted(entries)) or "（目录为空）"


@tool
def read_file(path: str) -> str:
    """读取当前项目内指定 UTF-8 文本文件；文件必须使用项目根目录下的相对路径。"""
    file_path = _resolve_workspace_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"不是文件：{path}")

    return file_path.read_text(encoding="utf-8")
