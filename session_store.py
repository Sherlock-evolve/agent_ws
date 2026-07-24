import json
import os
import re
import stat
import tempfile
from pathlib import Path


SESSION_STORE_ROOT = Path(__file__).resolve().parent / ".agent_sessions"
MAX_SNAPSHOT_SIZE_BYTES = 5 * 1024 * 1024
SESSION_FILE_SUFFIX = ".json"
PENDING_FILE_SUFFIX = ".pending.json"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionStoreError(Exception):
    """本地会话仓库操作失败。"""


class InvalidSessionIdError(SessionStoreError, ValueError):
    """会话 ID 不符合安全格式。"""


class SessionNotFoundError(SessionStoreError, FileNotFoundError):
    """指定会话不存在。"""


class CorruptSessionError(SessionStoreError):
    """会话文件不是有效的仓库快照。"""


def _validate_session_id(session_id: str) -> str:
    if (
        not isinstance(session_id, str)
        or SESSION_ID_PATTERN.fullmatch(session_id) is None
    ):
        raise InvalidSessionIdError(
            "会话 ID 只能包含字母、数字、下划线和短横线，"
            "且长度必须在 1 到 64 之间"
        )
    return session_id


def _get_store_directory(*, create: bool) -> Path | None:
    store_root = SESSION_STORE_ROOT
    if store_root.is_symlink():
        raise SessionStoreError("会话目录不能是符号链接")

    try:
        root_mode = store_root.lstat().st_mode
    except FileNotFoundError:
        if not create:
            return None
        try:
            store_root.mkdir(mode=0o700)
        except OSError as error:
            raise SessionStoreError(
                f"无法创建会话目录：{error}"
            ) from error
    else:
        if not stat.S_ISDIR(root_mode):
            raise SessionStoreError("会话仓库路径不是目录")

    try:
        os.chmod(store_root, 0o700)
    except OSError as error:
        raise SessionStoreError(
            f"无法设置会话目录权限：{error}"
        ) from error
    return store_root


def _session_file_path(
    session_id: str,
    *,
    create_store: bool,
) -> Path:
    validated_id = _validate_session_id(session_id)
    store_root = _get_store_directory(create=create_store)
    if store_root is None:
        raise SessionNotFoundError(f"会话不存在：{validated_id}")
    return store_root / f"{validated_id}{SESSION_FILE_SUFFIX}"


def _pending_file_path(
    session_id: str,
    *,
    create_store: bool,
) -> Path:
    validated_id = _validate_session_id(session_id)
    store_root = _get_store_directory(create=create_store)
    if store_root is None:
        raise SessionNotFoundError(f"会话不存在：{validated_id}")
    return store_root / f"{validated_id}{PENDING_FILE_SUFFIX}"


def _ensure_regular_session_file(
    session_path: Path,
    session_id: str,
) -> os.stat_result:
    try:
        file_stat = session_path.lstat()
    except FileNotFoundError as error:
        raise SessionNotFoundError(
            f"会话不存在：{session_id}"
        ) from error

    if stat.S_ISLNK(file_stat.st_mode):
        raise SessionStoreError("会话文件不能是符号链接")
    if not stat.S_ISREG(file_stat.st_mode):
        raise SessionStoreError("会话路径不是常规文件")
    return file_stat


def _encode_snapshot(snapshot: dict) -> bytes:
    if not isinstance(snapshot, dict):
        raise SessionStoreError("会话快照必须是 JSON 对象")
    try:
        snapshot_text = json.dumps(
            snapshot,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        encoded_snapshot = snapshot_text.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise SessionStoreError(
            f"会话快照无法序列化为 UTF-8 JSON：{error}"
        ) from error

    if len(encoded_snapshot) > MAX_SNAPSHOT_SIZE_BYTES:
        raise SessionStoreError(
            f"会话快照超过 {MAX_SNAPSHOT_SIZE_BYTES} 字节上限"
        )
    return encoded_snapshot


def _reject_json_constant(value: str):
    raise ValueError(f"不允许的 JSON 常量：{value}")


def save(session_id: str, snapshot: dict) -> None:
    """使用原子替换保存单个会话快照。"""
    session_path = _session_file_path(
        session_id,
        create_store=True,
    )
    encoded_snapshot = _encode_snapshot(snapshot)

    if session_path.is_symlink():
        raise SessionStoreError("会话文件不能是符号链接")
    if session_path.exists():
        _ensure_regular_session_file(session_path, session_id)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=session_path.parent,
            prefix=f".{session_id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            os.fchmod(temporary_file.fileno(), 0o600)
            temporary_file.write(encoded_snapshot)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        if session_path.is_symlink():
            raise SessionStoreError("会话文件不能是符号链接")
        os.replace(temporary_path, session_path)
        temporary_path = None
        _fsync_directory(session_path.parent)
    except SessionStoreError:
        raise
    except OSError as error:
        raise SessionStoreError(
            f"保存会话失败：{error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def save_pending(session_id: str, record: dict) -> None:
    """原子保存一个等待重新审批的未提交轮次。"""
    pending_path = _pending_file_path(
        session_id,
        create_store=True,
    )
    encoded_record = _encode_snapshot(record)

    if pending_path.is_symlink():
        raise SessionStoreError("待审批文件不能是符号链接")
    if pending_path.exists():
        _ensure_regular_session_file(pending_path, session_id)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=pending_path.parent,
            prefix=f".{session_id}.pending.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            os.fchmod(temporary_file.fileno(), 0o600)
            temporary_file.write(encoded_record)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        if pending_path.is_symlink():
            raise SessionStoreError("待审批文件不能是符号链接")
        os.replace(temporary_path, pending_path)
        temporary_path = None
        _fsync_directory(pending_path.parent)
    except SessionStoreError:
        raise
    except OSError as error:
        raise SessionStoreError(
            f"保存待审批轮次失败：{error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def load(session_id: str) -> dict:
    """加载并验证单个 UTF-8 JSON 会话文件。"""
    session_path = _session_file_path(
        session_id,
        create_store=False,
    )
    _ensure_regular_session_file(
        session_path,
        session_id,
    )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(session_path, flags)
    except OSError as error:
        raise SessionStoreError(
            f"无法安全打开会话文件：{error}"
        ) from error
    try:
        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise SessionStoreError("会话路径不是常规文件")
        if opened_stat.st_size > MAX_SNAPSHOT_SIZE_BYTES:
            raise CorruptSessionError("会话文件超过允许的大小")
        with os.fdopen(file_descriptor, "rb", closefd=False) as file:
            encoded_snapshot = file.read(MAX_SNAPSHOT_SIZE_BYTES + 1)
    except (SessionStoreError, CorruptSessionError):
        raise
    except OSError as error:
        raise SessionStoreError(
            f"读取会话失败：{error}"
        ) from error
    finally:
        os.close(file_descriptor)

    if len(encoded_snapshot) > MAX_SNAPSHOT_SIZE_BYTES:
        raise CorruptSessionError("会话文件超过允许的大小")

    try:
        snapshot_text = encoded_snapshot.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorruptSessionError(
            "会话文件不是有效的 UTF-8 文本"
        ) from error
    try:
        snapshot = json.loads(
            snapshot_text,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise CorruptSessionError(
            f"会话文件包含无效 JSON：{error}"
        ) from error
    if not isinstance(snapshot, dict):
        raise CorruptSessionError("会话文件顶层必须是 JSON 对象")
    return snapshot


def load_pending(session_id: str) -> dict | None:
    """加载待审批记录；不存在时返回 None。"""
    try:
        pending_path = _pending_file_path(
            session_id,
            create_store=False,
        )
        _ensure_regular_session_file(pending_path, session_id)
    except SessionNotFoundError:
        return None

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(pending_path, flags)
    except OSError as error:
        raise SessionStoreError(
            f"无法安全打开待审批文件：{error}"
        ) from error
    try:
        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise SessionStoreError("待审批路径不是常规文件")
        if opened_stat.st_size > MAX_SNAPSHOT_SIZE_BYTES:
            raise CorruptSessionError("待审批文件超过允许的大小")
        with os.fdopen(file_descriptor, "rb", closefd=False) as file:
            encoded_record = file.read(MAX_SNAPSHOT_SIZE_BYTES + 1)
    except (SessionStoreError, CorruptSessionError):
        raise
    except OSError as error:
        raise SessionStoreError(
            f"读取待审批轮次失败：{error}"
        ) from error
    finally:
        os.close(file_descriptor)

    if len(encoded_record) > MAX_SNAPSHOT_SIZE_BYTES:
        raise CorruptSessionError("待审批文件超过允许的大小")
    try:
        record = json.loads(
            encoded_record.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CorruptSessionError(
            f"待审批文件包含无效 JSON：{error}"
        ) from error
    if not isinstance(record, dict):
        raise CorruptSessionError("待审批文件顶层必须是 JSON 对象")
    return record


def list_sessions() -> list[str]:
    """列出名称合法的常规会话文件。"""
    store_root = _get_store_directory(create=False)
    if store_root is None:
        return []

    session_ids = []
    try:
        entries = list(store_root.iterdir())
    except OSError as error:
        raise SessionStoreError(
            f"无法列出会话目录：{error}"
        ) from error

    for entry in entries:
        if entry.name.endswith(PENDING_FILE_SUFFIX):
            session_id = entry.name[: -len(PENDING_FILE_SUFFIX)]
        elif entry.name.endswith(SESSION_FILE_SUFFIX):
            session_id = entry.name[: -len(SESSION_FILE_SUFFIX)]
        else:
            continue
        if SESSION_ID_PATTERN.fullmatch(session_id) is None:
            continue
        try:
            entry_mode = entry.lstat().st_mode
        except OSError:
            continue
        if stat.S_ISREG(entry_mode):
            session_ids.append(session_id)

    return sorted(set(session_ids))


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError as error:
        raise SessionStoreError(
            f"无法打开会话目录进行同步：{error}"
        ) from error
    try:
        os.fsync(directory_fd)
    except OSError as error:
        raise SessionStoreError(
            f"无法同步会话目录：{error}"
        ) from error
    finally:
        os.close(directory_fd)


def delete(session_id: str) -> None:
    """删除单个明确指定的会话及其待审批记录并同步目录。"""
    validated_id = _validate_session_id(session_id)
    store_root = _get_store_directory(create=False)
    if store_root is None:
        raise SessionNotFoundError(f"会话不存在：{validated_id}")

    candidates = [
        store_root / f"{validated_id}{SESSION_FILE_SUFFIX}",
        store_root / f"{validated_id}{PENDING_FILE_SUFFIX}",
    ]
    existing = []
    for candidate in candidates:
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        _ensure_regular_session_file(candidate, validated_id)
        existing.append(candidate)
    if not existing:
        raise SessionNotFoundError(f"会话不存在：{validated_id}")

    try:
        for candidate in existing:
            candidate.unlink()
    except OSError as error:
        raise SessionStoreError(
            f"删除会话失败：{error}"
        ) from error
    _fsync_directory(store_root)


def delete_pending(
    session_id: str,
    *,
    missing_ok: bool = True,
) -> None:
    """删除待审批记录，不影响已提交会话快照。"""
    try:
        pending_path = _pending_file_path(
            session_id,
            create_store=False,
        )
        _ensure_regular_session_file(pending_path, session_id)
    except SessionNotFoundError:
        if missing_ok:
            return
        raise
    try:
        pending_path.unlink()
    except OSError as error:
        raise SessionStoreError(
            f"删除待审批轮次失败：{error}"
        ) from error
    _fsync_directory(pending_path.parent)
