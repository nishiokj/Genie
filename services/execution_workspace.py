from __future__ import annotations

import sys
import tempfile
import shlex
import shutil
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from provider_errors import ProviderError


PLACEHOLDER_TEXTS = {"...", "todo", "tbd", "<content>", "<file content>", "# todo", "# tbd"}
PLACEHOLDER_FRAGMENTS = (
    "implementation omitted",
    "content omitted",
    "placeholder file",
    "todo: implement",
    "replace with actual",
)


class ExecutionWorkspaceError(ValueError):
    def __init__(self, subcode: str, path: str, message: str) -> None:
        super().__init__(message)
        self.subcode = subcode
        self.path = path
        self.message = message


@dataclass(frozen=True)
class WorkspaceCommandResult:
    returncode: int
    stdout: str
    stderr: str
    executor: str = "executioner"
    detail: str = ""


class ExecutionWorkspace:
    def __init__(
        self,
        *,
        root: Path | None = None,
        commands: dict[str, str] | None = None,
        binary_path: str | None = None,
        allow_exec: bool = False,
        timeout_ms: int = 30_000,
    ) -> None:
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if root is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="benchmark-executioner-workspace-")
            root = Path(self._temp_dir.name)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._active_subdir = ""
        self.commands = dict(commands or {})
        self._env = _create_executioner_env(
            root=self.root,
            binary_path=binary_path,
            allow_exec=allow_exec,
            timeout_ms=timeout_ms,
        )

    @property
    def active_root(self) -> Path:
        return self.root / self._active_subdir if self._active_subdir else self.root

    @classmethod
    def from_artifact(
        cls,
        artifact_payload: dict[str, Any],
        *,
        binary_path: str | None = None,
        allow_exec: bool = False,
        timeout_ms: int = 30_000,
    ) -> "ExecutionWorkspace":
        root_value = artifact_payload.get("workspace_root")
        if not isinstance(root_value, str) or not root_value.strip():
            raise ExecutionWorkspaceError("missing_workspace", "environment_artifact.payload.workspace_root", "execution workspace root is required")
        root = Path(root_value)
        if not root.exists() or not root.is_dir():
            raise ExecutionWorkspaceError("missing_workspace", "environment_artifact.payload.workspace_root", "execution workspace root does not exist")
        commands = artifact_payload.get("commands")
        return cls(
            root=root,
            commands={str(key): str(value) for key, value in commands.items()} if isinstance(commands, dict) else {},
            binary_path=binary_path,
            allow_exec=allow_exec,
            timeout_ms=timeout_ms,
        )

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    def reset(self, subdir: str | None = None) -> None:
        if subdir is not None:
            self._active_subdir = _normalize_workspace_subdir(subdir)
        active_root = self.active_root
        active_root.mkdir(parents=True, exist_ok=True)
        for child in active_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def write_file(self, path: str, content: str) -> None:
        normalized = normalize_workspace_path(path, "path")
        _validate_file_content(normalized, content)
        if normalized in self.list_files():
            current = self.read_file(normalized)
            if current == content:
                return
            self._submit(
                "Edit",
                {"path": normalized, "oldString": current, "newString": content, "replaceAll": False},
            )
            return
        self._submit("Write", {"path": normalized, "content": content})

    def edit_file(
        self,
        *,
        path: str,
        old_text: str | None,
        new_text: str,
        create_if_missing: bool = False,
        replace_all: bool = False,
    ) -> None:
        normalized = normalize_workspace_path(path, "path")
        _validate_file_content(normalized, new_text)
        exists = normalized in self.list_files()
        if create_if_missing:
            if old_text not in (None, ""):
                raise ExecutionWorkspaceError("invalid_workspace_file", "old_text", "old_text must be omitted when create_if_missing=true")
            if exists:
                raise ExecutionWorkspaceError("invalid_workspace_file", "path", f"path already exists: {normalized}")
            self.write_file(normalized, new_text)
            return
        if not isinstance(old_text, str) or old_text == "":
            raise ExecutionWorkspaceError("invalid_workspace_file", "old_text", "old_text must be a non-empty exact string")
        self._submit(
            "Edit",
            {
                "path": normalized,
                "oldString": old_text,
                "newString": new_text,
                "replaceAll": bool(replace_all),
            },
        )

    def read_file(self, path: str) -> str:
        normalized = normalize_workspace_path(path, "path")
        result = self._submit("Read", {"path": normalized})
        return str(result.output)

    def list_files(self) -> list[str]:
        return sorted(_list_files_recursive(self._env, self._logical_cwd()))

    def run_command(self, command: str, *, timeout_seconds: float) -> WorkspaceCommandResult:
        command = _local_command(command)
        result = self._env.submit(
            {
                "toolName": "Bash",
                "arguments": {"command": command, "timeout": int(timeout_seconds)},
                "cwd": self._logical_cwd(),
            }
        )
        output = str(result.output or "")
        stdout, stderr = _split_bash_output(output)
        returncode = int(result.metadata.get("returnCode", 1)) if isinstance(result.metadata, dict) else 1
        if result.status == "timeout":
            raise TimeoutError(result.error or f"Bash timed out after {timeout_seconds:g}s")
        return WorkspaceCommandResult(returncode=returncode, stdout=stdout, stderr=stderr)

    def artifact_payload(self) -> dict[str, Any]:
        session = self._env.session
        payload = {
            "session_id": session.id,
            "logical_root": session.workspace.logicalRoot,
            "workspace_root": str(self.active_root),
            "commands": dict(self.commands),
            "files": [{"path": path} for path in self.list_files()],
        }
        artifact = _sdk_artifact_payload(self._env)
        if artifact is not None:
            payload["artifact"] = artifact
        return payload

    def _submit(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        result = self._env.submit(
            {
                "toolName": tool_name,
                "arguments": arguments,
                "cwd": self._logical_cwd(),
            }
        )
        if result.status != "success":
            raise ProviderError(result.error or f"Executioner {tool_name} failed")
        return result

    def _logical_cwd(self) -> str:
        if not self._active_subdir:
            return "/workspace"
        return f"/workspace/{self._active_subdir}"


def normalize_workspace_path(path: Any, ref: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ExecutionWorkspaceError("invalid_workspace_file", ref, "workspace file path is missing")
    normalized_path = path.strip()
    parts = PurePosixPath(normalized_path).parts
    if normalized_path.startswith("/") or "\\" in normalized_path or ".." in parts:
        raise ExecutionWorkspaceError("invalid_workspace_path", ref, "workspace file paths must be relative POSIX paths and cannot traverse upward")
    return normalized_path


def _normalize_workspace_subdir(path: str) -> str:
    normalized = normalize_workspace_path(path, "workspace_subdir").rstrip("/")
    return "" if normalized == "." else normalized


def _list_files_recursive(env: Any, cwd: str, prefix: str = "") -> list[str]:
    list_files = getattr(env, "list_files", None)
    if not callable(list_files):
        raise ProviderError("Executioner SDK does not expose list_files")
    paths: list[str] = []
    for entry in list_files(cwd=cwd):
        name = str(entry).strip()
        if not name:
            continue
        if name.endswith("/"):
            dirname = name.rstrip("/")
            child_prefix = f"{prefix}{dirname}/"
            child_cwd = f"{cwd.rstrip('/')}/{dirname}"
            paths.extend(_list_files_recursive(env, child_cwd, child_prefix))
            continue
        path = f"{prefix}{name}"
        if _include_workspace_file(path):
            paths.append(path)
    return paths


def _include_workspace_file(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return ".validation-task" not in parts and parts[:1] != ("task",)


def _sdk_artifact_payload(env: Any) -> dict[str, Any] | None:
    to_artifact = getattr(env, "to_artifact", None)
    if not callable(to_artifact):
        return None
    artifact = to_artifact()
    if is_dataclass(artifact):
        return asdict(artifact)
    if isinstance(artifact, dict):
        return artifact
    model_dump = getattr(artifact, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return None


def looks_like_placeholder_file(content: str) -> bool:
    stripped = content.strip().lower()
    if not stripped:
        return False
    if looks_like_placeholder_text(stripped):
        return True
    return any(fragment in stripped for fragment in PLACEHOLDER_FRAGMENTS)


def looks_like_placeholder_text(value: str) -> bool:
    return value.strip().lower() in PLACEHOLDER_TEXTS


def _validate_file_content(path: str, content: str) -> None:
    if not isinstance(content, str) or (not content.strip() and not _allows_empty_file(path)):
        raise ExecutionWorkspaceError("invalid_workspace_file", "content", "workspace file content is missing")
    if looks_like_placeholder_file(content):
        raise ExecutionWorkspaceError("placeholder_workspace_file", "content", "workspace file content is placeholder text")


def _allows_empty_file(path: str) -> bool:
    return PurePosixPath(path).name == "__init__.py"


def _split_bash_output(output: str) -> tuple[str, str]:
    marker = "\n[stderr]: "
    if marker not in output:
        return output, ""
    stdout, stderr = output.split(marker, 1)
    return stdout, stderr


def _local_command(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if parts and Path(parts[0]).name in {"python", "python3"}:
        parts[0] = sys.executable
        return " ".join(shlex.quote(part) for part in parts)
    return command


def _create_executioner_env(*, root: Path, binary_path: str | None, allow_exec: bool, timeout_ms: int) -> Any:
    ExecutionerEnvironment = _load_executioner_environment()
    return ExecutionerEnvironment.create(
        binaryPath=binary_path,
        workspace={"kind": "existing", "root": str(root)},
        worker={"kind": "managed", "id": "synth-pipeline-worker", "idleSleepMs": 1},
        policy={
            "readRoots": ["/workspace"],
            "writeRoots": ["/workspace"],
            "process": {"allowExec": allow_exec},
            "network": {"enabled": False},
            "maxDurationMs": timeout_ms,
            "maxOutputBytes": 100_000,
        },
        lifecycle={"destroyOnClose": True, "cleanupQueueOnClose": True, "cleanupStateOnClose": True},
        submitTimeoutMs=timeout_ms,
    )


def _load_executioner_environment() -> Any:
    try:
        from executioner_sdk import ExecutionerEnvironment

        return ExecutionerEnvironment
    except ModuleNotFoundError:
        sibling_src = Path(__file__).resolve().parents[2] / "substrate" / "packages" / "executioner-python" / "src"
        if sibling_src.exists():
            sys.path.insert(0, str(sibling_src))
            from executioner_sdk import ExecutionerEnvironment

            return ExecutionerEnvironment
        raise
