from __future__ import annotations

import re
import shlex
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from config import DomainConfig
from models import CandidateSample, CheckResult, EvidenceRef, RouteCode
from provider_errors import ProviderError
from services.execution_workspace import ExecutionWorkspace, ExecutionWorkspaceError, looks_like_placeholder_file, normalize_workspace_path
from services.runtime_requirements import validate_supported_container_runtime


def validate_environment_artifact(
    candidate: CandidateSample,
    domain: DomainConfig,
    *,
    execution_workspace: ExecutionWorkspace | None = None,
) -> CheckResult:
    if domain.domain_id != "benchmark_code_debug":
        return CheckResult(check_id="environment_artifact", passed=True)

    artifact = candidate.agent_artifact.environment_artifact
    if artifact is None:
        return _failed_environment("missing_workspace", "environment_artifact is required for code benchmarks", "environment_artifact")
    if artifact.kind != "executioner_workspace":
        return _failed_environment("missing_workspace", "code benchmarks require environment_artifact.kind='executioner_workspace'", "environment_artifact.kind")

    workspace: ExecutionWorkspace | None = None
    close_workspace = False
    try:
        if _workspace_matches_artifact(execution_workspace, artifact.payload):
            workspace = execution_workspace
        else:
            workspace = ExecutionWorkspace.from_artifact(artifact.payload, allow_exec=True)
            close_workspace = True
    except ExecutionWorkspaceError as exc:
        return _failed_environment(exc.subcode, exc.message, exc.path)
    structure_error = _workspace_structure_error(workspace, artifact.payload)
    if structure_error is not None:
        if close_workspace:
            workspace.close()
        return structure_error

    if not bool(domain.deterministic_rules.get("execute_workspace_tests", False)):
        if close_workspace:
            workspace.close()
        return CheckResult(check_id="environment_artifact", passed=True)

    runtime_requirements = candidate.agent_artifact.runtime_requirements
    runtime_error = validate_supported_container_runtime(runtime_requirements)
    if runtime_error is not None:
        if close_workspace:
            workspace.close()
        return CheckResult(check_id="environment_artifact", passed=True)

    command = workspace.commands.get("test")
    runtime_commands = runtime_requirements.get("commands") if isinstance(runtime_requirements, dict) else None
    if isinstance(runtime_commands, dict) and runtime_commands.get("test") != command:
        if close_workspace:
            workspace.close()
        return CheckResult(check_id="environment_artifact", passed=True)
    argv = _safe_test_command_argv(command)
    if argv is None:
        if close_workspace:
            workspace.close()
        return _failed_environment(
            "unsupported_workspace_test_command",
            "workspace.commands.test must be a pytest command without shell syntax",
            "environment_artifact.payload.commands.test",
        )

    timeout_seconds = float(domain.deterministic_rules.get("workspace_test_timeout_seconds", 10))
    try:
        completed = workspace.run_command(command, timeout_seconds=timeout_seconds)
    except TimeoutError as exc:
        return _failed_environment(
            "workspace_test_timeout",
            f"workspace test command timed out after {timeout_seconds:g}s",
            "environment_artifact.payload.commands.test",
            output=str(exc),
        )
    except (OSError, RuntimeError, ProviderError) as exc:
        return _failed_environment(
            "workspace_test_command_failed",
            f"workspace test command could not be executed: {exc}",
            "environment_artifact.payload.commands.test",
        )
    finally:
        if close_workspace:
            workspace.close()

    if completed.returncode == 1:
        max_failure_files = int(domain.deterministic_rules.get("max_initial_failure_files", 0))
        failed_files = _pytest_failed_test_files(completed.stdout, completed.stderr)
        if max_failure_files > 0 and len(failed_files) > max_failure_files:
            return _failed_environment(
                "workspace_test_command_failed",
                f"workspace starter tests fail across too many test files: {', '.join(failed_files)}",
                "environment_artifact.payload.commands.test",
                output=_short_command_output(completed.stdout, completed.stderr),
            )
        return CheckResult(
            check_id="environment_artifact",
            passed=True,
            evidence=[
                EvidenceRef(
                    source="workspace_command",
                    path="environment_artifact.payload.commands.test",
                    value=_short_command_output(completed.stdout, completed.stderr),
                )
            ],
        )
    if completed.returncode == 0 and bool(domain.deterministic_rules.get("require_initial_test_failure", False)):
        return _failed_environment(
            "workspace_tests_do_not_reproduce_failure",
            "workspace tests passed on the starter code; the benchmark does not demonstrate a failing behavior before repair",
            "environment_artifact.payload.commands.test",
            output=_short_command_output(completed.stdout, completed.stderr),
        )
    return _failed_environment(
        "workspace_test_command_failed",
        f"workspace test command exited with {completed.returncode}; tests did not run cleanly to assertions via {completed.executor}",
        "environment_artifact.payload.commands.test",
        output=_short_command_output(completed.stdout, completed.stderr),
    )


def _safe_test_command_argv(command: Any) -> list[str] | None:
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None
    if any(token in command for token in (";", "|", "&", ">", "<", "`", "$(", "\n")):
        return None
    executable = PurePosixPath(parts[0]).name
    if executable == "pytest":
        return parts
    if executable in {"python", "python3"} and len(parts) >= 3 and parts[1:3] == ["-m", "pytest"]:
        return parts
    return None


def _workspace_matches_artifact(workspace: ExecutionWorkspace | None, payload: dict[str, Any]) -> bool:
    if workspace is None:
        return False
    root_value = payload.get("workspace_root")
    if not isinstance(root_value, str) or not root_value.strip():
        return False
    try:
        return workspace.active_root.resolve() == Path(root_value).resolve()
    except OSError:
        return False


def _workspace_structure_error(workspace: ExecutionWorkspace, payload: dict[str, Any]) -> CheckResult | None:
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or len(raw_files) < 3:
        return _failed_environment("missing_workspace", "execution workspace must contain at least 3 files", "environment_artifact.payload.files")
    seen: set[str] = set()
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            return _failed_environment("invalid_workspace_file", "workspace file entry must be an object", f"environment_artifact.payload.files.{index}")
        try:
            path = normalize_workspace_path(item.get("path"), f"environment_artifact.payload.files.{index}.path")
        except ExecutionWorkspaceError as exc:
            return _failed_environment(exc.subcode, exc.message, exc.path)
        if path in seen:
            return _failed_environment("duplicate_workspace_path", f"workspace file path is duplicated: {path}", f"environment_artifact.payload.files.{index}.path")
        seen.add(path)
        try:
            content = workspace.read_file(path)
        except Exception as exc:
            return _failed_environment("invalid_workspace_file", f"workspace file could not be read: {exc}", f"environment_artifact.payload.files.{index}.path")
        if not content.strip() and Path(path).name != "__init__.py":
            return _failed_environment("invalid_workspace_file", "workspace file content is missing", f"environment_artifact.payload.files.{index}.path")
        if looks_like_placeholder_file(content):
            return _failed_environment("placeholder_workspace_file", "workspace file content is placeholder text", f"environment_artifact.payload.files.{index}.path")
    return None


def _short_command_output(stdout: str | None, stderr: str | None) -> str:
    output = "\n".join(part for part in [stdout or "", stderr or ""] if part).strip()
    if len(output) > 1600:
        return output[:1600] + "\n...[truncated]"
    return output


def _pytest_failed_test_files(stdout: str | None, stderr: str | None) -> list[str]:
    output = "\n".join(part for part in [stdout or "", stderr or ""] if part)
    failed_files: set[str] = set()
    for line in output.splitlines():
        match = re.match(r"^FAILED\s+([^:\s]+)::", line.strip())
        if match:
            failed_files.add(match.group(1))
    return sorted(failed_files)


def _failed_environment(subcode: str, value: str, path: str, *, output: str = "") -> CheckResult:
    evidence = [EvidenceRef(source="deterministic_rule", path=path, value=value)]
    if output:
        evidence.append(EvidenceRef(source="workspace_command", path=path, value=output))
    return CheckResult(
        check_id="environment_artifact",
        passed=False,
        route_code=RouteCode.REJECT_SCHEMA,
        subcode=subcode,
        evidence=evidence,
    )
