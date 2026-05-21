from __future__ import annotations

from typing import Any


APPROVED_PYTHON_IMAGES = {
    "python:3.10-slim",
    "python:3.11-slim",
    "python:3.12-slim",
}


def validate_supported_container_runtime(runtime_requirements: dict[str, Any] | None) -> str | None:
    if not isinstance(runtime_requirements, dict) or runtime_requirements.get("kind") != "filesystem_task":
        return "executable workspaces must declare runtime_requirements.kind='filesystem_task'"

    execution = runtime_requirements.get("execution")
    if not isinstance(execution, dict) or execution.get("mode") not in {"task_image", "container"}:
        return "filesystem_task execution.mode must be task_image or container"
    base_image = execution.get("base_image")
    if base_image not in APPROVED_PYTHON_IMAGES:
        return f"filesystem_task base_image must be one of {sorted(APPROVED_PYTHON_IMAGES)}"

    language = runtime_requirements.get("language")
    if not isinstance(language, dict) or str(language.get("name", "")).lower() != "python":
        return "filesystem_task language.name must be python"

    dependencies = runtime_requirements.get("dependencies")
    if isinstance(dependencies, dict):
        policy = dependencies.get("policy")
        packages = dependencies.get("packages")
        if policy not in {None, "none", "stdlib_plus_runner"}:
            return "only dependency policy stdlib_plus_runner is supported for container validation"
        if packages is not None:
            if not isinstance(packages, list) or any(package != "pytest" for package in packages):
                return "only the pytest runner package is supported for container validation"

    network = runtime_requirements.get("network")
    if network not in {None, "disabled_during_eval"}:
        return "only disabled_during_eval network policy is supported for container validation"

    return None
