from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from codex_client import CodexResponse

def _strip_json_markdown_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(content)


def _structured_response_parts(response: Any) -> tuple[Any, Any | None, Any | None]:
    if isinstance(response, dict) and {"raw", "parsed", "parsing_error"}.intersection(response):
        return response.get("parsed"), response.get("raw"), response.get("parsing_error")
    return response, response, None


def _schema_with_title(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("title"):
        return schema
    titled = deepcopy(schema)
    titled["title"] = "StructuredOutput"
    return titled


def _structured_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return _strict_json_schema(_schema_with_title(schema))


def _codex_structured_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return _codex_ensure_root_object(_codex_normalize_for_codex(_structured_output_schema(schema)))


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict = deepcopy(schema)
    strict.pop("$schema", None)
    _make_schema_strict(strict)
    return strict


def _make_schema_strict(schema: Any) -> None:
    if not isinstance(schema, dict):
        return
    defs = schema.get("$defs")
    if isinstance(defs, dict):
        for item in defs.values():
            _make_schema_strict(item)
    definitions = schema.get("definitions")
    if isinstance(definitions, dict):
        for item in definitions.values():
            _make_schema_strict(item)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        originally_required = set(schema.get("required", []))
        for name, property_schema in list(properties.items()):
            _make_schema_strict(property_schema)
            if name not in originally_required:
                properties[name] = _nullable_json_schema(property_schema)
        schema["required"] = list(properties.keys())

    items = schema.get("items")
    if isinstance(items, dict):
        _make_schema_strict(items)
    elif isinstance(items, list):
        for item in items:
            _make_schema_strict(item)

    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                _make_schema_strict(variant)

    if _json_schema_has_object_type(schema) or isinstance(properties, dict):
        schema["additionalProperties"] = False
        if "required" not in schema:
            schema["required"] = []
        if "properties" not in schema:
            schema["properties"] = {}


def _nullable_json_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    optional = deepcopy(schema)
    schema_type = optional.get("type")
    if isinstance(schema_type, str):
        if schema_type != "null":
            optional["type"] = [schema_type, "null"]
        _add_null_to_enum(optional)
        return optional
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            optional["type"] = [*schema_type, "null"]
        _add_null_to_enum(optional)
        return optional
    return {"anyOf": [optional, {"type": "null"}]}


def _add_null_to_enum(schema: dict[str, Any]) -> None:
    enum = schema.get("enum")
    if isinstance(enum, list) and None not in enum:
        schema["enum"] = [*enum, None]


def _json_schema_has_object_type(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    return schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type)


_SCHEMA_MAP_KEYS = {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
_JSON_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}


def _codex_is_schema_object(value: Any) -> bool:
    return isinstance(value, dict)


def _codex_transform_child_value(value: Any, parent_key: str) -> Any:
    if isinstance(value, list):
        return [_codex_normalize_for_codex(entry) if _codex_is_schema_object(entry) else entry for entry in value]
    if not _codex_is_schema_object(value):
        return value
    if parent_key in _SCHEMA_MAP_KEYS:
        return {
            child_key: _codex_normalize_for_codex(child_value) if _codex_is_schema_object(child_value) else child_value
            for child_key, child_value in value.items()
        }
    return _codex_normalize_for_codex(value)


def _codex_const_type(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return None


def _codex_enum_types(values: list[Any]) -> list[str]:
    return [item for item in _unique_json_values([_codex_const_type(value) for value in values]) if isinstance(item, str)]


def _codex_type_list(type_value: Any) -> list[str]:
    if isinstance(type_value, str) and type_value in _JSON_SCHEMA_TYPES:
        return [type_value]
    if isinstance(type_value, list):
        return _unique_json_values([value for value in type_value if isinstance(value, str) and value in _JSON_SCHEMA_TYPES])
    return []


def _codex_schema_types(schema: dict[str, Any]) -> list[str]:
    declared = _codex_type_list(schema.get("type"))
    if declared:
        return declared
    if "const" in schema:
        const_type = _codex_const_type(schema.get("const"))
        return [const_type] if const_type else []
    enum = schema.get("enum")
    if isinstance(enum, list):
        return _codex_enum_types(enum)
    return []


def _codex_with_type_list(schema: dict[str, Any], types: list[str]) -> dict[str, Any]:
    unique_types = _unique_json_values(types)
    if not unique_types:
        return schema
    out = dict(schema)
    out["type"] = unique_types[0] if len(unique_types) == 1 else unique_types
    return out


def _codex_merge_with_fallback(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    out = {**secondary, **primary}
    if isinstance(secondary.get("properties"), dict) and isinstance(primary.get("properties"), dict):
        out["properties"] = {**secondary["properties"], **primary["properties"]}
    if isinstance(secondary.get("required"), list) and isinstance(primary.get("required"), list):
        out["required"] = _unique_json_values([*secondary["required"], *primary["required"]])
    return out


def _codex_merge_object_alternatives(options: list[dict[str, Any]]) -> dict[str, Any]:
    property_names: set[str] = set()
    for option in options:
        properties = option.get("properties")
        if isinstance(properties, dict):
            property_names.update(str(key) for key in properties)

    merged_properties: dict[str, Any] = {}
    for property_name in property_names:
        property_schemas = []
        for option in options:
            properties = option.get("properties")
            if isinstance(properties, dict) and isinstance(properties.get(property_name), dict):
                property_schemas.append(properties[property_name])
        if property_schemas:
            merged_properties[property_name] = _codex_merge_alternatives(property_schemas)

    required_intersection: list[str] | None = None
    for option in options:
        required = [value for value in option.get("required", []) if isinstance(value, str)] if isinstance(option.get("required"), list) else []
        if required_intersection is None:
            required_intersection = list(required)
        else:
            required_set = set(required)
            required_intersection = [key for key in required_intersection if key in required_set]

    merged: dict[str, Any] = {"type": "object", "properties": merged_properties}
    if required_intersection:
        merged["required"] = required_intersection
    explicit_additional = [option.get("additionalProperties") for option in options if isinstance(option.get("additionalProperties"), bool)]
    if explicit_additional:
        merged["additionalProperties"] = all(explicit_additional)
    return merged


def _codex_merge_alternatives(options: list[dict[str, Any]]) -> dict[str, Any]:
    if not options:
        return {}
    if len(options) == 1:
        return options[0]

    unique_options = _unique_json_values(options)
    if len(unique_options) == 1 and isinstance(unique_options[0], dict):
        return unique_options[0]

    const_values = [option.get("const") for option in options if "const" in option]
    if len(const_values) == len(options):
        enum_values = _unique_json_values(const_values)
        return _codex_with_type_list({"enum": enum_values}, _codex_enum_types(enum_values))

    enum_lists = [option.get("enum") for option in options]
    if enum_lists and all(isinstance(enum_values, list) for enum_values in enum_lists):
        merged_enum = _unique_json_values([value for enum_values in enum_lists for value in enum_values])
        return _codex_with_type_list({"enum": merged_enum}, _codex_enum_types(merged_enum))

    object_options = [option for option in options if option.get("type") == "object" or isinstance(option.get("properties"), dict)]
    if len(object_options) == len(options):
        return _codex_merge_object_alternatives(object_options)

    option_types = [_codex_schema_types(option) for option in options]
    if option_types and all(types for types in option_types):
        merged_types = _unique_json_values([schema_type for types in option_types for schema_type in types])
        non_null_types = [schema_type for schema_type in merged_types if schema_type != "null"]
        if "null" in merged_types and len(non_null_types) == 1:
            primary_type = non_null_types[0]
            primary_options = [option for option, types in zip(options, option_types) if primary_type in types]
            if len(primary_options) == 1:
                return _codex_with_type_list(dict(primary_options[0]), [primary_type, "null"])
            if primary_type == "object":
                return _codex_with_type_list(_codex_merge_object_alternatives(primary_options), ["object", "null"])
            return _codex_with_type_list(dict(primary_options[0]), [primary_type, "null"])
        return _codex_with_type_list({}, merged_types)

    return {}


def _codex_normalize_for_codex(schema: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    alternatives: list[dict[str, Any]] = []
    for key, value in schema.items():
        if key == "$schema":
            continue
        if key in {"anyOf", "oneOf", "allOf"} and isinstance(value, list):
            alternatives.extend(_codex_normalize_for_codex(entry) for entry in value if isinstance(entry, dict))
            continue
        base[key] = _codex_transform_child_value(value, key)
    if not alternatives:
        return base
    return _codex_merge_with_fallback(base, _codex_merge_alternatives(alternatives))


def _codex_ensure_root_object(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "object" and "anyOf" not in schema and "oneOf" not in schema and "allOf" not in schema:
        return schema
    return {
        "type": "object",
        "properties": {"result": schema},
        "required": ["result"],
        "additionalProperties": False,
    }


def _unique_json_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        try:
            key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        except TypeError:
            key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _codex_system_with_json_schema_instruction(system: str, schema: dict[str, Any]) -> str:
    serialized_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return (
        f"{system}\n\n"
        "CODEX JSON OUTPUT CONTRACT\n"
        "Return exactly one JSON object and no markdown, commentary, or surrounding text. "
        "The JSON object must conform to this JSON Schema:\n"
        f"{serialized_schema}"
    )


def _codex_structured_response(response: CodexResponse) -> dict[str, Any]:
    content = _strip_json_markdown_fence(_message_text(response))
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {"raw": response, "parsed": None, "parsing_error": exc}
    return {"raw": response, "parsed": parsed, "parsing_error": None}


def _usage_metadata(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0),
        "output_tokens": int(
            usage.get("output_tokens") or token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
        ),
    }


def _response_model_name(response: Any) -> str | None:
    response_metadata = getattr(response, "response_metadata", None) or {}
    value = response_metadata.get("model_name") or response_metadata.get("model")
    return str(value) if value else None
