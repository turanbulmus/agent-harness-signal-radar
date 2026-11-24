"""Strict-schema BaseTool wrapper enforcing JSON contracts and guided error recovery."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel, ValidationError
from typing_extensions import override

from ..observability.logging_config import get_logger
from ..observability.pii_redactor import SensitiveDataRedactor
from ..observability.tracing import start_span
from ..schemas import GuidedRecoveryAdvice

InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)

_logger = get_logger("harness_signal_radar.tools")


class StrictSchemaFunctionTool(BaseTool):
  """ADK BaseTool enforcing strict Pydantic input/output JSON schemas and guided recovery.

  Exposes explicit ``parameters_json_schema`` and ``response_json_schema`` in the
  Gemini ``FunctionDeclaration``, validates every invocation at runtime, catches
  validation and transient execution exceptions, and returns structured recovery
  instructions back to the LLM instead of crashing the agent turn.
  """

  def __init__(
      self,
      *,
      func: Callable[..., Any],
      input_schema: type[InputModelT],
      output_schema: type[OutputModelT],
      redactor: SensitiveDataRedactor | None = None,
      name: str | None = None,
  ) -> None:
    doc = inspect.getdoc(func)
    if not doc:
      raise ValueError(
          f"Tool callable '{func.__name__}' must define a comprehensive docstring."
      )
    tool_name = name or func.__name__
    super().__init__(name=tool_name, description=doc)
    self.func = func
    self.input_schema = input_schema
    self.output_schema = output_schema
    self.redactor = redactor or SensitiveDataRedactor()
    self._accepts_tool_context = "tool_context" in inspect.signature(func).parameters

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema=self.input_schema.model_json_schema(),
        response_json_schema=self.output_schema.model_json_schema(),
    )

  def _build_guided_error_response(
      self,
      *,
      error_code: str,
      error_message: str,
      recovery_instructions: str,
      suggested_parameters: dict[str, Any] | None = None,
      retryable: bool = True,
  ) -> dict[str, Any]:
    advice = GuidedRecoveryAdvice(
        error_code=error_code,
        error_message=self.redactor.scrub_text(error_message),
        recovery_instructions=recovery_instructions,
        suggested_parameters=suggested_parameters or {},
        retryable=retryable,
    )
    payload = self.output_schema.model_validate(
        {
            "status": "error",
            "recovery_advice": advice.model_dump(mode="json"),
        }
    )
    return payload.model_dump(mode="json")

  @staticmethod
  def _example_valid_parameters(schema_cls: type[BaseModel]) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for field_name, field_info in schema_cls.model_fields.items():
      if field_info.examples:
        examples[field_name] = field_info.examples[0]
      elif field_info.default is not None and repr(field_info.default) != "PydanticUndefined":
        examples[field_name] = field_info.default
    return examples

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> dict[str, Any]:
    with start_span(
        f"tool.{self.name}",
        attributes={"tool.name": self.name},
    ):
      try:
        validated_input = self.input_schema.model_validate(args)
      except ValidationError as exc:
        field_errors = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc']) or 'input'}: {err['msg']}"
            for err in exc.errors()
        )
        _logger.warning(
            "tool_input_validation_failed",
            tool_name=self.name,
            validation_errors=field_errors,
        )
        return self._build_guided_error_response(
            error_code="INVALID_TOOL_ARGUMENTS",
            error_message=f"Arguments for '{self.name}' failed schema validation: {field_errors}",
            recovery_instructions=(
                f"Correct the invalid parameter(s) according to {self.input_schema.__name__} "
                "constraints (check types, allowed ranges, and remove unexpected extra keys) "
                f"and call '{self.name}' again."
            ),
            suggested_parameters=self._example_valid_parameters(self.input_schema),
            retryable=True,
        )

      try:
        if self._accepts_tool_context:
          raw_result = self.func(validated_input, tool_context=tool_context)
        else:
          raw_result = self.func(validated_input)

        if inspect.isawaitable(raw_result):
          raw_result = await raw_result

        if isinstance(raw_result, BaseModel):
          validated_output = self.output_schema.model_validate(
              raw_result.model_dump(mode="python")
          )
        else:
          validated_output = self.output_schema.model_validate(raw_result)

        return self.redactor.scrub_payload(validated_output.model_dump(mode="json"))
      except Exception as exc:
        scrubbed_error = self.redactor.scrub_text(str(exc))
        _logger.warning(
            "tool_execution_failed_with_guided_recovery",
            tool_name=self.name,
            error_type=type(exc).__name__,
            error_message=scrubbed_error,
        )
        return self._build_guided_error_response(
            error_code="TOOL_EXECUTION_ERROR",
            error_message=f"{type(exc).__name__}: {scrubbed_error}",
            recovery_instructions=(
                f"Tool '{self.name}' encountered a runtime error. Verify query scope, "
                "reduce batch size or lookback_days if rate-limited, or try an "
                "alternative source stream tool and disclose partial coverage."
            ),
            suggested_parameters=self._example_valid_parameters(self.input_schema),
            retryable=True,
        )
