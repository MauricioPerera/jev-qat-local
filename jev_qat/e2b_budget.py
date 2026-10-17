"""Fail-closed context accounting using the loaded E2B multimodal tokenizer."""
import httpx

from .adapter import BackendError, JevQatAdapter

CONTEXT_LIMIT = 4096
COUNT_ROUTE = "/v1/chat/completions/input_tokens"


class ContextBudgetError(ValueError):
    def __init__(self, budget):
        self.detail = {
            "code": "context_limit_exceeded",
            "message": (f"E2B request needs {budget['total_tokens']} tokens "
                        f"({budget['input_tokens']} input + {budget['reserved_output_tokens']} output); "
                        f"context limit is {budget['context_limit_tokens']}. "
                        "Shorten the text/options or reduce the supplied media/video_frames."),
            **budget,
        }
        super().__init__(self.detail["message"])


def positive_integer(value, field):
    if type(value) is not int or value <= 0:
        raise BackendError(f"E2B context budget cannot be verified: invalid {field}")
    return value


def measure_context_budget(adapter, payload):
    """Count the exact completion payload, after all prompt/media preparation.

    llama.cpp's counting route uses placeholder media chunks of the same token
    size as inference. It decodes media, but never evaluates encoder/LLM weights.
    There is deliberately no text-only estimate or fallback when counting fails.
    """
    configured = positive_integer(adapter.config.get("context_tokens"), "configured context")
    try:
        response = adapter.client.get("/props")
        response.raise_for_status()
        actual = response.json()["default_generation_settings"]["n_ctx"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        raise BackendError("E2B context budget cannot be verified: backend context unavailable") from error
    actual = positive_integer(actual, "backend context")
    output = positive_integer(payload.get("max_tokens"), "reserved output tokens")
    # Call the base transport, avoiding the E2B completion interception.
    counted = JevQatAdapter._post(adapter, COUNT_ROUTE, payload)
    if not isinstance(counted, dict):
        raise BackendError("E2B context budget cannot be verified: invalid token count response")
    inputs = positive_integer(counted.get("input_tokens"), "input token count")
    limit = min(CONTEXT_LIMIT, configured, actual)
    budget = {
        "input_tokens": inputs, "reserved_output_tokens": output,
        "total_tokens": inputs + output, "context_limit_tokens": limit,
        "remaining_tokens": limit - inputs - output,
        "counter": COUNT_ROUTE, "includes_multimodal": True,
        "checked_before_inference": True,
    }
    return budget


def check_context_budget(adapter, payload):
    budget = measure_context_budget(adapter, payload)
    if budget["total_tokens"] > budget["context_limit_tokens"]:
        raise ContextBudgetError(budget)
    return budget
