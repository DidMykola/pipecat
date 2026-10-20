#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A simple backend for the ``LLMWithBackend`` examples: the everyday tool-calling case.

The backend holds the two tools many of the examples use, with no long job
behind them:

- ``get_current_weather`` answers at once.
- ``get_restaurant_recommendation`` takes a few seconds, long enough for the
  user to ask for something else while it runs.

This is the work a single LLM holding the tools would do in one round trip,
so the example shows what the pair adds and costs there: how soon the user
hears the answer, and what the frontend says in the meantime. The
engineering backend in ``backend.py`` is the one for work that takes a while.

A frontend example builds this backend with :func:`build_backend` and gives
its own model :data:`FRONTEND_INSTRUCTIONS`.
"""

import asyncio
import os
from datetime import datetime

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.workers.llm import BackendLLMWorker

FRONTEND_INSTRUCTIONS = """You are a friendly voice assistant. Your responses are spoken aloud,
so keep them to one or two natural sentences without any formatting. You can look up the
current weather and recommend restaurants; a lookup takes a moment."""

BACKEND_INSTRUCTIONS = """You look things up for a voice assistant: the current weather and
restaurant recommendations, with your tools. Give what you find in a sentence or two of plain
spoken text, and never give a result that a tool did not return."""

#: How long a restaurant recommendation takes, checking which tables are open:
#: long enough for the user to ask for something else while it runs.
RESTAURANT_LOOKUP_SECS = 4


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = 75 if format == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


@tool_options(cancel_on_interruption=False, cancellable_by_llm=True)
async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation, checking which tables are open. Takes a few seconds.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await asyncio.sleep(RESTAURANT_LOOKUP_SECS)
    await params.result_callback({"name": "The Golden Dragon"})


def build_backend() -> BackendLLMWorker:
    """Build the simple backend worker."""
    return BackendLLMWorker(
        llm=AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                system_instruction=BACKEND_INSTRUCTIONS,
                # Thinking summaries reach the frontend as silent "Backend (thinking):"
                # messages, so it can say how the work is going if asked. The budget
                # keeps a step to seconds: left to decide for itself, the model can
                # think for a while over one lookup.
                thinking=AnthropicLLMService.ThinkingConfig(
                    type="enabled", budget_tokens=2048, display="summarized"
                ),
                # Thinking counts against max_tokens; the budget plus the reply must fit.
                max_tokens=16384,
            ),
        ),
        context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
    )
