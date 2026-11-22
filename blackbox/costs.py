"""Cost tracking per model.

List prices (USD per 1M tokens, input/output) as published by the
providers, checked 2026-11-22. Prices change often - and some of
these models are already being retired - so treat the table as a
starting estimate, not a bill: override a price with
register_pricing(), or price a run yourself with cost_for().

Unknown models cost None, never a guessed number. A model_call
recorded with a plain int token count (no input/output split) is
counted in the per-model "tokens" bucket but not priced, because
there is no honest way to split it - pass tokens as
{"input": n, "output": m} to get a cost.
"""

import re

# USD per 1M tokens: (input, output). Checked 2026-11-22 against the
# providers' published pricing pages.
_PRICING = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-opus-4-5": (5.00, 25.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-2-5-pro": (1.25, 10.00),
    "gemini-2-5-flash": (0.30, 2.50),
    "gemini-2-5-flash-lite": (0.10, 0.40),
}

_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def _normalize(model):
    """Canonical form for lookup: lowercase, dots to dashes, and a
    trailing snapshot date stripped, so "gpt-5-2025-08-07" and
    "claude-sonnet-4.5" match the table entries "gpt-5" and
    "claude-sonnet-4-5"."""
    name = str(model).strip().lower().replace(".", "-")
    return _DATE_SUFFIX.sub("", name)


def price_for(model):
    """(input, output) USD per 1M tokens for a model, or None if the
    model is not in the table."""
    return _PRICING.get(_normalize(model))


def cost_for(model, input_tokens, output_tokens):
    """USD cost of a call, or None if the model has no known price.
    Token counts of None count as 0."""
    price = price_for(model)
    if price is None:
        return None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must not be negative")
    input_rate, output_rate = price
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return round(cost, 6)


def register_pricing(model, input_per_million, output_per_million):
    """Add or override a model's price (USD per 1M tokens). Use this
    when the provider changes a rate, or for a model the table does
    not cover."""
    if input_per_million < 0 or output_per_million < 0:
        raise ValueError("prices must not be negative")
    _PRICING[_normalize(model)] = (input_per_million, output_per_million)
