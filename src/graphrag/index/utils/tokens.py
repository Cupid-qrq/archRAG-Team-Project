# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Utilities for working with tokens."""

from __future__ import annotations

import logging
from typing import Protocol

import tiktoken

DEFAULT_ENCODING_NAME = "cl100k_base"
log = logging.getLogger(__name__)


class TokenEncoder(Protocol):
    def encode(self, text: str, *args, **kwargs) -> list[int]:
        ...

    def decode(self, tokens: list[int], *args, **kwargs) -> str:
        ...


class OfflineEncoding:
    """Fallback encoder that stays fully local and reversible."""

    def encode(self, text: str, *args, **kwargs) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return [ord(char) for char in text]

    def decode(self, tokens: list[int], *args, **kwargs) -> str:
        return "".join(chr(int(token)) for token in tokens)


_encoders: dict[str, TokenEncoder] = {}


def get_token_encoder(
    model: str | None = None, encoding_name: str | None = None
) -> TokenEncoder:
    """Get a token encoder, falling back to a local reversible encoder."""
    key = model or encoding_name or DEFAULT_ENCODING_NAME
    enc = _encoders.get(key)
    if enc is not None:
        return enc

    try:
        if model is not None:
            enc = tiktoken.encoding_for_model(model)
        else:
            enc = tiktoken.get_encoding(encoding_name or DEFAULT_ENCODING_NAME)
    except Exception:
        log.warning("Falling back to local token encoder for %s", key)
        enc = OfflineEncoding()

    _encoders[key] = enc
    return enc


def num_tokens_from_string(
    string: str, model: str | None = None, encoding_name: str | None = None
) -> int:
    """Return the number of tokens in a text string."""
    encoding = get_token_encoder(model=model, encoding_name=encoding_name)
    return len(encoding.encode(string))


def string_from_tokens(
    tokens: list[int], model: str | None = None, encoding_name: str | None = None
) -> str:
    """Return a text string from a list of tokens."""
    encoding = get_token_encoder(model=model, encoding_name=encoding_name)
    return encoding.decode(tokens)
