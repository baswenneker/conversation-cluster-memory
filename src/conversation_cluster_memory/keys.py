"""Conversation keys: which conversation a memory belongs to."""

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsKeyParts(Protocol):
    """An application's own key type: anything that yields an ordered tuple of strings."""

    def key_parts(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class ConversationKey:
    """A conversation id, optionally scoped by user and tenant."""

    conversation: str
    user: str | None = None
    tenant: str | None = None

    def key_parts(self) -> tuple[str, ...]:
        # Field names are part of each item, so keys that differ only in an optional field
        # (unset vs. set, even to "") never produce the same tuple.
        fields = {"conversation": self.conversation, "user": self.user, "tenant": self.tenant}
        return tuple(f"{name}={value}" for name, value in fields.items() if value is not None)


type KeyLike = str | SupportsKeyParts


def key_parts(key: KeyLike) -> tuple[str, ...]:
    """The ordered tuple of strings that identifies a conversation."""
    if isinstance(key, str):
        key = ConversationKey(key)
    if not isinstance(key, SupportsKeyParts):
        raise TypeError(f"a conversation key is a str or has key_parts(); got {type(key)!r}")
    parts = key.key_parts()
    if not parts or not all(isinstance(part, str) for part in parts):
        raise ValueError(f"key_parts() must return a non-empty tuple of strings; got {parts!r}")
    return tuple(parts)


def storage_id(key: KeyLike) -> str:
    """The identity under which a conversation's records are stored."""
    return json.dumps(key_parts(key))


def collection_name(storage: str) -> str:
    """A Chroma collection name for the conversation with this `storage_id` (Chroma restricts
    names to [a-zA-Z0-9._-])."""
    return "c" + hashlib.sha256(storage.encode()).hexdigest()[:40]
