"""Symmetric encryption for adapter weights at rest. SPEC.md §8."""

from __future__ import annotations

import pytest

from twin.core.encryption import InvalidToken, decrypt_bytes, encrypt_bytes, generate_key


def test_round_trips() -> None:
    key = generate_key()
    plaintext = b"lora adapter weights, or close enough for a test"
    assert decrypt_bytes(encrypt_bytes(plaintext, key), key) == plaintext


def test_ciphertext_does_not_contain_the_plaintext() -> None:
    key = generate_key()
    plaintext = b"a very distinctive marker string 12345"
    ciphertext = encrypt_bytes(plaintext, key)
    assert plaintext not in ciphertext


def test_wrong_key_fails_loudly_rather_than_returning_garbage() -> None:
    ciphertext = encrypt_bytes(b"secret", generate_key())
    with pytest.raises(InvalidToken):
        decrypt_bytes(ciphertext, generate_key())


def test_generate_key_produces_distinct_keys() -> None:
    assert generate_key() != generate_key()
