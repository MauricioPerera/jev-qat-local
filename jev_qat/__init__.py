"""Jev-Omni-shaped decision adapter for the local Gemma E4B QAT model."""

from .adapter import JevQatAdapter, load_jev_omni_qat
from .spanish import classify_spanish_message

__all__ = ["JevQatAdapter", "load_jev_omni_qat", "classify_spanish_message"]
