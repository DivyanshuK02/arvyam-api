# app/memory/__init__.py
"""
ARVYAM Memory Package — Phase 3.1

This package provides memory context building and post-selection reranking
for personalized user experiences while maintaining selection invariance.

Constitutional Rails:
- Selection Invariance: Memory CANNOT change SKU IDs, only ordering
- Post-Selection Only: Memory applied AFTER engine produces 3 SKUs
- Bounded Weights: Rerank weights capped at <0.5
- Deterministic: Same input + same context → same output
- Privacy-Safe: No raw prompts; only SKU IDs and frozen enums

Feature Flags (controlled in app/db.py):
- MEMORY_CONTEXT_ENABLED: Build memory context from user history
- MEMORY_RERANK_ENABLED: Apply post-selection reranking

Submodules:
- context: Build read-only memory context from orders + profiles
- rerank: Deterministic post-selection reranking with bounded weights
"""

from __future__ import annotations

__all__ = [
    # Context
    "build_context",
    "MemoryContext",
    # Rerank
    "rerank",
    "RerankedTriad",
]


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import pattern for clean module loading."""
    
    # Context
    if name in ("build_context", "MemoryContext"):
        from . import context
        return getattr(context, name)
    
    # Rerank
    if name in ("rerank", "RerankedTriad"):
        from . import rerank
        return getattr(rerank, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
