"""Automated, point-in-time research data ingestion.

Collectors in this package are deliberately isolated from the live trading
engine.  They preserve immutable raw responses and upsert normalized records
into a separate SQLite research store under ``data_lake/``.
"""

from .storage import DataStore

__all__ = ["DataStore"]
