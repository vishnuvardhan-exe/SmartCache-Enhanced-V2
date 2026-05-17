"""
Storage backend for cache entries.
Supports LMDB (default) and Redis (production).

Enhancements:
- Phase 3: Cost-Aware LRU eviction (layer_size / reload_cost ratio)
- Phase 4: Static Pinning of hot/critical entries
"""

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pickle
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached response."""
    prompt_hash: str
    response: Any
    embedding: np.ndarray
    created_at: float
    hit_count: int = 0
    token_count: int = 0
    model: str = ""
    temperature: float = 0.0
    # Phase 3: Cost-Aware LRU fields
    last_accessed: float = 0.0
    reload_cost: float = 1.0   # Relative cost to regenerate (token_count / 1000)
    layer_size: float = 1.0    # Relative size/weight of this entry
    # Phase 4: Static Pinning
    pinned: bool = False

    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = time.time()

    @property
    def eviction_score(self) -> float:
        """
        Cost-Aware LRU eviction score. Lower = evict first.
        Formula: (layer_size / reload_cost) * recency_weight
        """
        age = time.time() - self.last_accessed
        recency_weight = 1.0 / (1.0 + age / 3600.0)
        return (self.layer_size / max(self.reload_cost, 0.001)) * recency_weight

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if k != 'embedding'}


class StorageBackend(ABC):
    """Abstract storage backend."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]: pass

    @abstractmethod
    def set(self, key: str, entry: CacheEntry, embedding: np.ndarray): pass

    @abstractmethod
    def search_similar(self, embedding: np.ndarray, threshold: float = 0.92,
                       limit: int = 10) -> List[Tuple[str, CacheEntry, float]]: pass

    @abstractmethod
    def increment_hit(self, key: str): pass

    @abstractmethod
    def close(self): pass

    def pin(self, key: str): pass

    def unpin(self, key: str): pass

    def evict_lru_cost_aware(self, max_entries: int): pass


class LMDBStorage(StorageBackend):
    """LMDB-based storage with Cost-Aware LRU and static pinning."""

    def __init__(self, db_path: str = "./smartcache_data", max_db_size: int = 10 * 1024**3):
        try:
            import lmdb
        except ImportError:
            raise ImportError("lmdb package required. Install: pip install lmdb")

        self.db_path = db_path
        self.env = lmdb.open(db_path, map_size=max_db_size, max_dbs=3,
                             subdir=True, lock=True, readahead=True, meminit=True)
        self.db = self.env.open_db(b"cache_entries")
        self.index_db = self.env.open_db(b"embedding_index")
        self.meta_db = self.env.open_db(b"metadata")

    def _serialize(self, entry: CacheEntry) -> bytes:
        return pickle.dumps(entry)

    def _deserialize(self, data: bytes) -> CacheEntry:
        return pickle.loads(data)

    def get(self, key: str) -> Optional[CacheEntry]:
        with self.env.begin(db=self.db) as txn:
            data = txn.get(key.encode('utf-8'), db=self.db)
            if data:
                return self._deserialize(data)
        return None

    def set(self, key: str, entry: CacheEntry, embedding: np.ndarray):
        with self.env.begin(write=True, db=self.db) as txn:
            txn.put(key.encode('utf-8'), self._serialize(entry), db=self.db)
        with self.env.begin(write=True, db=self.index_db) as txn:
            txn.put(key.encode('utf-8'), embedding.tobytes(), db=self.index_db)

    def search_similar(self, query_embedding: np.ndarray, threshold: float = 0.92,
                       limit: int = 10) -> List[Tuple[str, CacheEntry, float]]:
        results = []
        with self.env.begin(db=self.index_db) as txn:
            cursor = txn.cursor(db=self.index_db)
            for key_bytes, embedding_bytes in cursor:
                key = key_bytes.decode('utf-8')
                stored = np.frombuffer(embedding_bytes, dtype=np.float32)
                sim = self._cosine_sim(query_embedding, stored)
                if sim >= threshold:
                    entry = self.get(key)
                    if entry:
                        results.append((key, entry, sim))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def increment_hit(self, key: str):
        entry = self.get(key)
        if entry:
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self.set(key, entry, entry.embedding)

    def pin(self, key: str):
        """Phase 4: Mark entry as permanently resident."""
        entry = self.get(key)
        if entry:
            entry.pinned = True
            self.set(key, entry, entry.embedding)
            logger.info(f"Pinned: {key}")

    def unpin(self, key: str):
        entry = self.get(key)
        if entry:
            entry.pinned = False
            self.set(key, entry, entry.embedding)

    def evict_lru_cost_aware(self, max_entries: int):
        """
        Phase 3: Cost-Aware LRU.
        Evicts entries with lowest (layer_size/reload_cost)*recency_weight score.
        Pinned entries are immune to eviction.
        """
        all_entries = self.all_entries()
        if len(all_entries) <= max_entries:
            return

        evictable = [(k, e) for k, e in all_entries if not e.pinned]
        to_remove = len(all_entries) - max_entries
        evictable.sort(key=lambda x: x[1].eviction_score)  # lowest score first

        for key, entry in evictable[:to_remove]:
            self._delete(key)
            logger.debug(f"Evicted {key} score={entry.eviction_score:.4f}")

        logger.info(f"Cost-Aware LRU: evicted {min(to_remove, len(evictable))} entries")

    def _delete(self, key: str):
        enc = key.encode('utf-8')
        with self.env.begin(write=True, db=self.db) as txn:
            txn.delete(enc, db=self.db)
        with self.env.begin(write=True, db=self.index_db) as txn:
            txn.delete(enc, db=self.index_db)

    def count(self) -> int:
        with self.env.begin(db=self.db) as txn:
            return txn.stat(db=self.db)['entries']

    def all_entries(self) -> List[Tuple[str, CacheEntry]]:
        results = []
        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor(db=self.db)
            for key_bytes, data in cursor:
                results.append((key_bytes.decode('utf-8'), self._deserialize(data)))
        return results

    def close(self):
        self.env.close()


class RedisStorage(StorageBackend):
    """Redis-based storage for production use."""

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, password: Optional[str] = None):
        try:
            import redis
        except ImportError:
            raise ImportError("redis package required. Install: pip install redis")

        self.client = redis.Redis(host=host, port=port, db=db,
                                  password=password, decode_responses=False)
        self.client.ping()

    def get(self, key: str) -> Optional[CacheEntry]:
        data = self.client.get(f"scache:entry:{key}")
        if data:
            return pickle.loads(data)
        return None

    def set(self, key: str, entry: CacheEntry, embedding: np.ndarray):
        ttl = 3600 * 24 * 7
        self.client.setex(f"scache:entry:{key}", ttl, pickle.dumps(entry))
        self.client.setex(f"scache:emb:{key}", ttl, embedding.tobytes())

    def search_similar(self, query_embedding: np.ndarray, threshold: float = 0.92,
                       limit: int = 10) -> List[Tuple[str, CacheEntry, float]]:
        logger.warning("Redis similarity search not implemented. Use LMDB.")
        return []

    def increment_hit(self, key: str):
        entry = self.get(key)
        if entry:
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self.set(key, entry, entry.embedding)

    def pin(self, key: str):
        entry = self.get(key)
        if entry:
            entry.pinned = True
            self.set(key, entry, entry.embedding)

    def evict_lru_cost_aware(self, max_entries: int):
        keys = [k.decode() for k in self.client.keys("scache:entry:*")]
        if len(keys) <= max_entries:
            return
        entries = []
        for fk in keys:
            data = self.client.get(fk)
            if data:
                e = pickle.loads(data)
                entries.append((fk.replace("scache:entry:", ""), e))
        evictable = [(k, e) for k, e in entries if not e.pinned]
        evictable.sort(key=lambda x: x[1].eviction_score)
        for key, _ in evictable[:len(entries) - max_entries]:
            self.client.delete(f"scache:entry:{key}", f"scache:emb:{key}")

    def close(self):
        self.client.close()


# Alias for backward compatibility
CacheStorage = StorageBackend
