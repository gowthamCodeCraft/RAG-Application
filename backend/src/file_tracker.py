import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional

class FileTracker:
    """
    Tracks which files have been indexed and their associated vector IDs.
    Enables incremental uploads: only new/changed files are processed.
    """

    def __init__(self, manifest_path: str = "data/indexed_files.json"):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, dict] = self._load()

    def _load(self) -> Dict:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file contents for change detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def is_indexed(self, file_path: str) -> bool:
        """Check if file exists in manifest with matching hash (unchanged)."""
        filename = Path(file_path).name
        file_hash = self.compute_hash(file_path)
        entry = self._data.get(filename)
        return entry is not None and entry.get("hash") == file_hash

    def has_file(self, filename: str) -> bool:
        return filename in self._data

    def get_chunk_ids(self, filename: str) -> List[str]:
        return self._data.get(filename, {}).get("chunk_ids", [])

    def get_file_hash(self, filename: str) -> Optional[str]:
        return self._data.get(filename, {}).get("hash")

    def register(self, file_path: str, chunk_ids: List[str]):
        """Register a file as indexed with its vector IDs."""
        filename = Path(file_path).name
        self._data[filename] = {
            "hash": self.compute_hash(file_path),
            "chunk_ids": chunk_ids,
            "status": "indexed",
            "vector_count": len(chunk_ids),
        }
        self._save()
        print(f"[TRACKER] Registered {filename} with {len(chunk_ids)} vectors")

    def unregister(self, filename: str):
        """Remove file from tracking."""
        if filename in self._data:
            del self._data[filename]
            self._save()
            print(f"[TRACKER] Unregistered {filename}")

    def get_all_filenames(self) -> Set[str]:
        return set(self._data.keys())

    def get_manifest(self) -> Dict:
        return self._data.copy()