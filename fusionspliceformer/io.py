"""Small file utilities used by prediction and input preparation."""
from pathlib import Path
import hashlib
import json
import os
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.j_' + uuid.uuid4().hex[:12] + '.tmp')
    try:
        with temporary.open('w',encoding='utf-8') as handle:
            json.dump(payload,handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
