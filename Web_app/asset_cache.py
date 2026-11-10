"""Prepare the held-out TIFFs in the app server's disk cache."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile
import zlib


def extract_test_tiles(archive_path, cache_root):
    """Extract test TIFFs once, without overwriting conflicting existing files.

    The Hugging Face snapshot path is revision-specific, so a changed archive gets
    a different directory. A completion marker is written only after extraction.
    """
    archive_path = Path(archive_path).resolve()
    key = hashlib.sha256(str(archive_path).encode("utf-8")).hexdigest()[:16]
    root = Path(cache_root).resolve() / ("test_" + key)
    marker = root / ".complete"
    if marker.is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = []
        seen = set()
        for info in archive.infolist():
            parts = PurePosixPath(info.filename.replace("\\", "/")).parts
            for i, name in enumerate(parts):
                if name not in ("BigEarthNet-S1", "BigEarthNet-S2") or parts[i + 1:i + 2] != ("test",):
                    continue
                relative = PurePosixPath(*parts[i:])
                if ".." in relative.parts or relative.is_absolute():
                    raise ValueError("Unsafe path in dataset ZIP")
                if info.is_dir():
                    break
                if len(relative.parts) != 3 or relative.suffix.lower() not in (".tif", ".tiff"):
                    break
                if relative in seen:
                    raise ValueError("Duplicate test tile in dataset ZIP: " + str(relative))
                seen.add(relative)
                members.append((info, relative))
                break
        modalities = {relative.parts[0] for _, relative in members}
        if modalities != {"BigEarthNet-S1", "BigEarthNet-S2"}:
            raise ValueError("Dataset ZIP must contain test TIFFs for both S1 and S2")
        for info, relative in members:
            dest = root.joinpath(*relative.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                crc = 0
                with dest.open("rb") as existing:
                    for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if dest.stat().st_size != info.file_size or crc != info.CRC:
                    raise ValueError("Conflicting cached tile; select a new cache directory: " + str(dest))
                continue
            try:
                handle = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                raise RuntimeError("Another extraction is writing this cache; retry startup.")
            try:
                with os.fdopen(handle, "wb") as out, archive.open(info) as source:
                    shutil.copyfileobj(source, out)
            except Exception:
                dest.unlink(missing_ok=True)
                raise
    marker.touch()
    return root
