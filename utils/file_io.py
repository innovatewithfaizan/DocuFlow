from __future__ import annotations
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocuFlowException
log = CustomLogger().get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Session ids become directory names under data/ and faiss_index/, so only allow
# characters that cannot escape those directories (no "/", "\" or "..").
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

def _session_id(prefix: str = "session")-> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def _safe_name(name: str) -> str:
    stem, ext = Path(Path(name).name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "file"
    return f"{stem[:80]}{ext}"

def original_name(path: str) -> str:
    """Inverse of the naming in save_uploaded_files: '1a2b3c4d__report.pdf' -> 'report.pdf'."""
    base = Path(path).name
    prefix, sep, rest = base.partition("__")
    return rest if sep and len(prefix) == 8 else base

def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"Invalid session_id {session_id!r}: use 1-64 letters, digits, '_' or '-'")
    return session_id

def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for uf in uploaded_files:
            name = getattr(uf, "name", "file")
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported  file skipped", filename=name)
                continue
            # Unique prefix avoids collisions; the original name is kept so answers can cite it
            fname = f"{uuid.uuid4().hex[:8]}__{_safe_name(name)}"
            out = target_dir / fname
            with open(out, "wb") as f:
                if hasattr(uf, "read"):
                    f.write(uf.read())
                else:
                    f.write(uf.getbuffer())
            saved.append(out)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))
        return saved
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocuFlowException("Failed to save uploaded files", e) from e
        