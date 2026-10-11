"""Paper library and workspace persistence.

A *workspace* is a plain JSON-serialisable dict:
    {"version": 1, "papers": {paper_id: metadata}, "notes": [note, ...], "topics": {topic_id: topic}}
PDF bytes are kept separately by a store.

Stores
------
* ``SessionStore`` (default): everything lives in this browser session's server memory. Nothing is written to
  disk and nothing is shared with other visitors. It is lost when the session ends (tab closed, app restarted).
* ``LocalStore`` (opt-in, ``READRAY_STORAGE=local``): JSON + PDFs in a directory on the machine running the app.
  Intended for single-user local use. Do NOT enable it on a public deployment: every visitor would share it, and
  Streamlit Community Cloud's filesystem is ephemeral anyway.

The workspace backup (export/import JSON) is the portable way to keep work across hosted sessions.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from pathlib import Path

from .utils import ValidationError, now_iso, validate_pdf_upload

WORKSPACE_VERSION = 1
MAX_BACKUP_MB = 200
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def empty_workspace() -> dict:
    return {"version": WORKSPACE_VERSION, "papers": {}, "notes": [], "topics": {}}


# --------------------------------------------------------------------------- stores


class SessionStore:
    """Holds the workspace and PDFs in a dict (normally ``st.session_state``)."""

    kind = "session"

    def __init__(self, state: dict):
        self.state = state
        self.state.setdefault("_rr_workspace", empty_workspace())
        self.state.setdefault("_rr_pdfs", {})

    def load(self) -> dict:
        return self.state["_rr_workspace"]

    def save(self, workspace: dict) -> None:
        self.state["_rr_workspace"] = workspace

    def get_pdf(self, paper_id: str) -> bytes | None:
        return self.state["_rr_pdfs"].get(paper_id)

    def put_pdf(self, paper_id: str, data: bytes) -> None:
        self.state["_rr_pdfs"][paper_id] = data

    def delete_pdf(self, paper_id: str) -> None:
        self.state["_rr_pdfs"].pop(paper_id, None)


class LocalStore:
    """Persists the workspace to ``<root>/workspace.json`` and PDFs to ``<root>/pdfs/<id>.pdf``."""

    kind = "local"

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        (self.root / "pdfs").mkdir(parents=True, exist_ok=True)
        self._cache: dict | None = None

    @property
    def path(self) -> Path:
        return self.root / "workspace.json"

    def load(self) -> dict:
        if self._cache is None:
            try:
                self._cache = validate_workspace(json.loads(self.path.read_text(encoding="utf-8")))
            except FileNotFoundError:
                self._cache = empty_workspace()
            except (ValueError, ValidationError):
                # Keep the unreadable file for manual recovery instead of overwriting it.
                self.path.replace(self.path.with_suffix(f".corrupt-{now_iso().replace(':', '')}.json"))
                self._cache = empty_workspace()
        return self._cache

    def save(self, workspace: dict) -> None:
        self._cache = workspace
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(workspace, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)  # atomic on the same filesystem

    def _pdf_path(self, paper_id: str) -> Path:
        if not _ID_RE.match(paper_id):
            raise ValueError("Invalid paper id.")
        return self.root / "pdfs" / f"{paper_id}.pdf"

    def get_pdf(self, paper_id: str) -> bytes | None:
        try:
            return self._pdf_path(paper_id).read_bytes()
        except (FileNotFoundError, ValueError):
            return None

    def put_pdf(self, paper_id: str, data: bytes) -> None:
        self._pdf_path(paper_id).write_bytes(data)

    def delete_pdf(self, paper_id: str) -> None:
        try:
            self._pdf_path(paper_id).unlink()
        except (FileNotFoundError, ValueError):
            pass


# --------------------------------------------------------------------------- papers


def paper_metadata(paper, size_bytes: int) -> dict:
    """Metadata record for a processed PaperDocument."""
    return {
        "id": paper.doc_id,
        "filename": paper.filename,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "abstract": paper.abstract,
        "page_count": paper.page_count,
        "size_bytes": size_bytes,
        "scanned_pages": len(paper.scanned_pages),
        "added_at": now_iso(),
        "last_opened": now_iso(),
        "last_page": 1,
        "metadata_source": "auto-detected",
    }


def add_paper(workspace: dict, metadata: dict) -> tuple[dict, bool]:
    """Add or refresh a paper. Returns (record, is_new). Manual edits to an existing record are kept."""
    existing = workspace["papers"].get(metadata["id"])
    if existing:
        existing["last_opened"] = now_iso()
        return existing, False
    workspace["papers"][metadata["id"]] = metadata
    return metadata, True


def update_metadata(workspace: dict, paper_id: str, **fields) -> dict:
    record = workspace["papers"][paper_id]
    for key in ("title", "authors", "year"):
        if key in fields and fields[key] is not None:
            value = " ".join(str(fields[key]).split())[:500]
            if key == "title" and not value:
                raise ValidationError("Title cannot be empty.")
            if record.get(key) != value:
                record[key] = value
                record["metadata_source"] = "edited by user"
    # keep note and literature-review titles in sync
    for note in workspace["notes"]:
        if note.get("paper_id") == paper_id:
            note["paper_title"] = record["title"]
    for topic in workspace["topics"].values():
        row = topic.get("rows", {}).get(paper_id)
        if row:
            for key in ("title", "authors", "year"):
                cell = row.get(key)
                if cell is not None and cell.get("source") in ("metadata", ""):
                    cell["value"] = record.get(key, "")
    return record


def remove_paper(workspace: dict, store, paper_id: str, delete_notes: bool = False) -> None:
    workspace["papers"].pop(paper_id, None)
    store.delete_pdf(paper_id)
    if delete_notes:
        workspace["notes"] = [n for n in workspace["notes"] if n.get("paper_id") != paper_id]
    for topic in workspace["topics"].values():
        if paper_id in topic.get("paper_ids", []):
            topic["paper_ids"].remove(paper_id)
        topic.get("rows", {}).pop(paper_id, None)


def search_papers(workspace: dict, query: str = "") -> list[dict]:
    query = (query or "").strip().lower()
    papers = list(workspace["papers"].values())
    if query:
        papers = [
            p for p in papers
            if all(t in f"{p.get('title', '')} {p.get('authors', '')} {p.get('filename', '')}".lower() for t in query.split())
        ]
    return sorted(papers, key=lambda p: p.get("last_opened", ""), reverse=True)


# --------------------------------------------------------------------------- backup


def validate_workspace(data) -> dict:
    if not isinstance(data, dict):
        raise ValidationError("Backup is not a Readray workspace.")
    if data.get("version") != WORKSPACE_VERSION:
        raise ValidationError("Unsupported workspace version.")
    papers, notes, topics = data.get("papers"), data.get("notes"), data.get("topics")
    if not isinstance(papers, dict) or not isinstance(notes, list) or not isinstance(topics, dict):
        raise ValidationError("Backup is missing papers, notes or topics.")
    for pid, record in papers.items():
        if not _ID_RE.match(str(pid)) or not isinstance(record, dict) or record.get("id") != pid:
            raise ValidationError("Backup contains an invalid paper record.")
    for note in notes:
        if not isinstance(note, dict) or not isinstance(note.get("content", ""), str):
            raise ValidationError("Backup contains an invalid note.")
    for topic in topics.values():
        if not isinstance(topic, dict) or not isinstance(topic.get("paper_ids", []), list):
            raise ValidationError("Backup contains an invalid literature-review topic.")
    return data


def export_backup(workspace: dict, pdfs: dict[str, bytes] | None = None) -> bytes:
    payload = {"readray_backup": True, "exported_at": now_iso(), "workspace": workspace}
    if pdfs:
        payload["pdfs"] = {pid: base64.b64encode(data).decode() for pid, data in pdfs.items()}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def import_backup(raw: bytes) -> tuple[dict, dict[str, bytes]]:
    if len(raw) > MAX_BACKUP_MB * 1024 * 1024:
        raise ValidationError(f"Backup files must be smaller than {MAX_BACKUP_MB} MB.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValidationError("This file is not a valid Readray backup (invalid JSON).") from None
    if not isinstance(payload, dict) or not payload.get("readray_backup"):
        raise ValidationError("This file is not a Readray backup.")
    workspace = validate_workspace(payload.get("workspace"))
    pdfs: dict[str, bytes] = {}
    for pid, b64 in (payload.get("pdfs") or {}).items():
        if pid not in workspace["papers"]:
            continue
        try:
            pdfs[pid] = validate_pdf_upload(base64.b64decode(b64, validate=True), max_mb=MAX_BACKUP_MB)
        except (ValueError, ValidationError):
            continue
    return workspace, pdfs


def merge_workspaces(current: dict, incoming: dict) -> dict:
    """Merge an imported workspace into the current one without duplicating papers, notes or topics."""
    for pid, record in incoming["papers"].items():
        current["papers"].setdefault(pid, record)
    known_notes = {n.get("id") for n in current["notes"]}
    current["notes"].extend(n for n in incoming["notes"] if n.get("id") not in known_notes)
    for tid, topic in incoming["topics"].items():
        current["topics"].setdefault(tid, topic)
    return current
