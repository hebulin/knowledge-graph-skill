"""
KG Skill Document Sync Module - Keep the knowledge graph in sync with source files.

Two modes:
1. File watcher: monitors a directory for create/modify/delete events
2. Git hook: processes changed files from git diff

When a file is modified, its content is re-extracted and merged into the graph.
When a file is deleted, associated knowledge is marked as deprecated.

Usage (file watcher):
    python kg_sync.py watch --path /path/to/project --port 8700

Usage (git hook):
    python kg_sync.py git-diff --repo /path/to/repo --port 8700
"""

import os
import sys
import json
import argparse
import hashlib
import subprocess
from typing import Optional

import requests


class KGSyncClient:
    """Client to communicate with the KG Skill API server for sync operations."""

    def __init__(self, api_url: str = "http://localhost:8700",
                 api_key: str = None):
        """Initialize sync client with API URL and optional key."""
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or os.environ.get("KG_API_KEY", "")
        self.headers = {}
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key
        # Track file hashes to detect actual changes
        self._file_hashes = {}

    def sync_file(self, file_path: str, content: str = None) -> dict:
        """Extract knowledge from a file and sync to graph."""
        if content is None:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        # Check if content actually changed
        file_hash = hashlib.md5(content.encode()).hexdigest()
        prev_hash = self._file_hashes.get(file_path)
        if prev_hash == file_hash:
            return {"status": "unchanged", "file": file_path}
        self._file_hashes[file_path] = file_hash

        # Determine format
        ext = os.path.splitext(file_path)[1].lower()
        fmt_map = {".md": "markdown", ".txt": "text", ".json": "json",
                   ".csv": "table", ".html": "text"}
        fmt = fmt_map.get(ext, "text")

        # Send to extraction API
        resp = requests.post(
            f"{self.api_url}/api/v1/extract",
            headers=self.headers,
            json={
                "content": content,
                "format": fmt,
                "extraction_strategy": "auto",
                "auto_resolve": True,
                "title": os.path.basename(file_path),
            },
            timeout=120,
        )
        result = resp.json()
        result["file"] = file_path
        result["file_hash"] = file_hash
        return result

    def handle_deletion(self, file_path: str) -> dict:
        """Mark knowledge associated with a deleted file as deprecated.

        Finds all chunks and entities whose provenance references the deleted
        file, and marks them as deprecated.
        """
        # The file path is used as the document title during extraction
        # We need to find documents with matching title and mark their knowledge
        filename = os.path.basename(file_path)

        # Search for entities that might be from this file
        resp = requests.post(
            f"{self.api_url}/api/v1/search/entities",
            headers=self.headers,
            json={"query": filename, "top_k": 50},
            timeout=30,
        )
        results = resp.json().get("results", [])

        deprecated_count = 0
        for entity in results:
            ent_id = entity.get("entity_id")
            if ent_id:
                # Mark entity as deprecated via update
                try:
                    requests.patch(
                        f"{self.api_url}/api/v1/entities/{ent_id}",
                        headers=self.headers,
                        json={"confidence": 0.1},
                        timeout=10,
                    )
                    deprecated_count += 1
                except Exception:
                    pass

        # Clear hash tracking
        self._file_hashes.pop(file_path, None)

        return {
            "status": "deprecated",
            "file": file_path,
            "entities_deprecated": deprecated_count,
        }

    def sync_git_diff(self, repo_path: str, ref: str = "HEAD~1") -> dict:
        """Sync changed files from git diff to the knowledge graph.

        Processes added/modified files (extract knowledge) and deleted files
        (mark associated knowledge as deprecated).
        """
        # Get changed files from git diff
        try:
            result = subprocess.run(
                ["git", "diff", "--name-status", ref],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as e:
            return {"status": "error", "message": str(e)}

        added = []
        modified = []
        deleted = []
        supported_exts = {".md", ".txt", ".json", ".csv", ".html"}

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            status = parts[0]
            filepath = parts[-1]
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in supported_exts:
                continue
            full_path = os.path.join(repo_path, filepath)
            if status.startswith("A"):
                added.append(full_path)
            elif status.startswith("M"):
                modified.append(full_path)
            elif status.startswith("D"):
                deleted.append(full_path)
            elif status.startswith("R"):
                # Renamed: treat old as deleted, new as added
                if len(parts) >= 3:
                    deleted.append(os.path.join(repo_path, parts[1]))
                    added.append(os.path.join(repo_path, parts[2]))

        results = {"added": [], "modified": [], "deleted": []}

        for f in added + modified:
            if os.path.exists(f):
                results["modified" if f in modified else "added"].append(
                    self.sync_file(f)
                )

        for f in deleted:
            results["deleted"].append(self.handle_deletion(f))

        return results


def watch_mode(path: str, api_url: str, api_key: str = None):
    """Start file watcher mode. Requires watchdog package."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("Error: watchdog package not installed. Run: pip install watchdog")
        sys.exit(1)

    client = KGSyncClient(api_url, api_key)
    supported_exts = {".md", ".txt", ".json", ".csv", ".html"}

    class KGFileHandler(FileSystemEventHandler):
        """Handle file system events and sync to KG."""

        def on_modified(self, event):
            """Handle file modification events."""
            if event.is_directory:
                return
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext not in supported_exts:
                return
            print(f"[SYNC] File modified: {event.src_path}")
            result = client.sync_file(event.src_path)
            print(f"  -> {result.get('status', 'unknown')}")

        def on_created(self, event):
            """Handle file creation events."""
            if event.is_directory:
                return
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext not in supported_exts:
                return
            print(f"[SYNC] File created: {event.src_path}")
            result = client.sync_file(event.src_path)
            print(f"  -> {result.get('status', 'unknown')}")

        def on_deleted(self, event):
            """Handle file deletion events."""
            if event.is_directory:
                return
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext not in supported_exts:
                return
            print(f"[SYNC] File deleted: {event.src_path}")
            result = client.handle_deletion(event.src_path)
            print(f"  -> deprecated {result.get('entities_deprecated', 0)} entities")

    observer = Observer()
    handler = KGFileHandler()
    observer.schedule(handler, path, recursive=True)
    observer.start()
    print(f"Watching: {path}")
    print(f"API: {api_url}")
    print(f"Supported: {', '.join(supported_exts)}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped.")
    observer.join()


def main():
    """CLI entry point for KG Sync."""
    parser = argparse.ArgumentParser(description="KG Skill Document Sync")
    subparsers = parser.add_subparsers(dest="command")

    # Watch mode
    watch_parser = subparsers.add_parser("watch", help="Watch a directory for changes")
    watch_parser.add_argument("--path", required=True, help="Directory to watch")
    watch_parser.add_argument("--port", type=int, default=8700, help="API port")
    watch_parser.add_argument("--url", default=None, help="Full API URL")

    # Git diff mode
    git_parser = subparsers.add_parser("git-diff", help="Sync from git diff")
    git_parser.add_argument("--repo", required=True, help="Git repo path")
    git_parser.add_argument("--ref", default="HEAD~1", help="Git ref to diff against")
    git_parser.add_argument("--port", type=int, default=8700, help="API port")
    git_parser.add_argument("--url", default=None, help="Full API URL")

    args = parser.parse_args()

    if args.command == "watch":
        api_url = args.url or f"http://localhost:{args.port}"
        watch_mode(args.path, api_url)
    elif args.command == "git-diff":
        api_url = args.url or f"http://localhost:{args.port}"
        client = KGSyncClient(api_url)
        result = client.sync_git_diff(args.repo, args.ref)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
