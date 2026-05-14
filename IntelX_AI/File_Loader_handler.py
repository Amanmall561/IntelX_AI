"""
File loader and handler utilities.

This module exposes a `FileLoaderHandler` class that inspects an input file,
guesses its type based on the extension, and delegates to the appropriate
handler to extract a structured payload. Supported types include PDF, DOCX,
XLSX, CSV, JSON, common image formats, and a generic binary fallback.

The handlers return a `FilePayload` dataclass instance that contains the
normalized file path, the detected logical type, a body (usually parsed
content), and auxiliary metadata describing what was extracted.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Union


class FileHandlerError(Exception):
    """Raised when a handler fails to process a supported file type."""


class UnsupportedFileTypeError(Exception):
    """Raised when no handler is registered for the supplied file type."""


@dataclass(slots=True)
class FilePayload:
    """Normalized representation of a processed file."""

    file_path: Path
    file_type: str
    body: Any
    metadata: MutableMapping[str, Any]


class FileLoaderHandler:
    """Route files to type-specific handlers based on extension."""

    _EXTENSION_TYPE_MAP: Dict[str, str] = {
        ".pdf": "pdf",
        ".doc": "doc",
        ".docx": "doc",
        ".xls": "xlsx",
        ".xlsx": "xlsx",
        ".xlsm": "xlsx",
        ".xltx": "xlsx",
        ".xltm": "xlsx",
        ".csv": "csv",
        ".json": "json",
        ".geojson": "json",
        ".txt": "text",
        ".md": "text",
        ".rtf": "text",
        ".log": "text",
        ".ini": "text",
        ".cfg": "text",
        ".yaml": "text",
        ".yml": "text",
        ".xml": "text",
        ".html": "text",
        ".htm": "text",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".bmp": "image",
        ".tif": "image",
        ".tiff": "image",
        ".gif": "image",
        ".webp": "image",
        ".heif": "image",
        ".heic": "image",
    }

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Path], FilePayload]] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Populate the handler registry with built-in handlers."""

        for extension in self._EXTENSION_TYPE_MAP.keys():
            self.register_handler(extension, self._handle_generic)

    def register_handler(self, extension: str, handler: Callable[[Path], FilePayload]) -> None:
        """Register a handler function for a given file extension."""

        normalized_extension = self._normalize_extension(extension)
        self._handlers[normalized_extension] = handler

    def handle(self, file_path: Union[str, Path]) -> FilePayload:
        """Load and parse a file, delegating to the appropriate handler."""

        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.is_dir():
            raise IsADirectoryError(f"Expected a file but got a directory: {path}")

        handler = self._get_handler_for_path(path)

        try:
            return handler(path)
        except UnsupportedFileTypeError:
            raise
        except Exception as exc:  # pragma: no cover - re-raising with context only
            raise FileHandlerError(f"Failed to handle file `{path}`: {exc}") from exc

    def _get_handler_for_path(self, path: Path) -> Callable[[Path], FilePayload]:
        extension = self._normalize_extension(path.suffix)

        if not extension and path.name.startswith("."):
            # No suffix but dotfile may still have type hints in name (e.g. `.env`)
            extension = path.name

        handler = self._handlers.get(extension)

        if handler:
            return handler

        # Fall back to the generic handler for any unregistered extension.
        return self._handle_generic

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        return extension.lower().strip()

    def _handle_generic(self, path: Path) -> FilePayload:
        extension = self._normalize_extension(path.suffix)
        file_type = self._EXTENSION_TYPE_MAP.get(extension, "other") if extension else "other"

        metadata: Dict[str, Any] = {"size_bytes": path.stat().st_size}
        if extension:
            metadata["extension"] = extension

        return FilePayload(file_path=path, file_type=file_type, body=None, metadata=metadata)


def file_handler_module(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Programmatic interface: detect the file type, parse it, and return a summary.
    Returns a dictionary with keys:
      - file: absolute path string
      - detected_type: logical file type
      - metadata: dict of extracted metadata
    On errors, returns: {"error": <category>, "message": <str>}
    """
    handler = FileLoaderHandler()

    try:
        payload = handler.handle(file_path)
    except (FileNotFoundError, IsADirectoryError) as exc:
        return {"error": "file_error", "message": str(exc)}
    except UnsupportedFileTypeError as exc:
        return {"error": "unsupported_type", "message": str(exc)}
    except FileHandlerError as exc:
        return {"error": "handler_error", "message": str(exc)}

    return {
        "file": str(payload.file_path),
        "detected_type": payload.file_type,
        "metadata": dict(payload.metadata),
    }




# def main(argv: Optional[Iterable[str]] = None) -> None:
#     """CLI entry-point to exercise FileLoaderHandler."""

#     parser = argparse.ArgumentParser(description="Detect and process structured files.")
#     parser.add_argument("file", help="Path to the file to inspect.")
#     parser.add_argument(
#         "--show-body",
#         action="store_true",
#         help="Print the extracted body payload (if available).",
#     )

#     args = parser.parse_args(argv)

#     handler = FileLoaderHandler()

#     try:
#         payload = handler.handle(args.file)
#     except (FileNotFoundError, IsADirectoryError) as exc:
#         parser.exit(status=1, message=f"[File Error] {exc}\n")
#     except UnsupportedFileTypeError as exc:
#         parser.exit(status=2, message=f"[Unsupported] {exc}\n")
#     except FileHandlerError as exc:
#         parser.exit(status=3, message=f"[Handler Error] {exc}\n")

#     print(f"File: {payload.file_path}")
#     print(f"Detected type: {payload.file_type}")
#     print("Metadata:")
#     print(json.dumps(payload.metadata, indent=2, default=str))

#     if not args.show_body:
#         return

#     if payload.body is None:
#         print("Body: <empty>")
#         return

#     print("Body preview:")
#     if isinstance(payload.body, bytes):
#         preview = payload.body[:200]
#         print(preview)
#         if len(payload.body) > len(preview):
#             print("... [truncated bytes output]")
#         return

#     if isinstance(payload.body, str):
#         preview = payload.body[:1000]
#         print(preview)
#         if len(payload.body) > len(preview):
#             print("... [truncated text output]")
#         return

#     print(json.dumps(payload.body, indent=2, default=str))


__all__ = [
    "FileHandlerError",
    "FileLoaderHandler",
    "FilePayload",
    "UnsupportedFileTypeError",
]


# if __name__ == "__main__":  # pragma: no cover - manual execution path
#     # Simple manual test; replace with any file path you want to check.
#     result = file_handler_module('/home/ubuntu/Airline_identify/pdf_layout_parser/logs/commercial_metrics.log')
#     print(json.dumps(result, indent=2, default=str))
