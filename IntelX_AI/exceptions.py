"""
Custom exception classes for IntelX_AI modules.
All modules should raise these exceptions instead of returning error strings.
"""
from typing import Optional


class IntelXAIError(Exception):
    """Base exception for all IntelX_AI errors."""
    def __init__(self, message: str, file_path: Optional[str] = None):
        self.message = message
        self.file_path = file_path
        super().__init__(self.message)


class FileHandlerError(IntelXAIError):
    """Raised when file handling fails."""
    pass


class ParserError(IntelXAIError):
    """Raised when parsing fails."""
    pass


class CSVParserError(ParserError):
    """Raised when CSV parsing fails."""
    pass


class JSONParserError(ParserError):
    """Raised when JSON parsing fails."""
    pass


class XLSXParserError(ParserError):
    """Raised when XLSX parsing fails."""
    pass


class DOCParserError(ParserError):
    """Raised when DOC/DOCX parsing fails."""
    pass


class PDFParserError(ParserError):
    """Raised when PDF parsing fails."""
    pass


class ModelError(IntelXAIError):
    """Raised when model operations fail."""
    pass


class LLMError(IntelXAIError):
    """Raised when LLM operations fail."""
    pass


class DependencyError(IntelXAIError):
    """Raised when required dependencies are missing."""
    pass

