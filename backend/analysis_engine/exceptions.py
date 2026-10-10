"""Custom exceptions for the Analysis Engine module."""


class AnalysisError(Exception):
    """Base exception for all analysis engine errors."""


class ValidationError(AnalysisError):
    """Input validation failed."""


class ModelSelectionError(AnalysisError):
    """Could not determine which model to use."""


class InferenceError(AnalysisError):
    """Model inference failed."""
