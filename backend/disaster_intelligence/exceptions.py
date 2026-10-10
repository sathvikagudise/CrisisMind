"""Custom exceptions for the Disaster Intelligence module."""


class DisasterIntelligenceError(Exception):
    """Base exception for all disaster intelligence errors."""


class ProviderUnavailable(DisasterIntelligenceError):
    """Raised when a data provider is unreachable or returns a non-200 status."""


class InvalidResponse(DisasterIntelligenceError):
    """Raised when a provider returns data that cannot be parsed or validated."""


class RateLimitExceeded(DisasterIntelligenceError):
    """Raised when the provider's rate limit has been exceeded."""


class InvalidConfiguration(DisasterIntelligenceError):
    """Raised when the module configuration is invalid or missing required fields."""


class NoDataAvailable(DisasterIntelligenceError):
    """Raised when a provider returns zero events for the requested parameters."""
