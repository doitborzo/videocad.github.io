class VideoCADOnshapeError(Exception):
    """Base exception for the autonomous Onshape runtime."""


class ConfigurationError(VideoCADOnshapeError):
    """Raised when required runtime configuration is missing or invalid."""


class UnsupportedPromptError(VideoCADOnshapeError):
    """Raised when a prompt cannot be normalized into the constrained CAD IR."""


class LiveRuntimeError(VideoCADOnshapeError):
    """Raised when the live Onshape runtime cannot proceed safely."""


class SafetyStopError(VideoCADOnshapeError):
    """Raised when a safety guard stops execution."""

