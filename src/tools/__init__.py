"""Built-in tool registry.  Imported once at startup; callers use BUILTIN_REGISTRY."""
from src.tools.registry import BUILTIN_REGISTRY  # noqa: F401

# Side-effect: registers all built-in tools into BUILTIN_REGISTRY
import src.tools.builtin  # noqa: F401
