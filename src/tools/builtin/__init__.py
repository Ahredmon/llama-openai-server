"""Registers all built-in tools into BUILTIN_REGISTRY."""
from src.tools.registry import BUILTIN_REGISTRY

from src.tools.builtin.datetime_tools import (
    GET_CURRENT_DATETIME_SCHEMA,
    CONVERT_TIMEZONE_SCHEMA,
    get_current_datetime,
    convert_timezone,
)
from src.tools.builtin.math_tools import (
    EVALUATE_MATH_SCHEMA,
    evaluate_math,
)
from src.tools.builtin.websearch import (
    WEB_SEARCH_SCHEMA,
    web_search,
)

BUILTIN_REGISTRY.register(GET_CURRENT_DATETIME_SCHEMA, get_current_datetime)
BUILTIN_REGISTRY.register(CONVERT_TIMEZONE_SCHEMA, convert_timezone)
BUILTIN_REGISTRY.register(EVALUATE_MATH_SCHEMA, evaluate_math)
BUILTIN_REGISTRY.register(WEB_SEARCH_SCHEMA, web_search)
