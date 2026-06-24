"""Date / time built-in tools."""
from __future__ import annotations

import zoneinfo
from datetime import datetime, timezone


def get_current_datetime(tz: str = "UTC") -> dict:
    """Return the current date, time, and related info for the given timezone."""
    try:
        zone = zoneinfo.ZoneInfo(tz)
    except zoneinfo.ZoneInfoNotFoundError:
        return {"error": f"Unknown timezone: {tz!r}. Use IANA names such as 'America/New_York'."}
    now = datetime.now(zone)
    return {
        "datetime_iso": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "timezone": tz,
        "utc_offset": now.strftime("%z"),
        "weekday": now.strftime("%A"),
        "unix_timestamp": int(now.timestamp()),
    }


def convert_timezone(dt_iso: str, from_tz: str, to_tz: str) -> dict:
    """Convert an ISO-8601 datetime string from one timezone to another."""
    try:
        src_zone = zoneinfo.ZoneInfo(from_tz)
    except zoneinfo.ZoneInfoNotFoundError:
        return {"error": f"Unknown source timezone: {from_tz!r}"}
    try:
        dst_zone = zoneinfo.ZoneInfo(to_tz)
    except zoneinfo.ZoneInfoNotFoundError:
        return {"error": f"Unknown destination timezone: {to_tz!r}"}
    try:
        src_dt = datetime.fromisoformat(dt_iso)
    except ValueError:
        return {"error": f"Could not parse datetime: {dt_iso!r}. Expected ISO-8601 format."}
    if src_dt.tzinfo is None:
        src_dt = src_dt.replace(tzinfo=src_zone)
    else:
        src_dt = src_dt.astimezone(src_zone)
    dst_dt = src_dt.astimezone(dst_zone)
    return {
        "input": src_dt.isoformat(),
        "output": dst_dt.isoformat(),
        "from_tz": from_tz,
        "to_tz": to_tz,
        "utc_offset": dst_dt.strftime("%z"),
    }


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

GET_CURRENT_DATETIME_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_current_datetime",
        "description": (
            "Returns the current date, time, weekday, and UTC offset for a given IANA timezone. "
            "Use this whenever the user asks about the current time, date, or day."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tz": {
                    "type": "string",
                    "description": (
                        "IANA timezone name, e.g. 'UTC', 'America/New_York', 'Europe/London'. "
                        "Defaults to 'UTC' when omitted."
                    ),
                }
            },
            "required": [],
        },
    },
}

CONVERT_TIMEZONE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "convert_timezone",
        "description": (
            "Converts an ISO-8601 datetime string from one IANA timezone to another. "
            "Useful when the user asks what time it is in another city or country."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dt_iso": {
                    "type": "string",
                    "description": "ISO-8601 datetime string to convert, e.g. '2026-05-12T14:00:00'.",
                },
                "from_tz": {
                    "type": "string",
                    "description": "Source IANA timezone name.",
                },
                "to_tz": {
                    "type": "string",
                    "description": "Destination IANA timezone name.",
                },
            },
            "required": ["dt_iso", "from_tz", "to_tz"],
        },
    },
}
