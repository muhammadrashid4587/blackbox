"""Redaction before write.

Secrets are replaced before a record is hashed or written, so the log
itself never contains them. Redaction is recursive: it walks dicts,
lists and tuples, and checks every string value.

Two rules, applied in order:
1. Key names: if a key contains a sensitive substring (case-insensitive),
   the whole value is replaced with "[REDACTED]".
2. Values: if a string value matches a known secret pattern (an
   `sk-...` key, a Bearer token, a GitHub PAT, an AWS access key id,
   a Slack token), the matching part is replaced with "[REDACTED]".

Clean values are left untouched.
"""

import re

REDACTED = "[REDACTED]"

# Words that mark a key as sensitive. Matching is on whole words, so a
# metadata key like "total_tokens" is NOT redacted: its words are
# "total" and "tokens", neither of which is in this set.
SENSITIVE_WORDS = {
    "apikey",
    "token",
    "password",
    "passwd",
    "secret",
    "authorization",
    "auth",
    "privatekey",
    "accesskey",
    "session",
    "cookie",
    "bearer",
    "clientsecret",
    "awssecret",
}

# (compiled pattern, replacement) for values that look like secrets.
SECRET_PATTERNS = (
    # OpenAI / Anthropic style keys: sk-... and sk-ant-...
    (re.compile(r"sk-(?:ant-)?[A-Za-z0-9_-]{16,}"), REDACTED),
    # Authorization: Bearer <token>
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-~+/=]+"), "Bearer " + REDACTED),
    # GitHub personal access tokens: ghp_, gho_, ghu_, ghs_, ghr_, github_pat_
    (re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), REDACTED),
    # AWS access key ids: AKIA + 16 uppercase/digits
    (re.compile(r"AKIA[0-9A-Z]{16}"), REDACTED),
    # Slack tokens: xoxb-, xoxp-, xoxa-, xoxr-, xoxs-
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]+"), REDACTED),
)


def _words(key):
    """Split a key into lowercase words on separators and camelCase
    boundaries: 'myTokenValue' -> ['my', 'token', 'value'],
    'api_key' -> ['api', 'key'], 'total_tokens' -> ['total', 'tokens']."""
    words = []
    for part in re.split(r"[^A-Za-z0-9]+", key):
        words.extend(
            re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", part))
    return [w.lower() for w in words if w]


def _key_is_sensitive(key):
    words = _words(key)
    if any(w in SENSITIVE_WORDS for w in words):
        return True
    joined = "".join(words)
    return any(joined.endswith(w) for w in SENSITIVE_WORDS)


def _redact_string(value):
    for pattern, replacement in SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def redact(obj):
    """Return a copy of obj with secrets replaced. The input is not mutated."""
    if isinstance(obj, dict):
        return {
            key: REDACTED if _key_is_sensitive(key) else redact(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [redact(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(redact(item) for item in obj)
    if isinstance(obj, str):
        return _redact_string(obj)
    return obj
