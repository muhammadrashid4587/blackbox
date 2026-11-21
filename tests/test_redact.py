"""Redaction: secrets in keys and values never reach the log."""

import unittest

from blackbox.redact import REDACTED, redact


class RedactTest(unittest.TestCase):
    def test_sensitive_key_names(self):
        payload = {
            "api_key": "abc123",
            "password": "hunter2",
            "Authorization": "Bearer xyz",
            "query": "leave me alone",
        }
        out = redact(payload)
        self.assertEqual(out["api_key"], REDACTED)
        self.assertEqual(out["password"], REDACTED)
        self.assertEqual(out["Authorization"], REDACTED)
        self.assertEqual(out["query"], "leave me alone")

    def test_key_match_is_case_insensitive_and_substring(self):
        out = redact({"API_KEY": "x", "myTokenValue": "y", "token": "z",
                      "totally_clean": "keep"})
        self.assertEqual(out["API_KEY"], REDACTED)
        self.assertEqual(out["myTokenValue"], REDACTED)
        self.assertEqual(out["token"], REDACTED)
        self.assertEqual(out["totally_clean"], "keep")

    def test_secret_patterns_in_values(self):
        out = redact({
            "note": "key is sk-proj-ABCDEFGHIJKLMNOP1234 end",
            "header": "Bearer abcdefghij1234567890",
            "aws": "AKIAIOSFODNN7EXAMPLE",
            "gh": "ghp_abcdefghij1234567890",
        })
        self.assertNotIn("sk-proj-", out["note"])
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", out["aws"])
        self.assertNotIn("ghp_abcdefghij", out["gh"])
        self.assertIn(REDACTED, out["note"])
        self.assertTrue(out["header"].startswith("Bearer " + REDACTED))

    def test_nested_structures(self):
        out = redact({"args": {"headers": {"api_key": "secret"},
                               "items": [{"password": "p"}, "plain"]}})
        self.assertEqual(out["args"]["headers"]["api_key"], REDACTED)
        self.assertEqual(out["args"]["items"][0]["password"], REDACTED)
        self.assertEqual(out["args"]["items"][1], "plain")

    def test_clean_payload_untouched(self):
        payload = {"tool": "search", "args": {"query": "hello"}, "n": 3,
                   "flag": True}
        self.assertEqual(redact(payload), payload)

    def test_metadata_keys_are_not_redacted(self):
        # Regression test: "total_tokens" contains "token" as a substring
        # but it is a count, not a secret. Whole-word matching must not
        # redact it (or "events", "task", "started", "ended").
        out = redact({"total_tokens": 220, "events": 3, "task": "t",
                      "started": "2026-01-01", "tokens": 120})
        self.assertEqual(out["total_tokens"], 220)
        self.assertEqual(out["events"], 3)
        self.assertEqual(out["task"], "t")
        self.assertEqual(out["tokens"], 120)

    def test_input_not_mutated(self):
        payload = {"api_key": "abc", "nested": {"token": "def"}}
        redact(payload)
        self.assertEqual(payload["api_key"], "abc")
        self.assertEqual(payload["nested"]["token"], "def")


if __name__ == "__main__":
    unittest.main()
