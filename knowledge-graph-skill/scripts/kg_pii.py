"""
KG Skill PII Detection & Masking Module.

Detects and masks personally identifiable information (PII) in text before
it enters the knowledge graph. Supports Chinese and international PII patterns.

Usage:
    from kg_pii import PIIDetector
    detector = PIIDetector()
    masked, report = detector.detect_and_mask(text)
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PIIMatch:
    """A single PII detection result."""
    pii_type: str
    start: int
    end: int
    original: str
    masked: str


class PIIDetector:
    """Detect and mask PII in text using regex patterns."""

    # PII patterns: (name, compiled_regex, masking_strategy)
    PATTERNS = [
        ("email", re.compile(
            r"[\w.+-]+@[\w-]+\.[\w.]+"
        ), "mask_email"),
        ("phone_cn", re.compile(
            r"(?<!\d)1[3-9]\d{9}(?!\d)"
        ), "mask_phone"),
        ("phone_intl", re.compile(
            r"\+\d{1,3}[\s\-]?\d{4,}[\s\-]?\d{3,}"
        ), "mask_phone"),
        ("id_card_cn", re.compile(
            r"(?<!\d)\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
        ), "mask_id_card"),
        ("bank_card", re.compile(
            r"(?<!\d)\d{16,19}(?!\d)"
        ), "mask_bank_card"),
        ("ip_address", re.compile(
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        ), "mask_ip"),
    ]

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect all PII occurrences in text. Returns list of matches."""
        matches = []
        for pii_type, pattern, strategy in self.PATTERNS:
            for m in pattern.finditer(text):
                original = m.group(0)
                masked = self._apply_mask(strategy, original)
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    start=m.start(),
                    end=m.end(),
                    original=original,
                    masked=masked,
                ))
        # Sort by position, handle overlaps (keep first)
        matches.sort(key=lambda x: x.start)
        # Remove overlapping matches
        filtered = []
        last_end = -1
        for m in matches:
            if m.start >= last_end:
                filtered.append(m)
                last_end = m.end
        return filtered

    def mask(self, text: str) -> str:
        """Return text with all PII masked."""
        matches = self.detect(text)
        if not matches:
            return text
        # Build masked text by replacing from end to start
        result = text
        for m in reversed(matches):
            result = result[:m.start] + m.masked + result[m.end:]
        return result

    def detect_and_mask(self, text: str) -> Tuple[str, List[dict]]:
        """Detect PII, mask it, and return (masked_text, report).

        Report is a list of dicts with type, original (masked), and position.
        Original values are NOT included in the report for safety.
        """
        matches = self.detect(text)
        if not matches:
            return text, []
        result = text
        report = []
        for m in reversed(matches):
            result = result[:m.start] + m.masked + result[m.end:]
            report.append({
                "type": m.pii_type,
                "masked": m.masked,
                "position": [m.start, m.end],
            })
        report.reverse()
        return result, report

    def _apply_mask(self, strategy: str, value: str) -> str:
        """Apply the appropriate masking strategy to a PII value."""
        if strategy == "mask_email":
            parts = value.split("@")
            if len(parts) == 2:
                user = parts[0]
                if len(user) > 2:
                    return user[0] + "***" + user[-1] + "@" + parts[1]
                return "***@" + parts[1]
            return "***@***"
        elif strategy == "mask_phone":
            if len(value) >= 7:
                return value[:3] + "****" + value[-4:]
            return "****"
        elif strategy == "mask_id_card":
            if len(value) >= 10:
                return value[:6] + "********" + value[-4:]
            return "****"
        elif strategy == "mask_bank_card":
            if len(value) >= 4:
                return "**** **** **** " + value[-4:]
            return "****"
        elif strategy == "mask_ip":
            parts = value.split(".")
            if len(parts) == 4:
                return parts[0] + "." + parts[1] + ".***.***"
            return "***.***"
        return "***"
