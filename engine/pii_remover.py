import re

from typing import List

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.predefined_recognizers import (
    AuAbnRecognizer, AuAcnRecognizer, AuTfnRecognizer, AuMedicareRecognizer, PhoneRecognizer,
    SpacyRecognizer
)
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import RecognizerResult

TITLES = {"mr", "mrs", "ms", "miss", "dr", "mx"}

def setup_presidio_analyzer() ->tuple[AnalyzerEngine, AnonymizerEngine]:
    """Setup Presidio Analyzer with custom recognizers if needed"""
    anonymizer = AnonymizerEngine()

    # Example: Add a custom recognizer for a specific pattern
    # custom_pattern = Pattern(name="CUSTOM_PATTERN", regex=r"\b[A-Z]{3}-\d{4}\b", score=0.85)
    # custom_recognizer = PatternRecognizer(supported_entity="CUSTOM_ENTITY", patterns=[custom_pattern])
    # analyzer.registry.add_recognizer(custom_recognizer)

    # ABN Pattern: Matches 11 digits total, capturing variations like "XX XXX XXX XXX" or "XXXXXXXXXXX"
    abn_pattern = Pattern(
        name="au_abn_pattern",
        regex=r"\b\d{2}(?: *\d{3}){3}\b",
        score=0.7
    )

    # Build Custom Recognizer with Context Words to avoid false positives
    au_abn_recognizer = PatternRecognizer(
        supported_entity="AU_ABN",  # Overlays / reinforces Presidio's built-in entity
        patterns=[abn_pattern],
        context=["abn", "australian business number", "bus no", "business registration"]
    )

    # BSB: Matches 123-456 or 123456
    bsb_pattern = Pattern(
        name="au_bsb_pattern",
        regex=r"\b\d{3}[- ]?\d{3}\b",
        score=0.1  # too generic on its own; only kept when a context word lifts it
    )

    # Build Custom Recognizer with Context Words to avoid false positives
    au_bsb_recognizer = PatternRecognizer(
        supported_entity="AU_BSB",
        patterns=[bsb_pattern],
        context=["bsb", "bank branch", "branch code", "bsb number"]
    )

    # Account Number: Matches standard 6 to 10 digit strings
    account_pattern = Pattern(
        name="au_account_pattern",
        regex=r"\b(?:\d *){5,9}\d\b",
        score=0.1  # too generic on its own (matched "03 62727138" phone numbers)
    )

    # Build Custom Recognizer with Context Words to avoid false positives
    au_account_recognizer = PatternRecognizer(
        supported_entity="AU_BANK_ACCOUNT",
        patterns=[account_pattern],
        context=["account number", "acc no", "acct", "bank account", "account #", "account"]
    )

    # Initialize Engine and Register New Recognizers
    # Drop weak matches, e.g. bare digit runs with no BSB/account context word nearby
    analyzer = AnalyzerEngine(default_score_threshold=0.4)
    analyzer.registry.add_recognizer(au_bsb_recognizer)
    analyzer.registry.add_recognizer(au_account_recognizer)

    # 4. Swap US/UK recognizers for Australian ones. The US ones match any long
    # digit run, e.g. an 11-digit ABN was tagged US_BANK_NUMBER and a 10-digit
    # phone number UK_NHS.
    for name in ["UsBankRecognizer", "UsLicenseRecognizer", "UsItinRecognizer",
                 "UsPassportRecognizer", "UsSsnRecognizer", "NhsRecognizer",
                 "PhoneRecognizer", "SpacyRecognizer"]:
        analyzer.registry.remove_recognizer(name)

    # spaCy's DATE tags were noisy ("1093", "20MG 30", "30 Days"), so keep it
    # for names/places only and leave dates to the regex DateRecognizer.
    analyzer.registry.add_recognizer(
        SpacyRecognizer(supported_entities=["PERSON", "LOCATION", "NRP", "ORGANIZATION"])
    )

    analyzer.registry.add_recognizer(AuAbnRecognizer())
    analyzer.registry.add_recognizer(AuAcnRecognizer())
    analyzer.registry.add_recognizer(AuTfnRecognizer())
    analyzer.registry.add_recognizer(AuMedicareRecognizer())
    analyzer.registry.add_recognizer(PhoneRecognizer(supported_regions=["AU"]))

    return analyzer, anonymizer

def propagate_person_names(text: str, results: List[RecognizerResult]) -> List[RecognizerResult]:
    """Mask every occurrence of a name that Presidio found as PERSON at least once in this document"""
    names = set()
    for r in results:
        if r.entity_type != "PERSON":
            continue
        # spaCy spans can run across line breaks ("MRS WENDY DEAKES\nMARY\nDate"); keep the first line
        span = text[r.start:r.end].splitlines()[0]
        words = [w for w in re.findall(r"[A-Za-z'-]+", span) if w.lower().rstrip(".") not in TITLES]
        # Single words ("MARY") are too likely to match ordinary text, so only use full names
        if len(words) >= 2:
            names.add(tuple(w.lower() for w in words))

    found = [(r.start, r.end) for r in results if r.entity_type == "PERSON"]
    extra = []
    for words in names:
        pattern = r"\b" + r"\s+".join(map(re.escape, words)) + r"\b"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if not any(s <= m.start() and m.end() <= e for s, e in found):
                extra.append(RecognizerResult("PERSON", m.start(), m.end(), 0.85))
    return results + extra
