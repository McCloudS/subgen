"""Tests for the LanguageCode enum and its helper methods."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from language_code import LanguageCode


class TestFromString:
    def test_iso_639_1(self):
        assert LanguageCode.from_string("en") == LanguageCode.ENGLISH

    def test_iso_639_2(self):
        assert LanguageCode.from_string("eng") == LanguageCode.ENGLISH

    def test_english_name(self):
        assert LanguageCode.from_string("English") == LanguageCode.ENGLISH

    def test_native_name(self):
        assert LanguageCode.from_string("Deutsch") == LanguageCode.GERMAN

    def test_none_input(self):
        assert LanguageCode.from_string(None) == LanguageCode.NONE

    def test_empty_string(self):
        assert LanguageCode.from_string("") == LanguageCode.NONE

    def test_invalid_code(self):
        assert LanguageCode.from_string("zz") == LanguageCode.NONE

    def test_case_insensitive_iso(self):
        assert LanguageCode.from_string("EN") == LanguageCode.ENGLISH
        assert LanguageCode.from_string("ENG") == LanguageCode.ENGLISH

    def test_case_insensitive_name(self):
        assert LanguageCode.from_string("english") == LanguageCode.ENGLISH
        assert LanguageCode.from_string("ENGLISH") == LanguageCode.ENGLISH

    def test_french(self):
        assert LanguageCode.from_string("fr") == LanguageCode.FRENCH
        assert LanguageCode.from_string("fra") == LanguageCode.FRENCH
        assert LanguageCode.from_string("fre") == LanguageCode.FRENCH

    def test_whitespace_stripped(self):
        assert LanguageCode.from_string("  en  ") == LanguageCode.ENGLISH


class TestFromName:
    def test_english_name_returns_language_code(self):
        result = LanguageCode.from_name("English")
        assert result is not None
        assert result == LanguageCode.ENGLISH

    def test_native_name_returns_language_code(self):
        result = LanguageCode.from_name("Deutsch")
        assert result is not None
        assert result == LanguageCode.GERMAN

    def test_no_match_returns_none_language_code(self):
        # Regression: from_name() previously had no `return` on the NONE path,
        # always returning Python None instead of LanguageCode.NONE.
        result = LanguageCode.from_name("Klingon")
        assert result is not None, "from_name() must return LanguageCode.NONE, not Python None"
        assert result == LanguageCode.NONE


class TestConversions:
    def test_to_iso_639_1(self):
        assert LanguageCode.ENGLISH.to_iso_639_1() == "en"
        assert LanguageCode.FRENCH.to_iso_639_1() == "fr"

    def test_to_iso_639_2_b(self):
        assert LanguageCode.ENGLISH.to_iso_639_2_b() == "eng"

    def test_roundtrip_iso_639_1(self):
        lang = LanguageCode.FRENCH
        code = lang.to_iso_639_1()
        assert LanguageCode.from_string(code) == lang

    def test_to_name_english(self):
        assert LanguageCode.ENGLISH.to_name() == "English"


class TestBoolAndEquality:
    def test_valid_language_is_truthy(self):
        assert bool(LanguageCode.ENGLISH) is True

    def test_none_language_is_falsy(self):
        assert bool(LanguageCode.NONE) is False

    def test_none_equals_python_none(self):
        assert LanguageCode.NONE == LanguageCode.NONE

    def test_valid_language_not_equal_to_none(self):
        assert LanguageCode.ENGLISH != LanguageCode.NONE


class TestIsValidLanguage:
    def test_valid_iso(self):
        assert LanguageCode.is_valid_language("en") is True
        assert LanguageCode.is_valid_language("eng") is True

    def test_invalid(self):
        assert LanguageCode.is_valid_language("zzz") is False
        assert LanguageCode.is_valid_language("") is False


class TestEnumConsistency:
    def test_all_members_except_none_have_iso_639_1(self):
        for lang in LanguageCode:
            if lang == LanguageCode.NONE:
                continue
            code = lang.to_iso_639_1()
            assert code is not None and len(code) > 0, (
                f"{lang.name} is missing an ISO 639-1 code"
            )
