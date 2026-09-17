"""Moderation examples are unit-test strings, never player data or recommendations."""
import pytest

from src.name_safety import mask_name, mask_known_names, mask_text


@pytest.mark.parametrize("name, expected", [
    ("Sh1tJungle", "****Jungle"),
    ("itchy butthole", "itchy ********"),
    ("FUCKSupport", "****Support"),
    ("f.u.c.k", "*******"),
    ("f u c k", "*******"),
    ("Ｆｕｃｋ", "****"),
    ("f\u200buck", "*****"),
    ("Assassin", "Assassin"),
    ("Scunthorpe", "Scunthorpe"),
    ("Dickson", "Dickson"),
    ("别办球女", "别办球女"),
])
def test_masked_spans_and_common_false_positives(name, expected):
    assert mask_name(name) == expected


def test_known_names_with_spaces_are_masked_in_prose():
    assert mask_known_names("f u c k has 20 matches", ["f u c k"]) == "******* has 20 matches"
    assert mask_text("Class and assists: 5.0") == "Class and assists: 5.0"
