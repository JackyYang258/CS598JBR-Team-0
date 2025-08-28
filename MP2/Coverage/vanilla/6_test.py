import pytest
from Task_6_solution import parse_nested_parens  # replace with your actual module name


def test_parse_nested_parens():
    assert parse_nested_parens("()") == [1]
    assert parse_nested_parens("(())") == [2]
    assert parse_nested_parens("()()") == [1, 1]
    assert parse_nested_parens("((()))") == [3]
    assert parse_nested_parens("((()))()") == [3, 1]
    assert parse_nested_parens("((()))(())") == [3, 2]
    assert parse_nested_parens("((()))((()))") == [3, 3]
    assert parse_nested_parens("") == []
    assert parse_nested_parens(" ") == []
    assert parse_nested_parens("  ") == []
    assert parse_nested_parens("   ") == []