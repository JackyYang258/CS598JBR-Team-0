import pytest
from Task_6_solution import parse_nested_parens  # replace 'your_module' with the name of the module where the function is defined


def test_parse_nested_parens():
    assert parse_nested_parens('(()())') == 2
    assert parse_nested_parens('((()))') == 3
    assert parse_nested_parens('()') == 1
    assert parse_nested_parens('((())()())') == 3
    assert parse_nested_parens('') == []
    assert parse_nested_parens(' ') == []
    assert parse_nested_parens('(() ()())') == 2
    assert parse_nested_parens('((()()()))') == 3
    assert parse_nested_parens('()()()()()') == 1
    assert parse_nested_parens('((((()))))') == 4
    assert parse_nested_parens('(()()(()()))') == 3