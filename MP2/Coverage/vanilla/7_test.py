import pytest
from Task_7_solution import List

def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    return [x for x in strings if substring in x]

def test_filter_by_substring():
    assert filter_by_substring(["hello", "world", "python", "programming"], "pro") == ["programming"]
    assert filter_by_substring(["apple", "banana", "cherry", "date"], "an") == ["banana"]
    assert filter_by_substring(["a", "ab", "abc", "abcd"], "bcd") == ["abcd"]
    assert filter_by_substring(["test", "fail", "pass"], "s") == ["test", "pass"]
    assert filter_by_substring(["none"], "n") == ["none"]
    assert filter_by_substring(["empty"], "") == ["empty"]
    assert filter_by_substring(["no match"], "xyz") == []