import pytest
from Task_7_solution import List

def filter_by_substring(strings: List[str], substring: str) -> List[str]:
    """ Filter an input list of strings only for ones that contain given substring
    >>> filter filter_by_substring([], 'a')
    []
    >>> filter filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
    ['abc', 'bacd', 'array']
    """
    return [x for x in strings if substring in x]

def test_filter_by_substring():
    assert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']
    assert filter_by_substring(['hello', 'world', 'pytest', 'unittest'], 't') == ['hello', 'pytest', 'unittest']
    assert filter_by_substring(['test', 'example', 'none'], 'x') == ['example']
    assert filter_by_substring(['test', 'example', 'none'], 'z') == []
    assert filter_by_substring([], 'a') == []