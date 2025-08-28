import pytest
from Task_86_solution import anti_shuffle  # replace 'your_module' with the name of the module where the function is defined

def test_anti_shuffle():
    assert anti_shuffle("") == ""
    assert anti_shuffle("ab") == "ab"
    assert anti_shuffle("abc") == "abc"
    assert anti_shuffle("aab") == "aba"
    assert anti_shuffle("abc ab") == "abc ba"
    assert anti_shuffle("abc abc") == "abc cba"
    assert anti_shuffle("aab bca") == "aba bac"
    assert anti_shuffle("aab bbca") == "aba bbac"
    assert anti_shuffle("aabb ccba") == "aabb cba"
    assert anti_shuffle("aabb ccbaa") == "aabb aacb"
    assert anti_shuffle("aabb ccbaab") == "aabb abc"
    assert anti_shuffle("aabb ccbaaab") == "aabb abcba"