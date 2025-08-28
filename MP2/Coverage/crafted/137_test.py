import pytest
from Task_137_solution import compare_one

def test_compare_one():
    assert compare_one(1, 2.5) == 2.5
    assert compare_one(1, "2,3") == "2,3"
    assert compare_one("5,1", "6") == "6"
    assert compare_one("1", 1) == None
    assert compare_one(10, 2) == 10
    assert compare_one("1.5", "2,3") == "2,3"
    assert compare_one("5,1", 6) == "6"
    assert compare_one(1, "1") == None
    assert compare_one("1.2", 1) == "1.2"
    assert compare_one("1", 1.2) == "1.2"
    assert compare_one(1.2, 1) == "1.2"
    assert compare_one("1.2", "1.2") == None