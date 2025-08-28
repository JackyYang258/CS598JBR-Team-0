import pytest
from Task_137_solution import compare_one

def test_compare_one_same_numbers():
    assert compare_one('1.23', '1.23') == None
    assert compare_one(1.23, 1.23) == None

def test_compare_one_first_number_greater():
    assert compare_one('1.23', '1.22') == '1.23'
    assert compare_one(1.23, 1.22) == '1.23'

def test_compare_one_second_number_greater():
    assert compare_one('1.22', '1.23') == '1.23'
    assert compare_one(1.22, 1.23) == '1.23'

def test_compare_one_with_strings_and_numbers():
    assert compare_one('1.23', 1.22) == '1.23'
    assert compare_one(1.23, '1.22') == '1.23'
    assert compare_one('1,23', 1.22) == '1.23'
    assert compare_one(1.23, '1,22') == '1.23'