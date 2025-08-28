import pytest
from Task_66_solution import digitSum  # replace 'your_module' with the name of the module where the function is defined

def test_digitSum_empty_string():
    assert digitSum("") == 0

def test_digitSum_single_uppercase_letter():
    assert digitSum("A") == ord("A")

def test_digitSum_single_lowercase_letter():
    assert digitSum("a") == 0

def test_digitSum_multiple_letters():
    assert digitSum("ABC") == ord("A") + ord("C")

def test_digitSum_mixed_case_letters():
    assert digitSum("aBc") == ord("B")

def test_digitSum_digits():
    assert digitSum("123") == 0

def test_digitSum_special_characters():
    assert digitSum("!@#") == 0