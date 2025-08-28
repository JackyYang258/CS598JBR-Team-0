import pytest
from Task_48_solution import is_palindrome  # replace 'your_module' with the name of the module where the function is defined

def test_is_palindrome():
    assert is_palindrome('radar')
    assert is_palindrome('level')
    assert is_palindrome('a')
    assert not is_palindrome('python')
    assert is_palindrome('Able was I ere I saw Elba')
    assert not is_palindrome(' ')