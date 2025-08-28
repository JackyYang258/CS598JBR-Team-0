import pytest
from Task_2_solution import truncate_number  # replace with your actual module name

def test_truncate_number():
    assert truncate_number(1.23) == 0.23
    assert truncate_number(10.99) == 0.99
    assert truncate_number(100.00) == 0.00
    assert truncate_number(0.00) == 0.00
    assert truncate_number(-1.23) == -0.23
    assert truncate_number(-10.99) == -0.99
    assert truncate_number(-100.00) == 0.00