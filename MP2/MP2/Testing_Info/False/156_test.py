import pytest
from Task_156_solution import int_to_mini_roman  # replace 'your_module' with the name of the module containing the function

def test_int_to_mini_roman():
    assert int_to_mini_roman(19) == 'xix'
    assert int_to_mini_roman(152) == 'clii'
    assert int_to_mini_roman(426) == 'cdxxvi'
    assert int_to_mini_roman(1000) == 'm'
    assert int_to_mini_roman(999) == 'cmxcix'
    assert int_to_mini_roman(587) == 'dlxxxvii'
    assert int_to_mini_roman(375) == 'ccclxxv'
    assert int_to_mini_roman(249) == 'ccxlix'
    assert int_to_mini_roman(123) == 'cxxiii'
    assert int_to_mini_roman(67) == 'lxvii'
    assert int_to_mini_roman(45) == 'xlv'
    assert int_to_mini_roman(29) == 'xxix'
    assert int_to_mini_roman(14) == 'xiv'
    assert int_to_mini_roman(9) == 'ix'
    assert int_to_mini_roman(5) == 'v'
    assert int_to_mini_roman(4) == 'iv'
    assert int_to_mini_roman(1) == 'i'