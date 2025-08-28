import pytest
from Task_11_solution import string_xor  # replace 'your_module' with the name of the module where the function is defined


def test_string_xor():
    assert string_xor('001', '110') == '111'
    assert string_xor('101', '010') == '111'
    assert string_xor('111', '000') == '111'
    assert string_xor('0', '1') == '1'
    assert string_xor('1', '0') == '1'
    assert string_xor('', '') == ''
    assert string_xor('1111', '0000') == '1111'
    assert string_xor('10101010', '01010101') == '11111111'


def test_string_xor_different_lengths():
    with pytest.raises(ValueError):
        string_xor('101', '1010')


def test_string_xor_non_binary_input():
    with pytest.raises(ValueError):
        string_xor('2', '3')