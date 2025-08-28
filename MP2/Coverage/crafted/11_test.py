import pytest
from Task_11_solution import string_xor  # replace 'your_module' with the name of the module where the function is defined

def test_string_xor():
    assert string_xor('010', '110') == '100'
    assert string_xor('101', '001') == '100'
    assert string_xor('111', '111') == '000'
    assert string_xor('0', '0') == '0'
    assert string_xor('1', '1') == '0'
    assert string_xor('', '') == ''

def test_string_xor_different_lengths():
    with pytest.raises(ValueError):
        string_xor('101', '1010')

def test_string_xor_non_binary_inputs():
    with pytest.raises(ValueError):
        string_xor('102', '101')
    with pytest.raises(ValueError):
        string_xor('101', '102')