import pytest
from Task_83_solution import starts_one_ends  # replace 'your_module' with the actual module name

def test_starts_one_ends():
    assert starts_one_ends(1) == 1
    assert starts_one_ends(2) == 18
    assert starts_one_ends(3) == 180
    assert starts_one_ends(4) == 1800
    assert starts_one_ends(5) == 18000

def test_starts_one_ends_with_non_integer():
    with pytest.raises(TypeError):
        starts_one_ends('a')
    with pytest.raises(TypeError):
        starts_one_ends(1.5)

def test_starts_one_ends_with_negative():
    with pytest.raises(ValueError):
        starts_one_ends(-1)