import pytest
from Task_0_solution import has_close_elements  # replace with your actual module name


def test_has_close_elements():
    # Test with a list of numbers and a threshold
    assert has_close_elements([1.0, 2.0, 3.0, 4.0], 1.5) == True
    assert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.5) == False
    assert has_close_elements([1.0, 1.0, 1.0, 1.0], 0.5) == True
    assert has_close_elements([1.0, 2.0, 3.0, 4.0], 1.0) == False
    assert has_close_elements([], 0.5) == False

    # Test with negative numbers
    assert has_close_elements([-1.0, -2.0, -3.0, -4.0], 1.5) == True
    assert has_close_elements([-1.0, 1.0, -2.0, 2.0], 3.0) == True

    # Test with zero
    assert has_close_elements([0.0, 1.0, 2.0, 3.0], 1.5) == True
    assert has_close_elements([0.0, 0.0, 0.0, 0.0], 0.5) == True

    # Test with large numbers
    assert has_close_elements([1e6, 1e6 + 0.5, 1e6 + 1.0], 0.5) == True
    assert has_close_elements([1e3, 2e3, 3e3, 4e3], 1e2) == False