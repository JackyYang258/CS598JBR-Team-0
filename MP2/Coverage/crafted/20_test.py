import pytest
from Task_20_solution import find_closest_elements

def test_find_closest_elements():
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == (1.0, 2.0)
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 0.5]) == (0.5, 1.0)
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, -2.0]) == (-2.0, 1.0)
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 1.5]) == (1.5, 2.0)
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.5]) == (2.0, 2.5)
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 1.9]) == (1.9, 2.0)