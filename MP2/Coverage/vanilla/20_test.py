import pytest
from Task_20_solution import find_closest_elements

def test_find_closest_elements():
    assert find_closest_elements([1.0, 2.0, 3.0, 4.0]) == (1.0, 2.0)
    assert find_closest_elements([1.0, 3.0, 6.0, 7.0]) == (6.0, 7.0)
    assert find_closest_elements([-1.0, -3.0, -6.0, -7.0]) == (-1.0, -3.0)
    assert find_closest_elements([1.0]) == None
    assert find_closest_elements([]) == None