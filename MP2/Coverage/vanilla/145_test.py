import pytest
from Task_145_solution import order_by_points  # replace 'your_module' with the name of the module where the function is defined

def test_order_by_points():
    assert order_by_points([15, 20, -30, 45]) == [-30, 15, 20, 45]
    assert order_by_points([10, 200, 3000, -4000]) == [10, 200, 3000, -4000]
    assert order_by_points([0, 1, 2, 3, 4, 5]) == [0, 1, 2, 3, 4, 5]
    assert order_by_points([-1, -2, -3, -4, -5]) == [-1, -2, -3, -4, -5]
    assert order_by_points([123, 321, 456, 654]) == [123, 321, 456, 654]
    assert order_by_points([]) == []