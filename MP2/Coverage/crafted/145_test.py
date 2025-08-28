import pytest
from Task_145_solution import order_by_points  # replace 'your_module' with the name of the module where the function is defined

def test_order_by_points():
    assert order_by_points([1, 11, -1, -11, -12]) == [-1, -11, 1, -12, 11]
    assert order_by_points([15, 2, 12, 1]) == [12, 15, 2, 1]  # test with positive numbers
    assert order_by_points([-15, -2, -12, -1]) == [-1, -2, -12, -15]  # test with negative numbers
    assert order_by_points([]) == []  # test with empty list
    assert order_by_points([111, 222, 333]) == [111, 222, 333]  # test with numbers having same digits sum
    assert order_by_points([111, 22, 3333]) == [22, 111, 3333]  # test with numbers having different digits sum