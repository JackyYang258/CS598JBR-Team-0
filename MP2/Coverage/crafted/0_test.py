import pytest
from Task_0_solution import has_close_elements  # replace with your actual module name


def test_has_close_elements():
    assert not has_close_elements([1.0, 2.0, 3.0], 0.5)
    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    assert not has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.5)
    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.4)
    assert not has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.2)
    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.1)