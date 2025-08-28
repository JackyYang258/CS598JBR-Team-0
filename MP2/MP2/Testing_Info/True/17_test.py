import pytest
from Task_17_solution import parse_music  # replace 'your_module' with the name of the module where the function is defined


def test_parse_music():
    assert parse_music('o') == [4]
    assert parse_music('o|') == [2]
    assert parse_music('.|') == [1]
    assert parse_music('o o| .|') == [4, 2, 1]
    assert parse_music('') == []
    assert parse_music('o o| .| ') == [4, 2, 1]  # test with trailing space
    assert parse_music(' o o| .| ') == [4, 2, 1]  # test with leading and trailing spaces