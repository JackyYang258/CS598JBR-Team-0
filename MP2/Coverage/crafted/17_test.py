import pytest
from Task_17_solution import parse_music  # replace 'your_module' with the name of the module where the function is defined


def test_parse_music():
    assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    assert parse_music('o| o| o| o|') == [2, 2, 2, 2]
    assert parse_music('o o o o o') == [4, 4, 4, 4]
    assert parse_music('') == []
    assert parse_music('o o| .| o| o| .| .| .| .| o o .|') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4, 1]


if __name__ == "__main__":
    pytest.main()