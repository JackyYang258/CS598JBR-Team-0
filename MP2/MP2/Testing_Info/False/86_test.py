import pytest
from Task_86_solution import anti_shuffle  # replace 'your_module' with the name of the module where the function is defined

def test_anti_shuffle():
    assert anti_shuffle('Hi') == 'Hi'
    assert anti_shuffle('hello') == 'ehllo'
    assert anti_shuffle('Hello World!!!') == 'Hello !!!Wdlor'
    assert anti_shuffle('') == ''
    assert anti_shuffle('a') == 'a'
    assert anti_shuffle('Aa') == 'Aa'
    assert anti_shuffle('A B C') == 'A B C'
    assert anti_shuffle('Aa Bb Cc') == 'Aa Bb Cc'
    assert anti_shuffle('123') == '123'
    assert anti_shuffle('1 2 3') == '1 2 3'
    assert anti_shuffle('123 456') == '123 456'
    assert anti_shuffle('A1 B2 C3') == 'A1 B2 C3'
    assert anti_shuffle('A1 B2 C3   D4 E5 F6') == 'A1 B2 C3   D4 E5 F6'