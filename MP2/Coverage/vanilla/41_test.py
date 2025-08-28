import pytest
from Task_41_solution import car_race_collision  # replace 'your_module' with the name of the module where the function is defined

def test_car_race_collision():
    assert car_race_collision(2) == 4
    assert car_race_collision(3) == 9
    assert car_race_collision(4) == 16
    assert car_race_collision(5) == 25

def test_car_race_collision_negative():
    assert car_race_collision(-2) == 4
    assert car_race_collision(-3) == 9
    assert car_race_collision(-4) == 16
    assert car_race_collision(-5) == 25

def test_car_race_collision_zero():
    assert car_race_collision(0) == 0