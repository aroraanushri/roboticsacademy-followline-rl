"""Paste into FollowLine's Python editor while external training controls the car.

Do not import HAL here: its MotorsNode is another velocity publisher.
"""
import WebGUI
import Frequency

while True:
    Frequency.tick(10)
