# utils.py
import math

class Utils:
    @staticmethod
    def mean(array):
        if not array:
            return 0.0
        return sum(array) / len(array)

    @staticmethod
    def std_dev(array):
        n = len(array)
        if n <= 1:
            return 0.0
        avg = Utils.mean(array)
        variance = sum((x - avg) ** 2 for x in array) / (n - 1)
        return math.sqrt(variance)