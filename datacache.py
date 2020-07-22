import random

class DataCache(object):
    def __init__(self, size):
        self.size = size
        self.x = []

    def fill(self, datapoint):
        while len(self.x) < self.size:
            self.add(datapoint)

    def add(self, datapoint):
        self.x.append(datapoint)

    def get_data(self):
        return self.x.pop(random.randrange(len(self.x)))

