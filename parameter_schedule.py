"""
Learning data-driven proposals through reinforcement learning

This file contains the classes for implementing parameter schedules. These are currently used for implementing
learning rate schedules.

13 Dec. 2016
https://github.com/gokererdogan
"""


class ParameterSchedule(object):
    def __init__(self):
        pass

    def get_value(self, iteration_no):
        raise NotImplementedError()


class ConstantSchedule(ParameterSchedule):
    def __init__(self, value):
        self.value = value

    def get_value(self, iteration_no):
        return self.value

    def __str__(self):
        return "Constant schedule, value: {0:f}".format(self.value)

    def __repr__(self):
        return self.__str__()


class LinearSchedule(ParameterSchedule):
    def __init__(self, start_value, end_value, decrease_for):
        ParameterSchedule.__init__(self)
        self.start_value = start_value
        self.end_value = end_value
        self.decrease_for = decrease_for
        self.decreasing = self.start_value > self.end_value

    def get_value(self, iteration_no):
        assert(iteration_no >= 0)
        value = self.start_value + (iteration_no * (self.end_value - self.start_value) / self.decrease_for)
        if (self.decreasing and value < self.end_value) or (not self.decreasing and value > self.end_value):
            value = self.end_value
        return value

    def __str__(self):
        return "Linear schedule, start: {0:f}, end: {1:f}, decrease for {2:d}".format(self.start_value,
                                                                                      self.end_value,
                                                                                      self.decrease_for)

    def __repr__(self):
        return self.__str__()

