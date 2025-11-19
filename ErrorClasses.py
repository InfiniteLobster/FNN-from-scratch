

class NotSupportedInputGiven(Exception):
    def __init__(self):
        Exception.__init__(self, 'Given input is not supported by implementation')