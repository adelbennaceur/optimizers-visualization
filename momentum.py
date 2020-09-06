


def init_momentum():
    pass
def momentum():
    pass


class RMSprop(object):

    def __init__(self, x,y ,params):
        self.x = x
        self.y = y
        self.params = params
        self.local = locals()
            
    def optimize(self):
        pass


    def get_config(self):
        return self.local


if __name__ == '__main__':
    rms = RMSprop(10 , 11, [13,23,32])
    print(rms.get_config())
