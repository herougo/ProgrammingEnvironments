import matplotlib.pyplot as plt

class ValHistory:
    def __init__(self, names_list):
        self.names = names_list
        self.n_names = len(self.names)
        self.values = []
    def add(self, val_list):
        assert(len(val_list) == self.n_names)
        self.values.append(val_list)
    def plot(self):
        if len(self.values) == 0:
            return
        
        self.values = zip(*self.values)
        
        for i in range(self.n_names):
            plt.subplot(self.n_names , 1, i + 1)
            plt.plot(self.values[i], 'bo')
            plt.plot(self.values[i], 'b')
            plt.title(self.names[i])
        plt.show()

        self.values = zip(*self.values)