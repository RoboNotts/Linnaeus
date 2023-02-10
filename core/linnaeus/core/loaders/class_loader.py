class ClassLoader():
    def __init__(self, path):
        with open(path, 'r') as f:
            self.classes = f.read().splitlines()
    
    def __getitem__(self, index):
        return self.classes[index]
    
    def index(self, key):
        return self.classes.index(key)