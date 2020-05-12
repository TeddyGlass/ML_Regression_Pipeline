import pickle


def load(path):
    with open('{}'.format(path), 'rb') as f:
        item = pickle.load(f)
    return item


def save(path, item):
    with open('{}'.format(path), 'wb') as f:
        pickle.dump(item, f)
