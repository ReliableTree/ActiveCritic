class data_bank:
    def __init__(self):
        self.keys = {}

    def add_field(self, row:dict, id):
        row_hash = id
        for key in row:
            if key in self.keys:
                if row[key] in self.keys[key]:
                    self.keys[key][row[key]].add(row_hash)
                else:
                    self.keys[key][row[key]] = set([row_hash])
            else:
                self.keys[key] = {row[key] : set([row_hash])}
        self.keys[row_hash] = row

    def filter(self, key, value):
        hits = self.keys[key][value]
        return hits

    def intersect(self, hits1:set, hits2:set):
        hits = hits1.intersection(hits2)
        return [self.keys[hit] for hit in hits]


if __name__ == '__main__':
    row11 = {
        'name': 'hendrik',
    }
    row12 = {
        'name': 'malte',
    }

    row21 = {
        'age': 100,
    }

    row22 = {
        'age': 100,
    }

    db1 = data_bank()
    db1.add_field(row11, hash('hendrik'))
    db1.add_field(row12, hash('malte'))
    db2 = data_bank()
    db2.add_field(row21, hash('hendrik'))
    db2.add_field(row22, hash('malte'))
    print(db2.keys)
    h2 = db2.filter('age', 100)

    h1 = db1.filter('name', 'hendrik')
    print([db1.keys[h] for h in h2])
    print('hello')