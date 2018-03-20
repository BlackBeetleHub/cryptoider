import json
import string


class CeaserData:
    indexes = []
    changes = []
    details = []

    def __init__(self):
        self.indexes = []
        self.changes = []
        self.details = []

    def add_step(self, index, state, details):
        self.indexes.append(index)
        self.changes.append(state)
        self.details.append(details)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class Ceaser:

    info = CeaserData()

    def ceaser(self, array, move):
        res = []
        i = 0
        for x in array:
            n_char = ord(x)
            res.append(chr(n_char + move))
            self.info.add_step(i, res[:],
                               {
                                   'old_char_code': n_char,
                                   'move': move,
                                   'new_char_code': n_char + move
                               })
            i +=1
        return res