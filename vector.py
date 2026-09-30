import math


class Vector:
    def __init__(self, entries):
        self.entries = [float(e) for e in entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    def __iter__(self):
        return iter(self.entries)

    def __repr__(self):
        return f"Vector({self.entries})"

    def __eq__(self, other):
        return isinstance(other, Vector) and self.entries == other.entries

    def __add__(self, other):
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length")
        return Vector([a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length")
        return Vector([a - b for a, b in zip(self, other)])

    def __neg__(self):
        return Vector([-a for a in self])

    def __mul__(self, scalar):
        return Vector([scalar * a for a in self])

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def dot(self, other):
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length")
        return sum(a * b for a, b in zip(self, other))

    def mean(self):
        if len(self) == 0:
            raise ValueError("Mean of an empty vector is undefined")
        return sum(self.entries) / len(self)

    def demean(self):
        m = self.mean()
        return Vector([a - m for a in self])

    def std(self):
        d = self.demean()
        return math.sqrt(sum(a * a for a in d) / len(d))
