import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

class XYZ:
    __slots__ = ['x', 'y', 'z']
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __repr__(self):
        return (f"({self.x}, {self.y}, {self.z})")
    def __eq__(self, other: 'XYZ'):
        return isinstance(other, XYZ) and self.x == other.x and self.y == other.y and self.z == other.z
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    def __add__(self, other: 'XYZ'):
        if type(other.x) == type(self.x):
            return XYZ(self.x+other.x, self.y+other.y, self.z+other.z)
        else:
            raise TypeError
    def __sub__(self, other: 'XYZ'):
        if type(other.x) == type(self.x):
            return XYZ(self.x-other.x, self.y-other.y, self.z-other.z)
        else:
            raise TypeError
    def __mul__(self, scaler):
        T = type(self.x)
        return XYZ(
            T(self.x * scaler),
            T(self.y * scaler),
            T(self.z * scaler)
        )

    def astype(self, T):
        return XYZ(
            T(self.x),
            T(self.y),
            T(self.z),
        )

    def norm(self, l=2):
        return (self.x**l + self.y**2 + self.z**2)**(1/l)
    def norm_infinity(self):
        re = 0
        if self.x != 0: re += 1
        if self.y != 0: re += 1
        if self.z != 0: re += 1
        return re