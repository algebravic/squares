"""
Use simulated annealing to find the max digit sum of a square
"""
from typing import List, Tuple, Iterable, Callable
import numpy as np
from numpy import base_repr
from math import sqrt
from random import randint, random
from simanneal import Annealer

def mysqrt(num: int) -> int:
    """
    Use Newton's method to find a square root.
    Start with int(sqrt(num)).
    """

    probe = int(sqrt(num))
    while True:
        nxt = (num // probe + probe) // 2
        if abs(nxt- probe) <= 1:
            return min(nxt, probe)
        probe = nxt

def square_distance(num: int) -> np.float64:
    """
    Normalized distance from a square.
    """
    small = mysqrt(num)
    bot = small ** 2
    # 1 - abs(2 * v/d - 1) = (d - abs(2 * v - d)) / d
    val = 2 * (num  - bot) - 2 * small - 1
    return (2 * small + 1 - abs(val)) / (2 * small + 1)

def squareit(num: int):
    """
    Squaring
    """
    return num ** 2

def ident(num: int):
    """
    Identity
    """
    return num

class Squares(Annealer):
    """Use Annealing to find a n digit number
    which is a square and has large digit sum

    For general weight we want a function f : {0, ..., b-1}
    which is non-decreasing, positive on positive numbers
    and f(0) = 0 and whose sum is 1.

    The energy contribution will then be sum_i (f(b-1 - x_i))
    where x_i are the digits.

    """
    
    def __init__(self, num: int,
                 base: int = 10,
                 wfun: Callable[[int], float] = squareit,
                 cutoff: float = 1000.0,
                 mult: float = 10.0):

        self._num = num # Number of Digits
        self._base = base
        self.state = ([randint(0, base - 1) for _ in range(num - 1)]
                      + [randint(1, base - 1)])
        self._mult = mult
        self._weights = np.array(list(map(wfun, np.arange(self._base-1, -1, -1))))
        self._weights = self._weights / (self._num * self._weights[0])
        self._cutoff = cutoff

    @property
    def _objective(self):
        return self._weights[self.state].sum()

    @property
    def _value(self):
        return sum(val * self._base ** ind for ind, val in enumerate(self.state))

    @property
    def _delta(self):
        return square_distance(self._value)

    def move(self):
        """Make a move. Pick a digit position and then a digit"""
        where = randint(0, self._num - (1 if self._base > 2 else 2))
        old = self.state[where]
        bot = 1 if where == self._num - 1 else 0
        which = randint(bot, self._base - 2)
        if which >= old:
            which += 1
        self.state[where] = which

    def energy(self):
        """Energy value"""
        incr = self._delta
        if incr > 0.0:
            incr = self._mult * max(1.0, incr * self._cutoff)
        return self._objective + incr

def _tolist(val: int, base: int) -> List[int]:
    """
    Convert a value to a list of int
    """
    alphanum = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return list(map(alphanum.find, base_repr(val, base = base)))

class FSquares(Annealer):
    """
    Use annealing to find an n digit number
    which is the square and has a large digit sum.
    Only using feasible numbers.
    """

    def __init__(self,
                 num: int,
                 base: int = 10,
                 hybrid: float = 0.1):


        self._num = num
        self._base = base
        self._hybrid = hybrid
        bot = mysqrt(int('1' + (self._num - 1) * '0', base = self._base)) + 1
        top = mysqrt(int('1' + self._num * '0', base = self._base)) - 1
        self._bot = _tolist(bot, self._base)
        self._top = _tolist(top, self._base)
        self._max = self._num * (self._base - 1)

        self._hnum = len(self._bot)
        self.state = self._top.copy()


    def _opt_move(self):
        """
        Pick a position, and search for the digit giving
        the best score.
        """
        for _ in range(self._hnum):
            where = randint(0, self._hnum - 1)
            # Be careful to stay within bounds.
            best = self.energy()
            old = self.state[where]
            loc = old
            for dig in range(self._base):
                self.state[where] = dig
                if self.state < self._bot or self.state > self._top:
                    continue
                value = self.energy()
                if value < best:
                    loc = dig
                    best = value
            if loc != old:
                self.state[where] = loc
                return
            self.state[where] = old

    def _old_move(self):
        """ Make the annealing move """
        where = randint(0, self._hnum - 1)
        old = self.state[where]
        which = randint(0, self._base - 1)
        self.state[where] = which
        if self.state < self._bot or self.state > self._top:
            self.state[where] = old

    def move(self):
        """ Annealing move"""

        if random() < self._hybrid:
            self._opt_move()
        else:
            self._old_move()
            
    def _valx(self, number: List[int]):
        """
        The list of digits of the square.
        """
        the_sqrt = int(''.join(map(str, number)), base = self._base)
        return _tolist(the_sqrt ** 2, self._base)

    @property
    def _value(self):
        """
        The value of the state
        """
        return self._valx(self.state)

    def energy(self):
        """
        The energy.
        """
        return self._max - sum(self._value)
