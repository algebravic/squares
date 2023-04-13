"""
Use MaxSat to find the n digit decimal number which is a perfect square
with maximal digit sum.

To enforce being a perfect square, we use the following result:

If a is a positive integer, it's a perfect square if and only if
a mod N is square with N >a^2.  We can enforce this by having N
the product of 8 and small primes.

We can use Max Sat.

Specifically we have variables d[i,k], for k in range(10), and i in range(n).
These indicate the digits.  We want to maximize
sum_{i in range(n)} sum_{k in range(1, 10)} k * d[i,k]

To make it really an n digit number we need ~ d[0,0]

We also have ExactlyOne([d[i,k] for k in range(10)]) for all i.

For each modulus, q, we have f[q,i,k] which is true if and only if
sum_{j=0 to i} sum_{l in range(10)} l * d[j,l] * 10**j == k mod q.

Then f[q,n,k] will say that the sum is k mod q.  If S is the set of quadratic
non-residues, then we want ~f[q,n,k] for each k in S.

For fixed q, then number of f[q,i,k] is q*(n+1).  And the
sum of all primes q <= 2 log_2(10) n is about 2 log_2(10) n**2.  So the number
of variables is around n ** 3.

For each modulus, the number of clauses is n+1 times the quantity below.

[It looks like q.  For each k1 in range(q) we have.
f[q,j-1,k1] AND d[j,l] ==> f[q,j, k1 + 10 ** j * l mod q].

We initialize ~f[q,-1,k] for k != 0, and f[q,-1,0].
"""
from typing import Iterable, List, Tuple
from collections import Counter
from math import log, exp
import sympy
from numpy import base_repr
from pysat.formula import WCNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2, RC2Stratified
from .timeit import Timer

class MaxDigits:
    """
    Use a Max Sat solver for the max digits problem.
    """
    def __init__(self, num: int, base: int = 10, coeff: float = 1.5):

        self._num = num
        self._base = base
        self._pool = IDPool()
        # Use the bound B^(c * n * log(n))
        self._generate_moduli(coeff * num * log(num) * log(base))
        print(f"Largest modulus = {self._moduli[-1]}")
        print(f"Sum of moduli = {sum(self._moduli)}")
        self._end_sums = {}

    def _generate_moduli(self, bound: float):
        """
        Generate the moduli needed.
        """
        self._moduli = [8]
        log_prod = log(8)
        prim = 2
        while log_prod < bound:
            prim = sympy.nextprime(prim)
            log_prod += log(prim)
            self._moduli.append(prim)

    def _addition_data(self, modulus: int) -> Iterable[List[int]]:
        """
        Addition mod p
        """
        prefix = ('f', modulus)
        power = 1
        for ind in range(self._num):
            for pdig in range(modulus):
                prev_f = self._pool.id(prefix + (ind - 1, pdig))
                for ddig in range(self._base):
                    this_d = self._pool.id(('d', ind, ddig))
                    this_f = self._pool.id(
                        prefix + (ind, (pdig + power * ddig) % modulus))
                    yield [- prev_f, - this_d, this_f]
            power = (self._base * power) % modulus
        self._end_sums[modulus] = prefix + (self._num - 1)

    def _digit_data(self,
                    prefix: Tuple[str, int, ...],
                    rbase: int) -> Iterable[List[int]]:
        """
        CNF constraining the digit variables.
        """
        yield from CardEnc.equals(lits = [self._pool.id(prefix + (_,))
                                          for _ in range(rbase)],
                                  bound = 1,
                                  encoding = EncType.ladder,
                                  vpool = self._pool)

    def _moduli_data(self, modulus: int) -> Iterable[List[int]]:
        """
        CNF encoding the sieve variables.
        """
        # Empty sum is 0
        prefix = ('f', modulus)
        yield [self._pool.id(prefix + (-1, 0))]
        for ind in range(-1, self._num):
            yield from self._digit_data(prefix + (ind,), modulus)

        square_mod = {(_ ** 2) % modulus for _ in range(modulus)}
        nonsquares = set(range(modulus)).difference(square_mod)
        # yield [self._pool.id(('f', modulus, self._num - 1, elt))
        #        for elt in square_mod]
        result = self._end_sums[modulus]
        yield from (
            [-self._pool.id(result + (elt,))] for elt in nonsquares)

    def model(self) -> Iterable[List[int]]:
        """
        The model for the max digits problem.
        """
        for ind in range(self._num):
            yield from self._digit_data(('d', ind), self._base)
        # Leading digit is not 0
        yield [-self._pool.id(('d', self._num - 1, 0))]
        for modulus in self._moduli:
            yield from self._moduli_data(modulus)
            yield from self._addition_data(modulus)

    def _extract(self, soln: List[int] | None) -> List[int] | None:
        """
        Extract the solution
        """
        if soln is None:
            return None

        pos = [self._pool.obj(_) for _ in soln if _ > 0]
        values = sorted([_[1:] for _ in pos if isinstance(_, tuple)
                         and _[0] == 'd'])
        return [_[1] for _ in values]

    def solve(self, **kwds) -> int:
        """
        Run the Max Sat solver and extract the answer
        """
        cnf = WCNF()
        with Timer('model'):
            cnf.extend(self.model()) # hard constraints

        print(f"Total clauses = {len(cnf.hard)}")
        for ind in range(self._num):
            for val in range(1, self._base):
                cnf.append([self._pool.id(('d', ind, val))], weight = val)

        max_solver = RC2(cnf, **kwds)
        digits = self._extract(max_solver.compute())
        print(f"Time = {max_solver.oracle_time()}")
        print(f"Average of digits = {sum(digits) / self._num}")
        value = sum(_[1] * self._base ** _[0] for _ in enumerate(digits))
        sqrt_val = int(sympy.sqrt(value))
        if sqrt_val ** 2 != value:
            print("Failure.  Make coeff larger")
            return value
        print(f"{(base_repr(sqrt_val, base = self._base))} ** 2"
              + f" = {base_repr(value, base = self._base)}")
        return ''.join(map(str, reversed(digits))) + f"_{self._base}"
