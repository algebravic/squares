"""
Use ADMM to attack the squares problem.

According to ADMM and Distributed by Boyd et. al. section 9.1

Minimize f(x) subject to x in S, where S is a non-convex set

x^* = arg min_x (f(x) + (mu/2) ||x - z + u||^2
z^* = Pi_S (x^* + u)
u^* = u + x^* - z^*

where Pi_S is the projection onto S.

Here f(x) = ||B - x||_1 + R(x)
where B is the vector (9, ..., 9) (of dimension n)
and R(x) is the (set wise) indicator for [0, 9]^n.

S is the set of digits of squares.

There is an alternative, since the projection onto S
is unpleasant.

Replace x in the above right hand side by c^T x,
where c_i = 10^i.  Now u is a scalar, as is z.

The function on the right hand side of the first line
is separable, so the minimization is easy.

There's a practical question of what the initial value of u
should be.  Choose the initial value x at random,
the initial value of z to be the closest integer square.
A problem: abs(z-u) is going to be large. This will
tend to drive the minimization to extreme values.

Details: since the right side is separable, we deal with
each coordinate independently:

Two case:
(9 - x_i) + (mu/2) (c^T x)^2 + mu (u-z) c^T x + (mu/2) (z-u)^2

Ugh: The right hand side is not separable.

(9 - x_i) + (mu/ 2) (c^T x) x_i + mu (u-z) c_i x_i, the relevant part

We can deal with this by making a 'replica' for c^T x

Let w be the replica variable for c^T x.  The penalty will
be (mu/2) (||w - z + u||^2 + ||w - c^T x + v||^2)

An aside: Can we actually find the closest integral square
in the following sense:

Let S be the set of all the n vectors of coefficients of
n-digit integral squares.  Can we (even approximately) project
onto x?

sum_i (y_i - x_i)^2 such that sum_j x_j * 10^j = z^2

We have:

y = c^T x
z = y

[c_0, ..., c_{n-1}, 0, 0 ] [x_0]
[0, ...,         0, -1, 0] [x_1] 
[0, ...,         0, 0, -1]  ...
                           [x_{n-1}]
                           [y]
                           [w]

[c_0, ..., c_{n-1}, 0, -1 ] [x_0]
[0, ...,         0, -1, 1 ] [x_1] 
                            ...
                            [x_{n-1}]
                            [y]
                            [w]

L((x,y),w), lambda) = f(x) + R(y) + g(w)  + lambda_1 (c^T y - w)
                                          + lambda_2^T (x - y)
                                          + mu_1 / 2 ||c^T y - w||^2
                                          + mu_2 / 2 ||x - y||^2
(x^*,y^*) = arg min_{(x,y)} f(x) + R(y)  + lambda_1 (c^T y - w)
                                         + lambda_2^T (x - y)
                                         + mu_1 / 2 ||c^T y - w||^2
                                         + mu_2 / 2 ||x - y||^2
w^* = arg min_w g(w) + lambda_1 (c^T x^* - w)
                     + mu_1 / 2 ||c^T x^* - w||^2
                     + lambda_2 (y^* - w) + mu_2 / 2 ||y^* - w||^2
lambda_1^* = lambda_1 + mu_1 (c^T x^* - w^*)
lambda_2^* = lambda_2 + mu_2 (y^* - w^*) + mu_1 (c^T x^* - w^*)

f(x) = (B-1) * e - x

g(w) is indicator function of integer squares.
Solve: arg min_{z in Z} a z^4 + b z^2 = a(z^2+(b/(2a)))^2 + ...
R(x) is the indicator function of [0, B-1]^(n-1) x [1, B-1]

Let S be the set [0,B-1]^(n-1) x [1, B]
I_1(x,y) is the indicator function of x = y
I_2(x) is the indicator function of S

Can I not do this?
"""
import numpy as np

def pmat(num: int, base: int = 10) -> np.ndarray:
    """
    A power matrix.
    """
    pwr = base ** np.arange(num)
    zer = np.zeros((2, num))
    top = np.concatenate([pwr, [0, 0]]).reshape((1, -1))
    bot = np.concatenate([zer, - np.identity(2)], axis=1)
    return np.concatenate([top, bot], axis=0)
