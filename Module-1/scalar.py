from __future__ import annotations

from numpy.lib.function_base import copy
from .autodiff import Context, FunctionBase, Variable, History
from . import operators
import numpy as np
from collections.abc import Callable


# Task 1.1
# Derivatives


def central_difference(f: Callable[[tuple[float, ...]], float], *vals: float, arg=0, epsilon=1e-6):
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
       f : arbitrary function from n-scalar args to one value
       *vals (floats): n-float values :math:`x_0 \ldots x_{n-1}`
       arg (int): the number :math:`i` of the arg to compute the derivative
       epsilon (float): a small constant

    Returns:
       float : An approximation of :math:`f'_i(x_0, \ldots, x_{n-1})`
    """
    valsh = np.copy(vals)*1.0
    vals_h = np.copy(vals)*1.0

    valsh[arg] = valsh[arg] + epsilon
    vals_h[arg] = vals_h[arg] - epsilon

    # TODO: Implement for Task 1.1.
    return (f(*valsh) - f(*vals_h)) / (2 * epsilon)

# Task 1.2 and 1.4
# Scalar Forward and Backward


class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.

    """

    def __init__(self, v, back=History(), name: str = None):
        super().__init__(back, name=name)
        self.data = float(v)

    def __repr__(self):
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return Mul.apply(self, b)

    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))

    def __add__(self, b):
        # TODO: Implement for Task 1.2.
        return Add.apply(self, b)

    def __lt__(self, b):
        # TODO: Implement for Task 1.2.
        return LT.apply(self, b)

    def __gt__(self, b):
        # TODO: Implement for Task 1.2.
        return Neg.apply(LT.apply(Neg.apply(self), Neg.apply(b)))

    def __sub__(self, b):
        # TODO: Implement for Task 1.2.
        return Add.apply(self, Neg.apply(b))

    def __neg__(self):
        # TODO: Implement for Task 1.2.
        return Neg.apply(self)

    def log(self):
        # TODO: Implement for Task 1.2.
        return Log.apply(self)

    def exp(self):
        # TODO: Implement for Task 1.2.
        return Exp.apply(self)

    def sigmoid(self):
        # TODO: Implement for Task 1.2.
        return Sigmoid.apply(self)

    def relu(self):
        # TODO: Implement for Task 1.2.
        return ReLU.apply(self)

    def get_data(self):
        # The wrap value is stored in data attribute
        return self.data


class ScalarFunction(FunctionBase):
    "A function that processes and produces Scalar variables."

    @staticmethod
    def forward(ctx: Context, *inputs: float) -> float:
        """Args:

           ctx (:class:`Context`): A special container object to save
                                   any information that may be needed for the call to backward.
           *inputs (list of numbers): Numerical arguments.

        Returns:
            number : The computation of the function :math:`f`
        """
        pass

    @staticmethod
    def backward(ctx: Context, d_out: float) -> float | tuple[float]:
        """
        Args:
            ctx (Context): A special container object holding any information saved during in the corresponding `forward` call.
            d_out (number):
        Returns:
            numbers : The computation of the derivative function :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.
        """
        pass

    # checks.
    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a


# Examples
class Add(ScalarFunction):
    "Addition function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> tuple[float]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a = ctx.saved_values
        return operators.log_back(a, d_output)


class LT(ScalarFunction):
    "Less-than function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return 0.0


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward((a, b))
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> tuple[float]:
        # TODO: Implement for Task 1.4.
        a, b = ctx.saved_values
        return operators.mul(b, d_output), operators.mul(a, d_output)


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        a = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        a = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        a = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float):
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        # TODO: Implement for Task 1.4.
        a = ctx.saved_values
        return operators.mul(d_output, operators.exp(a))


def derivative_check(f: Callable, *scalars: Variable):

    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    out.backward()

    vals = [v for v in scalars]

    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        print(x.derivative, check)
        np.testing.assert_allclose(x.derivative, check.data, 1e-2, 1e-2)
