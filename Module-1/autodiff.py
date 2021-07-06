from __future__ import annotations
import uuid


def wrap_tuple(x: float | tuple[float]) -> tuple[float]:
    """ Convierte a tuplas """
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x: float | tuple[float]) -> float:
    """ Si tiene un solo elemento, retorna ese elemento, caso contrario retornas la tupla """
    if len(x) == 1:
        return x[0]
    return x


class Variable:
    """
    Attributes:
        history (:class:`History`) : the Function calls that created this variable or None if constant
        derivative (number): the derivative f  with respect to this variable
        name (string) : an optional name for debugging
    """

    def __init__(self, history: History, name: str = None):
        assert history is None or isinstance(history, History), history
        self.history = history
        self._derivative: float = None

        # For debugging can have a name.
        if name is not None:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

    def requires_grad_(self, val: bool):
        self.history = History(None, None, None)

    def backward(self, d_output: float = None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(VariableWithDeriv(self, d_output))

    @property
    def derivative(self):
        return self._derivative

    # IGNORE
    def __hash__(self):
        return hash(self._name)

    def _add_deriv(self, val: float):
        assert self.history.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += val

    def zero_grad_(self):
        self._derivative = self.zeros()

    def __radd__(self, b: Variable):
        return self + b

    def __rmul__(self, b: Variable):
        return self * b

    def zeros(self):
        return 0.0

    def expand(self, x: float) -> float:
        return x

    # IGNORE


class Context:
    """
    Context class is used by.
    """

    def __init__(self, no_grad: bool = False):
        self._saved_values: tuple[float, ...] = None
        self.no_grad = no_grad

    def save_for_backward(self, *values: float) -> None:
        if self.no_grad:
            return
        self._saved_values = values

    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)


class History:
    """
    `History` stores all of the `Function` operations that were used to
    construct an autodiff object.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last function that was called.
        ctx (:class:`Context`): The context for that function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.
    """

    def __init__(self, last_fn: FunctionBase = None, ctx: Context = None, inputs: tuple[Variable] = None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs

    def is_leaf(self) -> bool:
        return self.last_fn is None

    def backprop_step(self, d_output: float):
        # d_output inicia con 1.0 y va acumulando el resultado
        return self.last_fn.chain_rule(self.ctx, self.inputs, d_output)


class VariableWithDeriv:
    "Holder for a variable with it derivative."

    def __init__(self, variable: Variable, deriv: float):
        # deriv es el valor que se acumula de la derivada
        self.variable = variable
        self.deriv = variable.expand(deriv)


class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history: History) -> Variable:
        pass

    @classmethod
    def apply(cls, *vals: Variable):
        # un classmethod retorna el metodo de una clase de la funcion dada cuando es llamada, por ejemplo en la clase Scalar le estamos estamos pasando self, por lo tanto se refiere a Scalar el cual tiene los metodos forward y backward
        raw_vals = []
        need_grad = False
        for v in vals:
            if isinstance(v, Variable):
                if v.history is not None:
                    need_grad = True
                raw_vals.append(v.get_data())
            else:
                raw_vals.append(v)
        ctx = Context(not need_grad)
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)

    @classmethod
    def chain_rule(cls, ctx: Context, inputs: tuple[Variable], d_output: float) -> list[VariableWithDeriv]:
        """
        Implement the derivative chain-rule.

        Args:
            cls (:class:`FunctionBase`): The Function
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of :class:`VariableWithDeriv`: A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)

        """
        # TODO: Implement for Task 1.3.

        # Ejemplo: cls Add, ctx almacena v3 y v4 para el backward, adjoint_v5 = d_output = 1
        # v5(v3, v4) = v3 + v4
        # adjoint_v4 = dv5/dv4 * adjoint_v5
        # adjoint_v3 = dv5/dv3 * adjoint_v5

        # adjoint_v = (adjoint_v3, adjoint_v4)
        adjoint_v: float | tuple[float] = cls.backward(ctx, d_output)

        # si fuera un solo numero lo convertimos a tupla para ponerlo junto con el zip
        adjoint_v = wrap_tuple(adjoint_v)

        # VariableWithDeriv(vi, adjoint_vi)
        v_adjoint_v: list[VariableWithDeriv] = []
        for vi, adjoint_vi in zip(inputs, adjoint_v):
            if not is_constant(vi):
                v_adjoint_v.append(VariableWithDeriv(vi, adjoint_vi))
        return v_adjoint_v


def is_leaf(val: Variable):
    return isinstance(val, Variable) and val.history.is_leaf()


def is_constant(val: Variable):
    return not isinstance(val, Variable) or val.history is None


def backpropagate(final_variable_with_deriv: VariableWithDeriv):
    """
    Runs a breadth-first search on the computation graph in order to
    backpropagate derivatives to the leaves.

    See :doc:`backpropagate` for details on the algorithm

    Args:
       final_variable_with_deriv (:class:`VariableWithDeriv`): The final variable
           and its derivative that we want to propagate backward to the leaves.
    """
    # TODO: Implement for Task 1.4.

    # Step 0) Initialize a queue with the final Variable-derivative
    queue: list[VariableWithDeriv] = [final_variable_with_deriv]
    # al comienzo es vn = f, adjoint_vn = 1
    while len(queue) > 0:
        # Step 1) While the queue is not empty, pull a Variable+derivative from the queue
        v_adjoint_v = queue.pop(0)
        vi, adjoint_vi = v_adjoint_v.variable, v_adjoint_v.deriv

        if is_leaf(vi):
            # Step 1.a) if the Variable is a leaf, add its final derivative (_add_deriv) and loop to (1)
            vi._add_deriv(adjoint_vi)
        else:
            # Step 1.b) if the Variable is not a leaf,
            # 1) call .chain_rule on the last function that created it with derivative as d_out
            v_adjoint_v_list = vi.history.last_fn.chain_rule(ctx=vi.history.ctx,
                                                             inputs=vi.history.inputs,
                                                             d_output=adjoint_vi)
            # 2) loop through all the Variables+derivative produced by the chain rule (removing constants)
            for vi_adjoint_vi in v_adjoint_v_list:
                vi, adjoint_vi = vi_adjoint_vi.variable, vi_adjoint_vi.deriv
                if not is_constant(vi):
                    # 4) add to the queue
                    queue.append(vi_adjoint_vi)
