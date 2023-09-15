"""Implementation of benchmark functions."""
import math
import numpy as np
import json

DOMAINS = json.load(open("./experiments/benchmarkexperiments/benchmarks/domains.txt"))
ADAPTED_DOMAINS = json.load(
    open("./experiments/benchmarkexperiments/benchmarks/adapted_domains.txt")
)
SAMPLED_DOMAINS = json.load(
    open("./experiments/benchmarkexperiments/benchmarks/sampled_domains.json")
)
MINIMA = json.load(open("./experiments/benchmarkexperiments/benchmarks/minima.txt"))
LENGTHSCALES = json.load(
    open("./experiments/benchmarkexperiments/benchmarks/lengthscales.txt")
)



def dewrap(x):
    return [x[:, i] for i in range(x.shape[1])]


def braninhelper(xlist):
    """Branin function.

    f(x1, x2) = a(x2 -bx2^2+cx1-r)^2 +s(1-t)cos(x1)+s

    :param [np.array[Float], np.array[Float]] xlist: [x1, x2]
    :return: y = f(x1, x2).
    :rtype: Float

    """
    x1, x2 = xlist[0], xlist[1]
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return y[:, None]


def branin(x):
    return braninhelper(dewrap(x))


def sixhumpcamelhelper(xlist):
    """SixHumpCamel function.

    :param [np.array[Float], np.array[Float]] xlist: [x1, x2]
    :return: y = f(x1, x2).
    :rtype: Float

    """
    x1, x2 = xlist[0], xlist[1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    y = term1 + term2 + term3
    return y[:, None]


def sixhumpcamel(x):
    return sixhumpcamelhelper(dewrap(x))


def bealehelper(xlist):
    """Beale function.

    :param [np.array[Float], np.array[Float]] xlist: [x1, x2]
    :return: y = f(x1, x2).
    :rtype: Float

    """
    x1, x2 = xlist[0], xlist[1]
    term1 = (1.5 - x1 + x1 * x2) ** 2
    term2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
    term3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
    y = term1 + term2 + term3
    return y[:, None]


def beale(x):
    return bealehelper(dewrap(x))


def bohachevsky_a_helper(xlist):
    """Bohachevsky A function.

    :param [np.array[Float], np.array[Float]] xlist: [x1, x2]
    :return: y = f(x1, x2).
    :rtype: Float

    """
    x1, x2 = xlist[0], xlist[1]
    term1 = x1 ** 2
    term2 = 2 * x2 ** 2
    term3 = -0.3 * np.cos(3 * np.pi * x1)
    term4 = -0.4 * np.cos(4 * np.pi * x2)
    y = term1 + term2 + term3 + term4 + 0.7
    return y[:, None]


def bohachevsky_a(x):
    return bohachevsky_a_helper(dewrap(x))


def bohachevsky_b_helper(xlist):
    """Bohachevsky B function.

    :param [np.array[Float], np.array[Float]] xlist: [x1, x2]
    :return: y = f(x1, x2).
    :rtype: Float

    """
    x1, x2 = xlist[0], xlist[1]
    term1 = x1 ** 2
    term2 = 2 * x2 ** 2
    term3 = -0.3 * np.cos(3 * np.pi * x1)
    term4 = np.cos(4 * np.pi * x2)
    y = term1 + term2 + term3 * term4 + 0.3
    return y[:, None]


def bohachevsky_b(x):
    return bohachevsky_b_helper(dewrap(x))


def bohachevsky_c_helper(xlist):
    """Bohachevsky C function.

    :param [np.array[Float], np.array[Float]] xlist: [x1, x2]
    :return: y = f(x1, x2).
    :rtype: Float

    """
    x1, x2 = xlist[0], xlist[1]
    term1 = x1 ** 2
    term2 = 2 * x2 ** 2
    term3 = -0.3 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2)
    y = term1 + term2 + term3 + 0.3
    return y[:, None]


def bohachevsky_c(x):
    return bohachevsky_c_helper(dewrap(x))


def rosenbrock2_helper(xlist):
    """Rosenbrock function.

    :param [np.array[Float], np.array[Float]] xlist: [x1, x2]
    :return: y = f(x1, x2).
    :rtype: Float

    """
    x1, x2 = xlist[0], xlist[1]
    y = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return y[:, None]


def rosenbrock2(x):
    return rosenbrock2_helper(dewrap(x))


def ackley2_helper(xlist):
    """Ackley function.

    :param [np.array[Float], np.array[Float]] xlist: [x1, x2]
    :return: y = f(x1, x2).
    :rtype: Float

    """
    x1, x2 = xlist[0], xlist[1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = np.sqrt((x1 ** 2 + x2 ** 2) / 2)
    term2 = (np.cos(c * x1) + np.cos(c * x2)) / 2
    y = -a * np.exp(-b * term1) - np.exp(term2) + a + np.exp(1)
    return y[:, None]


def ackley2(x):
    return ackley2_helper(dewrap(x))


def hartmann3_helper(xlist):
    x1, x2, x3 = xlist[0], xlist[1], xlist[2]
    alpha = np.asarray([1, 1.2, 3, 3.2])
    A = np.asarray([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    P = np.asarray(
        [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
    )
    P = 10 ** (-4) * P
    x = np.vstack([x1, x2, x3]).T

    def term_i(i):
        return alpha[i] * np.exp(-np.dot((x - P[i]) ** 2, A[i]))

    y = -np.sum([term_i(i) for i in range(4)], axis=0)
    return y[:, None]


def hartmann3(x):
    return hartmann3_helper(dewrap(x))


def trid4_helper(x):
    x = np.vstack(x).T
    term1 = np.sum([(x[:, i] - 1) ** 2 for i in range(4)], axis=0)
    term2 = np.sum([x[:, i] * x[:, i - 1] for i in range(1, 4)], axis=0)
    y = term1 - term2
    return y[:, None]


def trid4(x):
    return trid4_helper(dewrap(x))


def shekel_helper(x):
    x = np.vstack(x).T
    beta = np.asarray([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) / 10
    C = np.asarray(
        [
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        ]
    )

    def term_i(i):
        return beta[i] + np.sum((x - C[:, i]) ** 2, axis=1)

    y = -np.sum([1 / term_i(i) for i in range(10)], axis=0)
    return y[:, None]


def shekel(x):
    return shekel_helper(dewrap(x))


def dixonprice10_helper(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x
    term1 = (x1 - 1) ** 2
    term2 = np.sum([i * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, 10)], axis=0)
    y = term1 + term2
    return y[:, None]


def dixonprice10(x):
    return dixonprice10_helper(dewrap(x))


def benchmark_function(benchmark):
    return BENCHMARKS[benchmark][0], BENCHMARKS[benchmark][1]


####################################################################################################
# Benchmark functions in the right format for TURBO
####################################################################################################


class Branin:
    def __init__(self, lb, ub):
        self.dim = 2
        self.lb = lb
        self.ub = ub
        # self.lb = np.asarray([-5, 10])
        # self.ub = np.asarray([0, 15])

    def __call__(self, x):
        a, b, c, r, s, t = (
            1,
            5.1 / (4 * math.pi ** 2),
            5 / math.pi,
            6,
            10,
            1 / (8 * math.pi),
        )
        val = (
            a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
            + s * (1 - t) * np.cos(x[0])
            + s
        )
        return val


class SixHumpCamel:
    def __init__(self, lb, ub):
        self.dim = 2
        self.lb = lb
        self.ub = ub
        # self.lb = np.asarray([-3, 3])
        # self.ub = np.asarray([-2, 2])

    def __call__(self, x):
        val = (
            (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2
            + x[0] * x[1]
            + (-4 + 4 * x[1] ** 2) * x[1] ** 2
        )
        return val


class Ackley:
    def __init__(self, lb, ub):
        self.dim = 2
        self.lb = lb
        self.ub = ub
        # self.lb = -32.768 * np.ones(dim)
        # self.ub = 32.768 * np.ones(dim)

    def __call__(self, x):
        a, b, c = 10, 0.2, 2 * math.pi
        val = (
            -a * np.exp(-b * np.sqrt(1 / self.dim * np.sum(np.multiply(x, x))))
            - np.exp(1 / self.dim * np.sum(np.cos(c * x)))
            + a
            + np.exp(1)
        )
        return val


class DixonPrice:
    def __init__(self, lb, ub):
        self.dim = 10
        self.lb = lb
        self.ub = ub
        # self.lb = -10 * np.ones(dim)
        # self.ub = 10 * np.ones(dim)

    def __call__(self, x):
        index = np.arange(1, self.dim)
        term1 = 2 * np.multiply(x[1:], x[1:]) - x[:-1]
        term2 = np.multiply(term1, term1)
        val = (x[0] - 1) ** 2 + np.sum(np.multiply(index, term2))
        return val


class Beale:
    def __init__(self, lb, ub):
        self.dim = 2
        self.lb = lb
        self.ub = ub
        # self.lb = -4.5 * np.ones(dim)
        # self.ub = 4.5 * np.ones(dim)

    def __call__(self, x):
        term1 = 2 * np.multiply(x[1:], x[1:]) - x[:-1]
        term2 = np.multiply(term1, term1)
        val = (
            (1.5 - x[0] + x[0] * x[1]) ** 2
            + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        )
        return val


class BohachevskyA:
    def __init__(self, lb, ub):
        self.dim = 2
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        term1 = 0.3 * math.cos(3 * math.pi * x[0])
        term2 = 0.4 * math.cos(4 * math.pi * x[1])
        return x[0] ** 2 + 2 * x[1] ** 2 - term1 - term2 + 0.7


class BohachevskyB:
    def __init__(self, lb, ub):
        self.dim = 2
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        term1 = 0.3 * math.cos(3 * math.pi * x[0]) * math.cos(4 * math.pi * x[1])
        return x[0] ** 2 + 2 * x[1] ** 2 - term1 + 0.3


class BohachevskyC:
    def __init__(self, lb, ub):
        self.dim = 2
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        term1 = 0.3 * math.cos(3 * math.pi * x[0] + 4 * math.pi * x[1])
        return x[0] ** 2 + 2 * x[1] ** 2 - term1 + 0.3


class Rosenbrock:
    def __init__(self, lb, ub):
        self.dim = 2
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2


class Hartmann3:
    def __init__(self, lb, ub):
        self.dim = 3
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        alpha = np.asarray([1, 1.2, 3, 3.2])
        A = np.asarray([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        P = np.asarray(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        )
        P = 10 ** (-4) * P

        def term_i(i):
            return alpha[i] * np.exp(-np.dot((x - P[i]) ** 2, A[i]))

        y = -np.sum([term_i(i) for i in range(4)], axis=0)
        return y


class Trid4:
    def __init__(self, lb, ub):
        self.dim = 4
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        term1 = (x[0] - 1) ** 2 + (x[1] - 1) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 2
        term2 = x[1] * x[0] + x[2] * x[1] + x[3] * x[2]
        return term1 - term2


class Shekel4:
    def __init__(self, lb, ub):
        self.dim = 4
        self.lb = lb
        self.ub = ub

    def __call__(self, x):
        beta = np.asarray([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) / 10
        C = np.asarray(
            [
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            ]
        )

        def term_i(i):
            return beta[i] + np.sum((x - C[:, i]) ** 2, axis=0)

        y = -np.sum([1 / term_i(i) for i in range(10)], axis=0)
        return y



def calc_cut_point(cut_point, dim, input_range, sample):
    total_distance = input_range[dim][1] - input_range[dim][0]
    cut_distance = cut_point - input_range[dim][0]
    distance = cut_distance / total_distance
    return np.min(
        [
            np.max([0, int(distance * sample.shape[dim])]),
            sample.shape[dim] - 1,
        ]
    )


def get_function_value(sample, x, inputrange):
    indices = np.array(
        [
            calc_cut_point(cut_point, dim, inputrange, sample)
            for dim, cut_point in enumerate(x)
        ]
    )
    x_ = np.squeeze(indices)
    if len(sample.shape) > 1:
        return sample[tuple(x_)]
    else:
        return sample[x_]


class Sample:
    def __init__(self, lb, ub, sample, inputrange, params):
        self.dim = len(inputrange)
        self.lb = lb
        self.ub = ub
        self.sample = sample
        self.inputrange = inputrange
        self.params = params

    def __call__(self, x):
        if self.params.max_or_min == "max":
            return -get_function_value(self.sample, x, self.inputrange)
        return get_function_value(self.sample, x, self.inputrange)


class DirectSample:
    def __init__(self, lb, ub, sample, inputrange, params):
        self.dim = len(inputrange)
        self.lb = lb
        self.ub = ub
        self.sample = sample
        self.inputrange = inputrange
        self.params = params

    def __call__(self, x):
        if self.params.max_or_min == "max":
            return -get_function_value(self.sample, x, self.inputrange)
        return get_function_value(self.sample, x, self.inputrange)


def get_turbo_benchmark_function(params):
    lower_bounds = np.asarray([l for l, u in params.input_range])
    upper_bounds = np.asarray([u for l, u in params.input_range])
    if params.benchmark == "branin":
        return Branin(lower_bounds, upper_bounds)
    elif params.benchmark == "sixhumpcamel":
        return SixHumpCamel(lower_bounds, upper_bounds)
    elif params.benchmark == "ackley2":
        return Ackley(lower_bounds, upper_bounds)
    elif params.benchmark == "dixonprice10":
        return DixonPrice(lower_bounds, upper_bounds)
    elif params.benchmark == "beale":
        return Beale(lower_bounds, upper_bounds)
    elif params.benchmark == "bohachevsky_a":
        return BohachevskyA(lower_bounds, upper_bounds)
    elif params.benchmark == "bohachevsky_b":
        return BohachevskyB(lower_bounds, upper_bounds)
    elif params.benchmark == "bohachevsky_c":
        return BohachevskyC(lower_bounds, upper_bounds)
    elif params.benchmark == "rosenbrock2":
        return Rosenbrock(lower_bounds, upper_bounds)
    elif params.benchmark == "hartmann3":
        return Hartmann3(lower_bounds, upper_bounds)
    elif params.benchmark == "trid4":
        return Trid4(lower_bounds, upper_bounds)
    elif params.benchmark == "shekel":
        return Shekel4(lower_bounds, upper_bounds)
    else:
        raise NotImplementedError()


####################################################################################################


BENCHMARKS = {
    "branin": (braninhelper, branin),
    "sixhumpcamel": (sixhumpcamelhelper, sixhumpcamel),
    "beale": (bealehelper, beale),
    "bohachevsky_a": (bohachevsky_a_helper, bohachevsky_a),
    "bohachevsky_b": (bohachevsky_b_helper, bohachevsky_b),
    "bohachevsky_c": (bohachevsky_c_helper, bohachevsky_c),
    "rosenbrock2": (rosenbrock2_helper, rosenbrock2),
    "ackley2": (ackley2_helper, ackley2),
    "hartmann3": (hartmann3_helper, hartmann3),
    "trid4": (trid4_helper, trid4),
    "shekel": (shekel_helper, shekel),
    "dixonprice10": (dixonprice10_helper, dixonprice10),
}
