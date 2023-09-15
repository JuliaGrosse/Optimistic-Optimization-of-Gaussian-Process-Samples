"""Definition of some kernel functions."""
import numpy as np
import GPy
import probnum


def get_probnum_kernel(params):
    """Select the right probnum kernel based on experimental configuration.

    :param Parameters params: Experimental configuration.
    :return: kernel
    :rtype: GPy kernel object

    """
    for i, name in enumerate(params.kernelname):
        if name == "linear":
            probnum_kernel = probnum.randprocs.kernels.Linear(input_shape=(params.dim))
        elif name == "matern":
            assert params.variance[i] == 1
            probnum_kernel = probnum.randprocs.kernels.Matern(
                input_shape=(params.dim), lengthscale=params.lengthscale[i], nu=1.5
            )
        elif name == "polynomial":
            probnum_kernel = probnum.randprocs.kernels.Polynomial(
                input_shape=(params.dim), constant=0, exponent=2
            )
        elif name == "squaredexponential":
            assert params.variance[i] == 1
            probnum_kernel = probnum.randprocs.kernels.ExpQuad(
                input_shape=(params.dim), lengthscale=params.lengthscale[i]
            )
        elif name == "wiener":
            raise NotImplementedError()
        else:
            print("No valid kernelname.")
            raise NotImplementedError()
    return probnum_kernel


def get_gpy_kernel(params):
    """Select the right GPY kernel based on experimental configuration.

    :param Parameters params: Experimental configuration.
    :return: kernel
    :rtype: GPy kernel object

    """
    gpy_kernels = []
    for i, name in enumerate(params.kernelname):
        if name == "linear":
            gp_kernel = GPy.kern.src.linear.Linear(
                params.dim, variances=[params.variance[i]]
            )
        elif name == "matern":
            gp_kernel = GPy.kern.Matern32(
                input_dim=params.dim,
                lengthscale=params.lengthscale[i],
                variance=params.variance[i],
            )
        elif name == "polynomial":
            gp_kernel = GPy.kern.src.poly.Poly(
                input_dim=params.dim,
                order=2,
                bias=params.c[i],
                variance=params.variance[i],
            )
        elif name == "squaredexponential":
            gp_kernel = GPy.kern.src.rbf.RBF(
                params.dim,
                variance=params.variance[i],
                lengthscale=params.lengthscale[i],
            )
        elif name == "wiener":
            gp_kernel = GPy.kern.src.brownian.Brownian()
        else:
            print("No valid kernelname.")
            gp_kernel = None
        if not gp_kernel is None:
            gpy_kernels.append(gp_kernel)
    return GPy.kern.src.add.Add(gpy_kernels)


def get_my_kernel(params):
    """Select the right kernel based on experimental configuration
    (Sum over all specified kernels).

    :param Parameters params: Experimental configuration.
    :return: Kernel function.
    :rtype: Function

    """
    kernels = []
    for i in range(len(params.kernelname)):
        kernel = get_my_kernel_by_name(params, i)
        kernels.append(kernel)
    return sum_kernels(kernels)


def get_my_kernel_by_name(params, i):
    """Select the right kernel based on experimental configuration.

    :param Parameters params: Experimental configuration.
    :param Int i: Index for kernel.
    :return: Kernel function.
    :rtype: Function

    """
    name = params.kernelname[i]
    if name == "linear":
        kernel = linear_kernel(params, i)
    elif name == "matern":
        kernel = matern_kernel(params, i)
    elif name == "polynomial":
        kernel = polynomial_kernel(params, i)
    elif name == "squaredexponential":
        kernel = se_kernel(params, i)
    elif name == "wiener":
        kernel = wiener_kernel(params, i)
    return kernel


def sum_kernels(kernel_functions):
    """Sum of kernels.

    :param [functions] kernel_functions: Kernel functions.
    :return: Sum kernel function.
    :rtype: Function.

    """

    def sum_kernel(x, y):
        return np.sum([kernel(x, y) for kernel in kernel_functions])

    return sum_kernel


def linear_kernel(params, i):
    """Linear kernel

    k(x1, x2) = var * x1^Tx2

    :param Parameters params: Experimental configuration.
    :param Int i: Index for parameters of kernel function.
    :return: Kernel function.
    :rtype: Function

    """

    def cov(x1, x2):
        return params.variance[i] * np.dot(x1, x2)

    return cov


def matern_kernel(params, i):
    """Matern kernel with nu = 3/2

    k(x1, x2) = var * (1 + sqrt{3*(x1-x2)^2}/l)*exp{-sqrt{3*(x1-x2)^2}/l}

    :param Parameters params: Experimental configuration.
    :param Int i: Index for parameters of kernel function.
    :return: Kernel function.
    :rtype: Function

    """

    def cov(x1, x2):
        return (
            params.variance[i]
            * (1 + (np.sqrt(3 * np.linalg.norm(x1 - x2) ** 2) / params.lengthscale[i]))
            * np.exp(-np.sqrt(3 * np.linalg.norm(x1 - x2) ** 2) / params.lengthscale[i])
        )

    return cov


def polynomial_kernel(params, i):
    """Polynomial kernel (2nd degree).

    k(x1, x2) = var * (x1^Tx2 + c)^2

    :param Parameters params: Experimental configuration.
    :param Int i: Index for parameters of kernel function.
    :return: Kernel function.
    :rtype: Function

    """

    def cov(x1, x2):
        return params.variance[i] * (np.dot(x1, x2) + params.c[i]) ** 2

    return cov


def se_kernel(params, i):
    """Squared exponential kernel

    k(x1, x2) = var * exp{- ||x1-x2||_2^2/2*l^2}

    :param Parameters params: Experimental configuration.
    :param Int i: Index for parameters of kernel function.
    :return: Kernel function.
    :rtype: Function

    """

    def cov(x1, x2):
        return params.variance[i] * np.exp(
            -np.linalg.norm(x1 - x2) ** 2 / (2 * params.lengthscale[i] ** 2)
        )

    return cov


def wiener_kernel(params, i):
    """Wiener kernel

    k(x1, x2) = max(0, min(x1, x2))

    :param Parameters params: Experimental configuration.
    :param Int i: Index for parameters of kernel function.
    :return: Kernel function.
    :rtype: Function

    """

    def cov(x1, x2):
        return np.max([0, np.min([x1, x2])])

    return cov
