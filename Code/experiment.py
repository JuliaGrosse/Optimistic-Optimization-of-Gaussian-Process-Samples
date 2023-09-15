"""Set up for experiments on benchmark functions and samples from the GP."""
import time
import sys
import random
import numpy as np

sys.path.append(".")
import utils
import kernel_functions
import benchmark_functions

# import warnings
# warnings.simplefilter('error', RuntimeWarning)

np.random.seed(28)


class Experiment:
    """Short summary.

    :param Parameters params: Description of parameter `params`.
    :attr np.meshgrid domain: Description of parameter `domain`.
    :attr np.array[Float] samples: Description of parameter `samples`.
    :attr GPy.kern kernel: Gpy.kern object for BO methods.
    :attr function mykernel: Kernel function for GP-OO method.
    :attr String benchmark: Name of the benchmark.
    :attr None minimum_value: None.
    :attr params:

    """

    def __init__(self, params):
        """Experiment.

        :param Parameters params: Experimental configuration.
        :return: None
        :rtype: None

        """
        self.params = params
        self.domain = np.meshgrid(
            *[
                np.arange(start, stop, params.range_length / params.step_size)
                for start, stop in params.input_range
            ],
            indexing="ij"
        )
        self.samples = None
        self.kernel = kernel_functions.get_gpy_kernel(params)
        self.mykernel = kernel_functions.get_my_kernel(params)
        self.benchmark = params.benchmark
        self.minimum_value = None
        self.params.minimum_value = None

    ### Basic functions
    def regret(self, y, sample):
        """Calculate simple regret.

        :param Float y: Observation.
        :param np.array[Float] sample: Function.
        :return: Simple regret.
        :rtype: Float

        """
        if self.params.benchmark == "groundtruth":
            if self.params.max_or_min == "max":
                return np.max(sample) - y
            return y - np.min(sample)
        return y

    def calc_cut_point(self, cut_point, dim):
        """Calculate the index for evaluation at cut_point along dim.

        :param Float cut_point: Description of parameter `cut_point`.
        :param Int dim: Axis of the cube along which to cut.
        :return: Index for evaluation.
        :rtype: Int

        """
        total_distance = (
            self.params.input_range[dim][1] - self.params.input_range[dim][0]
        )
        cut_distance = cut_point - self.params.input_range[dim][0]
        distance = cut_distance / total_distance
        return np.min(
            [
                np.max([0, int(distance * self.samples[0].shape[dim])]),
                self.samples[0].shape[dim] - 1,
            ]
        )

    def get_function_value(self, sample, x):
        """Get the value of the function/sample at x, i.e. f(x)

        :param np.array[Float] sample: GP sample/function.
        :param Float x: location for function evaluation.
        :return: f(x)
        :rtype: Float

        """
        indices = np.array(
            [self.calc_cut_point(cut_point, dim) for dim, cut_point in enumerate(x)]
        )
        x_ = np.squeeze(indices)
        if len(sample.shape) > 1:
            return sample[tuple(x_)]
        return sample[x_]


class GTExperiment(Experiment):
    """Short summary.

    :param Parameters params: Experimental configuration.
    :attr String benchmark: Name of the benchmark (here: "groundtruth").

    """

    def __init__(self, params):
        """Experiment with the squared exponential kernel.

        :param Parameters params: Parameters for the experiment.
        :param String benchmark: Name of the benchmark function.
        :return: None.
        :rtype: None

        """
        Experiment.__init__(self, params)
        assert self.benchmark == "groundtruth"

    def _gram_matrix(self, xs):
        """Calculate gram matric."""
        return np.asarray([[self.mykernel(x1, x2) for x2 in xs] for x1 in xs])

    def cholesky_sampling(self, gram):
        """Sampling via Cholesky and Kronecker decomposition."""
        gram = gram * (1 / self.params.variance[0])
        try:
            L = np.linalg.cholesky(gram)
        except:
            gram += 1e-12 * np.identity(gram.shape[0])  # 1e-05
            L = np.linalg.cholesky(gram)
            print("Cholesky was not numerically stabel.")
        d = L.shape[0] ** self.params.dim
        samples = []
        np.random.seed(28)
        for _ in range(self.params.nb_samples):
            random_U = np.random.normal(loc=0, scale=1, size=d).reshape(
                [L.shape[0]] * self.params.dim
            )
            sample = np.dot(np.dot(L, random_U), L.T)
            if self.params.dim == 1:
                sample = np.dot(L, random_U)
            elif self.params.dim == 2:
                first = np.einsum("jk,kl->jl", L, random_U)
                sample = np.einsum("jk,kl->jl", first, L.T)
            elif self.params.dim == 3:
                first = np.einsum("jk,klm->jlm", L, random_U)
                second = np.einsum("jk,lkm->jlm", L, first)
                sample = np.einsum("jk,lmk->jlm", L, second)
            else:
                raise ValueError("Dimensions > 3 are not supported ...")
            samples.append(sample)
        samples_shape = [self.params.nb_samples] + [
            self.params.step_size  # * self.params.range_length
        ] * self.params.dim
        samples = np.asarray(samples) * np.sqrt(self.params.variance[0])
        samples = samples.reshape(samples_shape)
        return samples

    def generate_samples(self):
        """Draw some samples from a centered multivariate normal given the gram matrix.

        :return: Samples.
        :rtype: np.[Float](nb_samples, samplesize)

        """
        if self.params.benchmark == "groundtruth":
            if self.params.kernelname[0] == "polynomial":
                samples = utils.polynomial_samples(self.params, 0)
            else:
                myinputrange = [self.params.input_range[0]]
                mydomain = np.meshgrid(
                    *[
                        np.arange(
                            start,
                            stop,
                            self.params.range_length / self.params.step_size,
                        )
                        for start, stop in myinputrange
                    ],
                    indexing="ij"
                )
                gram = self._gram_matrix(mydomain[0])
                # Cholesky: Sigma_i = L_iL_i^T
                # Kronecker: Cholesky(Sigma) = L_i x L_j x L_k ...
                # Sampling: Cholesky(Sigma) v, where x sim N(0,1)
                samples = self.cholesky_sampling(gram)

            np.save(utils.get_filename(self, "samples_") + ".npy", samples)
            self.samples = np.asarray(samples)
            return samples

    def load_samples(self, filename=None):
        """Load sampels from stored array.

        :return: None
        :rtype: None

        """
        if filename is None:
            filename = utils.get_filename(self, "samples_") + ".npy"
        samples = np.load(filename)
        self.samples = samples


class RealExperiment(Experiment):
    """Experiment on benchmark functions.

    :param Parameters params: Experimental configuration.
    :attr String benchmark: Name of the benchmark.
    :attr function benchmark_function: Benchmark function.
    :attr Float minimum_value: Optimal (i.e. minimal) value of the benchmark function.
    :attr ? turbo_benchmark_function: Benchmark function in format for TurBO.
    :attr params:

    """

    def __init__(self, params):
        """Experiment on benchmarl functions.

        :param Parameters params: Parameters for the experiment.
        :return: None.
        :rtype: None

        """
        Experiment.__init__(self, params)
        self.benchmark = params.benchmark
        self.benchmark_function = benchmark_functions.benchmark_function(
            params.benchmark
        )
        if self.params.max_or_min == "max":
            raise NotImplementedError(
                "Maximization of benchmark functions is not implemented!"
            )
        self.minimum_value = benchmark_functions.MINIMA[params.benchmark][1]
        self.params.minimum_value = self.minimum_value
        self.turbo_benchmark_function = (
            benchmark_functions.get_turbo_benchmark_function(params)
        )
        self.samples = [None]

    def gp_ucb_f(self, sample):
        """Benchmark function in format for BO.

        :param None sample: Not used.
        :return: Description of returned object.
        :rtype: type

        """
        return self.benchmark_function[1]
