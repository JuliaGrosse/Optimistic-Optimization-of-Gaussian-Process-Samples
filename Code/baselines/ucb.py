###############################################################################
###.                          THIS FILE HAS BEEN MODIFIED (22.05.23)
###
###(The original version of the file belongs to emukit. See LICENSE_EMUKIT for more information.)
###
###############################################################################

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union

import numpy as np
import math

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition


class NegativeLowerConfidenceBound(Acquisition):
    def __init__(
        self,
        model: Union[IModel, IDifferentiable],
        dim,
        epsilon: np.float64 = np.float64(1),
        params=None,
    ) -> None:

        """
        This acquisition computes the negative lower confidence bound for a given input point. This is the same
        as optimizing the upper confidence bound if we would maximize instead of minimizing the objective function.
        For information as well as some theoretical insights see:
        Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
        Niranjan Srinivas, Andreas Krause, Sham Kakade, Matthias Seeger
        In Proceedings of the 27th International Conference  on  Machine  Learning
        :param model: The underlying model that provides the predictive mean and variance for the given test points
        :param dim: Dimension of the function space.
        :param epsilon: Upper bounds hold with probability 1-epsilon
        """
        self.model = model
        self.epsilon = epsilon
        self.iteration = 1
        self.dim = dim
        self.params = params
        self.b = self.params.b
        self.a = self.params.a
        ### Calculate number of points for union bound based on discretization size
        if self.params.beta is None and self.params.ucb_discretization is not None:
            self.discretization_size = np.prod(
                [
                    self.params.ucb_discretization * (interval[1] - interval[0])
                    for i, interval in enumerate(self.params.input_range)
                ]
            )
            self.beta = np.sqrt(
                2
                * np.log(
                    self.discretization_size
                    * self.iteration ** 2
                    * math.pi ** 2
                    / (6 * self.params.epsilon)
                )
            )
        ### Calculate number of points for union bound based on a and b
        elif self.params.beta is None and self.params.ucb_discretization is None:
            self.r = np.mean(self.params.input_range[0][1])
            self.dimension_lengths = [
                self.params.input_range[i][1] - self.params.input_range[i][0]
                for i in range(len(self.params.input_range))
            ]
            term1 = 2 * np.log(
                self.iteration ** 2 * 2 * math.pi ** 2 / (3 * self.params.epsilon)
            )
            discretization_per_dimension = [
                self.dim
                * self.iteration
                * self.b
                * dimension_length
                * np.sqrt(np.log(4 * self.dim * self.a / self.params.epsilon))
                for dimension_length in self.dimension_lengths
            ]
            term2 = 4 * np.log(np.prod(discretization_per_dimension))
            self.beta = np.sqrt(term1 + term2)
        else:
            self.beta = self.params.beta

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the negative lower confidence bound
        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        return -(mean - self.beta * standard_deviation)

    def update_parameters(self):
        self.iteration += 1
        ### Calculate number of points for union bound based on discretization size
        if self.params.beta is None and self.params.ucb_discretization is not None:
            self.beta = np.sqrt(
                2
                * np.log(
                    self.discretization_size
                    * self.iteration ** 2
                    * math.pi ** 2
                    / (6 * self.params.epsilon)
                )
            )
        ### Calculate number of points for union bound based on a and b
        elif self.params.beta is None and self.params.ucb_discretization is None:
            term1 = 2 * np.log(
                self.iteration ** 2 * 2 * math.pi ** 2 / (3 * self.params.epsilon)
            )
            discretization_per_dimension = [
                self.dim
                * self.iteration
                * self.b
                * dimension_length
                * np.sqrt(np.log(4 * self.dim * self.a / self.params.epsilon))
                for dimension_length in self.dimension_lengths
            ]
            term2 = 4 * np.log(np.prod(discretization_per_dimension))
            self.beta = np.sqrt(term1 + term2)
        else:
            self.beta = self.params.beta

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the negative lower confidence bound and its derivative
        :param x: points where the acquisition is evaluated.
        """
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        lcb = -(mean - self.beta * standard_deviation)

        dlcb_dx = -(dmean_dx - self.beta * dstandard_deviation_dx)

        return lcb, dlcb_dx

    @property
    def has_gradients(self):
        return self.params.ucb_has_gradients
