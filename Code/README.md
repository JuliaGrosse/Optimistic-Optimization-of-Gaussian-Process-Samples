

# README

This code repository accompanies the "Optimistic Optimization of Gaussian Process Samples" paper.

## Getting started

Create a `Parameters` object (See parameters.py) to specify your search domain,
prior assumptions and the optimization method that you want to use. Then,
create an `Experiment` object (See GPOO.gpoo_exeriment, baselines.EI_experiment etc). If you want to run experiments on synthetic samples, you can sample "ground truth functions" (i.e. exact samples from the GP) by calling `experiment.generate_samples()`. Start the optimization by calling `experiment.run_experiment()`.

The subfolder "experiments" contains many examples for how to do so.

The main purpose of this code repository is to enable reproducability of the results in the experimental settings of the "Optimistic Optimization of Gaussian Process Samples" paper. If you
would like to use this code in your own settings you could start
by implementing the function you would like to optimize in `benchmark_functions.py` and then follow steps for the benchmark experiments.

## Project Structure


### GPOO

The subfolder **GPOO** contains the implementation of GPOO.

* `gpoo_experiment.py`: set up the optimization process
* `construct_children_utils.py`: subroutines to partition a nodes cell
* `maxheap.py`: implementation of a maxheap for the search tree
* `optimizer.py`: subroutines for the optimization loop
* `gpoo_diameters.py`: subroutines to calculate the upper bounds

### baselines

The subfolder **baselines** contains the implementations of the baselines based on [emukit](https://emukit.github.io) and [TurBO](https://github.com/uber-research/TuRBO) code.

* `<name_of_basline>_experiment.py`: set up the optimization process with the corresponding baseline

The additional files in this folder belong to the implementations from emukit/Turbo with small modifications, e.g. to use a custom lengthscale for TurBO etc.

### experiments

* `benchmarkexperiments`: scripts for running the experiments with GP-OO and the baselines on the benchmark functions
     * `benchmarks`: used hyperparameters for the domains and the optimization methods
     *  `benchmark_<name_of_baseline>.py`: scripts for running the experiment
     * `find_beta_parameters.py`: grid searches for best hyperparameter
     * `sample_random_domains.py`: subsample the domains
* `syntheticexperiments`: scripts for running the experiments with GP-OO and the baselines on samples from a GP
* `betaexperiments`: scripts for running the experiments from Appendix C for heuristical choices of beta in GP-OO
* `directcomparison`: scripts for a more detailed comparison between GP-OO and DIRECT as described in the Appendix D

Run, for example, by calling from the folder main `Code`:

`python3 -m experiments.benchmarkexperiments.benchmark_gpoo`

or

`python3 -m experiments.syntheticexperiments.synthetic_ei`

### plottingscripts

The subfolder **plottingscripts** contains the scripts used to generate the figures shown in the paper:

* `benchmark_average_regret_TMLR_plot.py`: Figure 6
* `benchmark_plot_cumulative_time_TMLR.py`: Figure 10
* `comparison_with_direct.py`: Figure 12
* `benchmark_plot_time_TMLR.py`: Figure 7, 8, 9
* `beta_plot_TMLR.py`: Figure  11
* `partitions_plot_TMLR.py`: Figure 2
* `quadratic experiment.py`: Figure 5 (+ corresponding experiment)
* `synthetic_plot_regret_TMLR.py`: Figure 3
* `timeplot_synthetic`: Figure 4
* `timing_results_rebuttal_logscale.py`, `timing_results_rebuttal`, `timing_results`: helper functions for plotting



### plots

Generated plots in pdf format end up in this folder.

### results

Logged results end up in this folder.

### additional files:

* `benchmark_functions.py`: Implementation of the benchmark functions in the formats as required by GP-OO or the baselines
* `kernel_functions.py`: Implementation of some common kernels for use with GP-OO
* `parameteres.py`: Container classes for all kinds of hyperparameters from the GP's lengthscale, to the size of the search domain or the exploration constant of the search methods.
* `experiment.py`: Base class for setting up experiments.
* `utils.py`: Miscellaneous.
