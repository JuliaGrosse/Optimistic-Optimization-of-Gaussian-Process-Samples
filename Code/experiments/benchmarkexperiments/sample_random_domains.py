"""Generate and store randomly subsampled domains."""
import random
import json

import numpy as np

import utils

np.random.seed(28)
random.seed(28)

NB_DOMAINS = 10
SAMPLED_DOMAINS = {}
DOMAINS = json.load(open("./experiments/benchmarkexperiments/benchmarks/domains.txt"))

for benchmark, domain in DOMAINS.items():
    sampled_domains = []
    for i in range(NB_DOMAINS):
        new_domain = utils.sample_domain(benchmark, domain)
        sampled_domains.append(new_domain)
    SAMPLED_DOMAINS[benchmark] = sampled_domains

with open(
    "./experiments/benchmarkexperiments/benchmarks/sampled_domains.json", "w"
) as fp:
    json.dump(SAMPLED_DOMAINS, fp)
