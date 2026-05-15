#!/usr/bin/env python
# coding: utf-8

# # EXPERIMENT 3 (Table 1)
# 
# Reproducing Table 1: axiom satisfaction of three architectures (MLP, CNN, WEC) trained unsupervised on axiom losses, plus 16 known voting rules for comparison. IC sampling, 55 voters, 5 alternatives.
# 
# Common hyperparameters (paper §5.1, §6.3):
# * 55 voters, 5 alternatives, IC sampling
# * 15,000 gradient steps, batch size 200, AdamW
# * Cosine annealing with warm restarts (T_0 = 100)
# * `sample_size_applicable = 400` for axiom checking
# 
# Architecture-specific:
# * MLP: 4 hidden layers à 128. Optimizes NW, Anonymity, Condorcet, Pareto, Independence.
# * CNN: kernel1 = (5,1), kernel2 = (1,5), 64 channels, 3 linear layers à 128. Same axioms as MLP.
# * WEC: pretrained Word2Vec (corpus 100k, dim 200, window 7, skip-gram) + averaging + 3 linear layers à 128. Optimizes NW, Condorcet, Pareto only (anonymous by design).
# 
# Decoding variants reported in Table 1: plain (`p`), neutrality-averaged (`n`), neutrality-and-anonymity-averaged (`na`). MLP/CNN report all three; WEC reports `p` and `n` only (anonymous by design).
# 
# Paper run times (Fig. 16): MLP ≈ 5h 29min, CNN ≈ 6h 08min, WEC ≈ 2h 05min on a CPU laptop.

# In[1]:


import exp3
import utils
import plot_and_visual

MAX_NUM_VOTERS = 55
MAX_NUM_ALTERNATIVES = 5
ELECTION_SAMPLING = {'probmodel': 'IC'}
NUM_GRADIENT_STEPS = 15000
LOSS_REPORT_INTERVALS = 100
BATCH_SIZE = 200
LEARNING_SCHEDULER = 100
EVAL_DATASET_SIZE = 500
SAMPLE_SIZE_APPLICABLE = 400
SAMPLE_SIZE_MAXIMAL = int(1e6)
DISTANCE = 'KLD'

AXIOMS_CHECK_MODEL = ['Anonymity', 'Neutrality', 'Condorcet', 'Pareto', 'Independence']

AXIOM_OPT_MLP_CNN = {
    'No_winner':   {'weight': 10, 'period': 'always'},
    'All_winners': None,
    'Inadmissible': None,
    'Resoluteness': None,
    'Parity':      None,
    'Anonymity':   {'weight': 1, 'period': 'always', 'sample': 50},
    'Neutrality':  None,
    'Condorcet1':  {'weight': 2, 'period': 'always'},
    'Condorcet2':  None,
    'Pareto1':     None,
    'Pareto2':     {'weight': 1, 'period': 'always'},
    'Independence':{'weight': 1, 'period': 'always', 'sample': 4},
}

print('Training CNN (NW, A, C, P, I)')
location_cnn = exp3.experiment3(
    architecture = 'CNN',
    max_num_voters = MAX_NUM_VOTERS,
    max_num_alternatives = MAX_NUM_ALTERNATIVES,
    election_sampling = ELECTION_SAMPLING,
    num_gradient_steps = NUM_GRADIENT_STEPS,
    report_intervals = NUM_GRADIENT_STEPS,
    loss_report_intervals = LOSS_REPORT_INTERVALS,
    eval_dataset_size = EVAL_DATASET_SIZE,
    model_to_rule = {
        'plain': True,
        'neut-averaged': None,
        'neut-anon-averaged': [12, 10],
    },
    sample_size_applicable = SAMPLE_SIZE_APPLICABLE,
    sample_size_maximal = SAMPLE_SIZE_MAXIMAL,
    architecture_parameters = {
        'kernel1': [5, 1],
        'kernel2': [1, 5],
        'channels': 64,
    },
    axioms_check_model = AXIOMS_CHECK_MODEL,
    axioms_check_rule = [],
    axiom_opt = AXIOM_OPT_MLP_CNN,
    comp_rules_axioms = [],
    comp_rules_similarity = [],
    distance = DISTANCE,
    batch_size = BATCH_SIZE,
    learning_scheduler = LEARNING_SCHEDULER,
    save_model = True,
)
print(f'CNN results: {location_cnn}')
