"""
Training and evaluation methods
"""

from tqdm import tqdm
import time
from random import randint
import numpy as np

import pref_voting
from pref_voting.generate_profiles import generate_profile

import torch

import utils
from utils import recast_profile_wo_mult, dict_axioms


def train(dataloader, model, loss_fn, optimizer):
    """The training function for a model. Returns the average batch loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction as logits
        logits = model(x)
        # Compute error the prediction
        loss = loss_fn(logits, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)

def loss(model, loss_fn, dataloader):
    """Computing the loss of a model on a dataset"""
    model.eval()
    with torch.no_grad():
        loss = torch.zeros(1, dtype=torch.float)
        for batch, (x, y) in enumerate(dataloader):
            loss += loss_fn(model(x), y)
        avg_loss = loss/len(dataloader)
        return avg_loss.item()


def accuracy(model, dataloader):
    """Computing the accuracy of a model on a dataset"""
    model.eval()
    with torch.no_grad():
        correct = 0
        considered = 0
        for batch, (x, y) in enumerate(dataloader):
            # Compute prediction as logits
            logits = model(x)
            # Turn logits into binary classification
            pred = torch.round(torch.sigmoid(logits))
            # Check equality entry-wise, then sum along labels
            comparison = (pred == y).sum(1).tolist()
            correct += sum([int(item == len(y[0])) for item in comparison])
            considered += len(y)
        return correct / considered


def axiom_satisfaction(
    rule,
    axiom,
    max_num_voters,
    max_num_alternatives,
    election_sampling,
    sample_size_applicable,
    sample_size_maximal,
    permutation_sample,
    full_profile=False,  
    comparison_rule=None,
):
    """
    Compute how much a rule satisfies an axiom
    

    Inputs: 
    The voting rule `rule` which should be check on its satisfaction of the
    voting axiom `axiom` (from `axioms.py`). 
    
    To do so, we sample profiles of up to `max_num_voters` and 
    `max_num_alternatives` using `election_sampling` as probabilistic model. 
    For each sampled profile, we check if the axiom is applicable and, if so, 
    whether it is satisfied of violated. 
    
    We sample as many profiles until we find `sample_size_applicable` 
    many where the axiom is applicable or until we have sampled 
    `sample_size_maximal` many profiles. 
    
    The `permutation_sample` describes the number of permutations/samples we 
    should sample when checking the axiom on a given profile (see 
    `utils.dict_axioms_sample` for recommendations). 
    
    If `full_profile` is true (default is false), only profiles of maximal 
    num_voters and maximal num_alternatives are considered. 
    
    A `comparison_rule` can be given to check how often the axiom satisfaction 
    of the given rule coincides with the axiom satisfaction of the comparison 
    rule.

    Output: a dictionary with the following entries
    * 'sampling_as_desired': whether enough samples where the axiom is 
      applicable have been found (and if not, how many)
    * 'cond_satisfaction': percentage of the axiom being satisfied given that 
      it is applicable.
    * 'absolute_satisfaction': percentage of the axiom either being applicable 
      and satisfied or not applicable
    * 'percent_applicable': percentage of the axiom being applicable
    * And if a comparison rule was provided, 'percent_coincidence': percentage 
      of the given rule coinciding with the comparison rule in axiom 
      satisfaction    
    """

    # Initialize the satisfaction percentage of the axiom that we compute
    satisfaction = 0

    # Starting iteration of sampling profiles and check if axiom is
    # applicable and satisfied
    iteration = 0
    applicable = 0
    satisfaction = 0
    if comparison_rule is not None:
        coincidence = 0
    while iteration < sample_size_maximal:
        if applicable < sample_size_applicable:
            # Choose a number of alternatives and a number of voters
            if full_profile:
                num_alternatives = max_num_alternatives
                num_voters = max_num_voters
            else:
                num_alternatives = randint(1, max_num_alternatives)
                num_voters = randint(1, max_num_voters)
            # Generate a profile
            prof = generate_profile(
                num_alternatives,
                num_voters,
                election_sampling
            )
            prof = recast_profile_wo_mult(prof)

            # Compute axiom satisfaction
            sat = axiom(rule, prof, permutation_sample)

            if sat == 0:  # so axiom wasn't applicable
                iteration += 1
            if sat == -1:  # so axiom was applicable but not satisfied
                iteration += 1
                applicable += 1
            if sat == 1:  # so axiom was applicable and satisfied
                iteration += 1
                applicable += 1
                satisfaction += 1

            # and, if given, also axiom satisfaction of the comparison rule
            if comparison_rule is not None:
                comparison_satisfaction = axiom(
                    comparison_rule, prof, permutation_sample
                )
                coincidence += int(sat == comparison_satisfaction)
        else:  # so we sampled enough profiles where the axiom is applicable
            break
    if iteration == sample_size_maximal:
        sampling_achieved = "No"
        axiom_name = list(dict_axioms.keys())[list(dict_axioms.values()).index(axiom)]
        print(
            f"Warning: When checking axiom {axiom_name}, the maximum number ({sample_size_maximal}) of sampled profiles was reached without finding the desired number of profiles ({sample_size_applicable}) for which the axiom was applicable. Only {applicable} many where found. We still go through with computing the satisfaction scores, albeit with not enough samples."
        )
    else:
        sampling_achieved = "Yes"

    if applicable != 0:
        conditional_satisfaction = satisfaction / applicable
    else:
        conditional_satisfaction = np.nan
    absolute_satisfaction = (
        satisfaction + (iteration - applicable)
    ) / iteration  # how often it was either satisfied or not applicable
    avg_applicable = applicable / iteration
    if comparison_rule is not None:
        avg_coincidence = coincidence / iteration

    if comparison_rule is None:
        return {
            "sampling_as_desired": f"{sampling_achieved} ({applicable}/{sample_size_applicable})",
            "cond_satisfaction": conditional_satisfaction,
            "absolute_satisfaction": absolute_satisfaction,
            "percent_applicable": avg_applicable,
        }
    else:
        return {
            "sampling_as_desired": f"{sampling_achieved} ({applicable}/{sample_size_applicable})",
            "cond_satisfaction": conditional_satisfaction,
            "absolute_satisfaction": absolute_satisfaction,
            "percent_applicable": avg_applicable,
            "percent_coincidence": avg_coincidence,
        }


def rule_similarity(rule, rule_comparison_list, profiles, verbose=False):
    """
    Computes the closeness of `rule` to the rules in `rule_comparison_list`

    Input: 
    * a voting rule F, 
    * a nonempty list of names for voting rules G_1, ..., G_m (the names 
      must be keys in utils.dict_rules_all)
    * a list of profiles.
    * If `verbose` is set to True, then the comparison rules that took 
      more than 10 sec to compute are mentioned

    Output: a dictionary whose keys are the names of the rules G_k and the 
    corresponding value for rule G_k is again a dictionary with the following
    keys and values:
    * 'hamming':the average hamming distance (number of disagreements between
      rule F and rule G_k about whether an alternative is a winner, divided
      by the number of alternatives)
    * 'identity_accu':the percentage of the F-winning set being *identical* to
      the G_k-winning set.
    * 'overlap_accu':the percentage of the F-winning set *overlapping*
      (i.e., having nonempty intersection) to the G_k-winning set.
    * 'subset_accu':the percentage of the F-winning set being a *subset* of
      the G_k-winning set.
    * 'superset_accu':the percentage of the F-winning set being a *superset* of
      the G_k-winning set.
    """

    size = len(profiles)
    m = len(rule_comparison_list)
    assert m > 0, "The rule_comparison_list needs to be nonempty"
    # Initialize the dictionary of dictionaries
    similarities = {
        rule_name
        :
        {
            "hamming": 0,
            "identity_accu": 0,
            "overlap_accu": 0,
            "subset_accu": 0,
            "superset_accu": 0,
        }
        for rule_name in rule_comparison_list
    }
    for prof in profiles:
        # Compute the winning sets, cast as characteristic function
        num_alternatives = prof.num_cands
        # Compute F_winning_set
        F_winning_set = utils.winner_to_vec(rule(prof), num_alternatives)
        # Compute G_winning_sets
        G_winning_sets = {}
        for rule_name in rule_comparison_list:
            start = time.time()
            winners = utils.dict_rules_all[rule_name](prof)
            stop = time.time()
            if verbose and stop - start > 10:
                print(f'Computing the winners for rule {rule_name} on the current profile (with {prof.num_voters} voters and {prof.num_cands} alternatives) took {stop-start}')
            G_winning_sets[rule_name] = utils.winner_to_vec(winners, num_alternatives)

        # Now compare model output to the targets
        for rule_name in rule_comparison_list:
            # Compute hamming distance between F-winning_set and G_winning_set
            similarities[rule_name]["hamming"] += (
                len(
                    [
                        i
                        for i in range(num_alternatives)
                        if F_winning_set[i] != G_winning_sets[rule_name][i]
                    ]
                )
                / num_alternatives
            )
            # Compute identity accuracy between F-winning_set and G_winning_set
            similarities[rule_name]["identity_accu"] += int(F_winning_set == G_winning_sets[rule_name])


            # Compute overlap accuracy between F-winning_set and G_winning_set
            similarities[rule_name]["overlap_accu"] += int(
                1
                in [
                    int(F_winning_set[i] == 1 and G_winning_sets[rule_name][i] == 1)
                    for i in range(num_alternatives)
                ]
            )
            # Compute subset accuracy between F-winning_set and G_winning_set
            similarities[rule_name]["subset_accu"] += int(
                all(
                    [
                        (F_winning_set[i] != 1 or G_winning_sets[rule_name][i] == 1)
                        for i in range(num_alternatives)
                    ]
                )
            )
            # Compute superset accuracy between F-winning_set and G_winning_set
            similarities[rule_name]["superset_accu"] += int(
                all(
                    [
                        (F_winning_set[i] == 1 or G_winning_sets[rule_name][i] != 1)
                        for i in range(num_alternatives)
                    ]
                )
            )

    # Finally, take the averages
    for rule_name in rule_comparison_list:
        similarities[rule_name]["hamming"] /= size
        similarities[rule_name]["identity_accu"] /= size
        similarities[rule_name]["overlap_accu"] /= size
        similarities[rule_name]["subset_accu"] /= size
        similarities[rule_name]["superset_accu"] /= size

    return similarities


def resoluteness(rule, profiles):
    """
    Computes the resoluteness of rule on the profiles

    Any profile with no alternatives is ignored.

    """
    # Ignore profiles with 0 alternatives
    profiles_nonzero = [prof for prof in profiles if prof.num_cands > 0]
    assert len(profiles_nonzero) > 0, 'There are no profiles with 1 or more alternatives'

    # For each profile, compute the ratio of winners to all alternatives
    winner_ratios = [len(rule(prof))/prof.num_cands for prof in profiles_nonzero]
    # Take the average
    a = sum(winner_ratios)/len(profiles_nonzero)
    return a










def admissibility(rule, profiles):
    """
    How admissible the outputted winners of the rule are on the profiles
    
    Inputs: `rule` is a voting rule, that here is just assumed to be a 
    function taking profiles (in the sense of the `pref_voting` package) 
    as input and produces a list/set of winners as outputs, though the 
    winners can, but need not be, candidates in the profile. Winners that
    are not candidates in the profile are called 'inadmissible'.
    `profiles` is a list of profiles on which admissibility is checked.

    Output: A dictionary describing the percentages of profiles where
    * no winner was outputted at all
    * no admissible winner was outputted
    * all admissable winners were outputted
    * some inadmissible winner was outputted
    """    
    # First check that profiles is nonempty 
    # (empty sequences are considered False)
    assert profiles, 'The list of profiles should be nonempty'
    # Initialize the output dictionary
    summary = {
        'no_winner_at_all':0,
        'no_admissible_winner':0,
        'all_admissible_winner':0,        
        'some_inadmissible_winner':0,
    }
    for profile in profiles:
        # Compute the winners according to the rule on the given profile
        winners = set(rule(profile))
        # Divide into admissible and inadmissible winners
        admissible_winners = set(
            [i for i in winners if i in profile.candidates]
        )
        inadmissible_winners = winners - admissible_winners
        # Update the summary
        if not winners: #true if no winners
            summary['no_winner_at_all'] += 1
        if not admissible_winners: #true if no admissible winners
            summary['no_admissible_winner'] += 1
        if admissible_winners == set(profile.candidates):
            summary['all_admissible_winner'] += 1
        if inadmissible_winners: #true if there are inadmissible winners
            summary['some_inadmissible_winner'] += 1
    # Take averages
    summary['no_winner_at_all'] /= len(profiles)
    summary['no_admissible_winner'] /= len(profiles)
    summary['all_admissible_winner'] /= len(profiles)
    summary['some_inadmissible_winner'] /= len(profiles)

    return summary