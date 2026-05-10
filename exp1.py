"""
The implementation of experiment 1 and variations of it
"""
import os
from datetime import datetime
import time
import numpy as np
import random
import json
from tqdm import tqdm
import math

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

import utils
from utils import flatten_list, flatten_onehot_profile, kendall_tau_order
import generate_data
from generate_data import generate_profile_data, pad_profile_data
from generate_data import onehot_profile_data
import models
from models import MLP, MLP_small, MLP_large, CNN, WEC
import train_and_eval

from gensim.models import Word2Vec

import wandb_integration










def experiment1(
        architecture,
        rule_names,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        num_gradient_steps,
        eval_dataset_size,
        sample_size_applicable,
        sample_size_maximal,
        architecture_parameters=None,
        axioms_check_model = list(utils.dict_axioms.keys()),
        axioms_check_rule = list(utils.dict_axioms_rules.keys()),
        max_num_voters_generation=None,
        max_num_alternatives_generation=None,
        merge="accumulative",
        comparison_rules = None,
        compute_resoluteness = False,
        random_seed=None,
        report_intervals = {'plot':100,'print':1000},
        batch_size=64,
        learning_rate = 1e-3,
        learning_scheduler = None,
        weight_decay = 0,
        save_model=False,        
    ):
    """
    Implements our experiment 1

    Inputs:
    * `architecture` can be either `MLP`, `MLP_small`, `MLP_large`, `CNN`, or 
      `WEC`. The latter two require additional parameters which can be passed 
      as `architecture_parameters` below (otherwise default values are chosen).

    * `rule_names`: list of names of rules used for data generation. Typically 
      just a singleton list. If multiple rules are given, their output is 
      merged in the dataset. 
    * `max_num_voters`: a positive integer describing the maximal number of 
      voters that will be considered
    * `max_num_alternatives`: a positive integer describing the maximal number 
      of alternatives that will be considered
    * `election_sampling`: a dictionary describing the parameters for the
      probability model with which profiles are generated. The most important 
      key is `probmodel`. See 
        https://pref-voting.readthedocs.io/en/latest/generate_profiles.html
    
    * `num_gradient_steps`: a positive integer describing the number of 
      gradient steps performed during training the model.
    * `eval_dataset_size`: a positive integer describing the number of profiles
      with their corresponding winning sets that should be used for testing.
    
    * `sample_size_applicable`: a positive integer describing the number of 
      sampled profiles on which the axioms are checked and applicable. 
    * `sample_size_maximal`: a positive integer describing how many profiles 
      are at most sampled when trying to find profiles on which the axioms are 
      applicable. 

    * `architecture_parameters` is not needed for MLPs, but CNNs and WEC they 
      are given the following default values
        * For CNN:
          {
            'kernel1':[5,1] , 
            'kernel2':[1,5], 
            'channels':64
          }
          Here `kernel1`: a list [height, width] of the dimension for the 
          kernel/filter in the first convolutional layer of the model. (The 
          height of the images is max_num_alternatives and the width of the 
          images is max_num_voters.) If max num voters/alternatives are quite 
          small, the kernel cannot be too big, otherwise error will be raised.  
          Similarly, `kernel2` is the [height, width]-dimension for the 
          kernel/filter in the second convolutional layer of the model.
          Finally, `channels` is the number of channels of the feature maps of 
          the model.
          If the also  
            'kendall_tau_ordered':<version>
          is added, then  every profile passed to the CNN is first ordered by 
          Kendall tau distance. The version can be either 'global' or 'local'. 
          In the former, rankings are ordered by distance from the 
          first ranking; in the latter, one also starts with the first ranking 
          but then always picks as the next ranking the one which is closest to
          the just picked ranking.
        * For WEC: 
          {
            'we_corpus_size':int(1e5), 
            'we_size':100, 
            'we_window':5, 
            'we_algorithm':1
          } 
          Here 'we_corpus_size' is the number of profiles used to pretrain the 
          word embeddings,  `we_size` is the size (i.e., length of vector) of 
          the word embeddings, `we_window` is the size of the window used when 
          training the word embeddings. Finally, `we_algorithm` is either 0 or 
          1 depending on whether one uses the CBOW algorithm or the skip gram 
          algorithm.
              
    * `axioms_check_model` is a list of names of axioms whose satisfaction is 
      checked for the trained model. By default, it is set to all axioms. If 
      empty, no axioms are checked.
    * `axioms_check_rule` is a list of names of axioms whose satisfaction is 
      checked for the rules. By default, it is only the condorcet and the 
      independence axiom, since the other axioms are always satisfied for 
      common voting rules. If empty, no axioms are checked.   

    * `max_num_voters_generation` (optional): a positive integer which is
      <= max_num_voters. It describes the maximal number of voters that will be
      considered *during training*. During testing the full max_num_voters will
      be considered.
    * `max_num_alternatives_generation` (optional): a positive integer which is
      <= max_num_alternatives. It describes the maximal number of alternatives 
      that will be considered *during training*. During testing the full 
      max_num_alternatives will be considered.
    
    * `merge`: By default set to `accumulative`, but could also be `selective`.
      If `rule_names` has length 1 (i.e., only a single rule is considered), 
      both are equivalent. If there are several rules, `selective` means a 
      profile is added to the training dataset only if all rules output the 
      same winning set, while with `accumulative` all profiles are added.
    
    * `comparison_rules` is, if not None, a nonempty list of names of voting 
      rules, to which the rule found by the model is compared to with respect 
      to various measures of similarity.
    * `compute_resoluteness` is False by default. If True, the resoluteness 
      coefficient of the learned rule is computed. A rule is highly resolute if
      it picks very few winners, and it is not resolute at all if it always 
      declares all alternatives as winners.

    * `random_seed` (optional): If not None, set all random seeds to this 
      provided value.
    * `report_intervals` by default the dictionary {'plot':100,'print':1000}, 
      saying that after every 10 (resp., 1000) gradient steps the dev-loss and 
      dev-accuracy is computed to later plot (resp. print) the learning 
      performance.
    * `batch_size`: a positive integer describing the size of the batches when 
      training the model. The default is 64.
    * `learning_rate`: a float number describing the learning rate when 
      training the model. The default value is 1e-3.
    * `learning_scheduler`: By default None, but if given, then a positive 
      integer describing the T_0 value for the CosineAnnealingWarmRestarts 
      scheduler (i.e., the number of iterations for the first restart).
    * `weight_decay` of the optimizer which we set by default to 0 since we use 
      synthetic data and hence don't need regularization (its usual default 
      value is 0.01).  
    * `save_model`: By default False, but if true, the neural network model 
      will be saved.
    """

    # SET UP BASICS

    start_time = time.time()

    assert (
        architecture in ['MLP', 'MLP_small', 'MLP_large', 'CNN', 'WEC']
    ), f"The supported architectures are 'MLP', 'MLP_small', 'MLP_large', 'CNN', and 'WEC' but {architecture} was given"

    if architecture_parameters is None:
        if architecture == 'CNN':
            architecture_parameters = {
                'kernel1':[5,1] , 
                'kernel2':[1,5], 
                'channels':64} 
        if architecture == 'WEC':
            architecture_parameters = {
                'we_corpus_size':int(1e5),
                'we_size':100, 
                'we_window':5, 
                'we_algorithm':1}


    assert (
        max_num_voters_generation is None 
        or 
        max_num_voters_generation <= max_num_voters
    ), f"max_num_voters_generation has to be <= max_num_voters"
    assert (
        max_num_alternatives_generation is None
        or max_num_alternatives_generation <= max_num_alternatives
    ), f"max_num_alternatives_generation has to be <= max_num_alternatives"

    if max_num_voters_generation is None:
        max_num_voters_generation = max_num_voters
    if max_num_alternatives_generation is None:
        max_num_alternatives_generation = max_num_alternatives

    do_kendall_tau = False
    if (architecture_parameters is not None and 
        'kendall_tau_ordered' in architecture_parameters):
        do_kendall_tau = True
        kendall_tau_version = architecture_parameters['kendall_tau_ordered']

    # Set seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


    # Set up saving of results
    rules_short = ""
    for rule_name in rule_names:
        rules_short += rule_name
    prob_model = election_sampling['probmodel']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    location = f"./results/exp1/{architecture}/exp1_{current_time}_{rules_short}_{prob_model}"
    os.mkdir(location)
    print(f'Saving location: {location}')


    results = {
        "location": location,        
        "architecture": architecture,
        "rule_names": rule_names,
        "max_num_voters": max_num_voters,
        "max_num_alternatives": max_num_alternatives,
        "election_sampling": election_sampling,
        "num_gradient_steps": num_gradient_steps,
        "eval_dataset_size": eval_dataset_size,
        "sample_size_applicable": sample_size_applicable,
        "sample_size_maximal": sample_size_maximal,
        "architecture_parameters": architecture_parameters,
        "axioms_check_model": axioms_check_model,
        "axioms_check_rule": axioms_check_rule,
        "max_num_voters_generation": max_num_voters_generation,
        "max_num_alternatives_generation": max_num_alternatives_generation,
        "merge": merge,
        "comparison_rules": comparison_rules,
        "compute_resoluteness": compute_resoluteness,
        "random_seed": random_seed,
        "report_intervals": report_intervals,
        "batch_size": batch_size,        
        "learning_rate": learning_rate,
        "learning_scheduler": learning_scheduler,
        "weight_decay": weight_decay,
        "save_model": save_model,
    }


    with open(f"{location}/results.json", "w") as json_file:
        json.dump(results, json_file)

    wandb_integration.init_run("experiment1", results, location)


    # GENERATING DATA

    # Training data will be generated before each training batch
    # Only WEC first needs to pretrain word embeddings

    if architecture == 'WEC':
        # First gather architecture parameters
        we_corpus = architecture_parameters['we_corpus_size']
        we_size = architecture_parameters['we_size']
        we_window = architecture_parameters['we_window']
        we_algorithm = architecture_parameters['we_algorithm']
        load_embeddings_from = architecture_parameters.get(
                'load_embeddings_from', None
            ) 


        if load_embeddings_from is None:
            print('Now pretraining word embeddings')

            print("First generate profiles and turn them into corpus")
            # Generate profiles and their winning sets
            X_train_profs, _ , _ = generate_profile_data(
                max_num_voters_generation,
                max_num_alternatives_generation,
                we_corpus,
                election_sampling,
                [],
                merge='empty'
            )

            # Turn set of profiles X into a corpus (each profile a sentence)
            train_sentences = [
                    [models.ranking_to_string(ranking)
                    for ranking in profile.rankings]
                for profile in X_train_profs]
            # Add the 'UNK' word for future unknown words. And add 'PAD' for
            # padding sentences to desired length. (Adding these after training
            # the embeddings seems inefficient.)
            train_sentences_with_UNK_and_PAD=train_sentences+[['UNK'], ['PAD']]

            # Pretrain an word embedding on this corpus
            print('Now train word embeddings')
            pre_embeddings = Word2Vec(
                    train_sentences_with_UNK_and_PAD, 
                    vector_size=we_size, 
                    window=we_window, 
                    min_count=1, 
                    workers=8, 
                    sg=we_algorithm
                )
            print('Done pretraining word embeddings.')

            if save_model:
                print('Save the word embeddings.')
                # We save the pre_embedding word2vec model, so it can be used
                # when initializing a WEC model that we then load with 
                # previously trained parameters 
                pre_embeddings.save(f"{location}/pre_embeddings.bin")

        # Load back with memory-mapping = read-only, shared across processes.
        if load_embeddings_from is not None:
            print('Load the word embeddings.')
            load_embeddings_from = architecture_parameters['load_embeddings_from']
            pre_embeddings = Word2Vec.load(f"{load_embeddings_from}/pre_embeddings.bin")





    # Dev dataset
    print('Now generate dev and test data')
    # Note: we dev on same number of voters/alternatives as for training
    X, y, sample_rate = generate_profile_data(
        max_num_voters_generation,
        max_num_alternatives_generation,
        eval_dataset_size,
        election_sampling,
        [utils.dict_rules_all[rule_name] for rule_name in rule_names],
        merge=merge,
    )

    if architecture in ['MLP', 'MLP_small', 'MLP_large']:
        dev_dataloader = generate_data.tensorize_profile_data_MLP(
            X,y,max_num_voters,max_num_alternatives,batch_size
        )

    if architecture == 'CNN':
        if do_kendall_tau:
            X = [kendall_tau_order(profile, kendall_tau_version) 
                 for profile in X]
        dev_dataloader = generate_data.tensorize_profile_data_CNN(
            X,y,max_num_voters,max_num_alternatives,batch_size
        )

    if architecture == 'WEC':
        dev_sentences = [[models.ranking_to_string(ranking) 
                           for ranking in profile.rankings] 
                         for profile in X]
        dev_dataloader,summary_unks = generate_data.tensorize_profile_data_WEC(
            pre_embeddings,
            dev_sentences,
            y,
            max_num_voters,
            max_num_alternatives,
            batch_size,
            num_of_unks=True
        )

        # Initialize computation of number of UNKs
        number_of_unks = {}
        # Compute number of UNKs in dev set
        ratio = summary_unks['ratio']
        num_unks = summary_unks['num_unks']
        all_words = summary_unks['all_words']
        print(f'Occurrences of UNK in dev data: {ratio}% ({num_unks} of {all_words} words)')
        number_of_unks['num_unks_dev_set'] = summary_unks 



    # Test dataset
    # We test on the same number of voters/alternatives as for training;
    # and, if the maximum number of voters/alternatives was restricted during
    # training, we also test on the full number
    X, y, sample_rate = generate_profile_data(
        max_num_voters_generation,
        max_num_alternatives_generation,
        eval_dataset_size,
        election_sampling,
        [utils.dict_rules_all[rule_name] for rule_name in rule_names],
        merge=merge,
    )
    # Save these profiles for later on
    testing_profiles = X

    if (max_num_voters_generation < max_num_voters
        or
        max_num_alternatives_generation < max_num_alternatives):
        X_full, y_full, sample_rate = generate_profile_data(
            max_num_voters,
            max_num_alternatives,
            eval_dataset_size,
            election_sampling,
            [utils.dict_rules_all[rule_name] for rule_name in rule_names],
            merge=merge,
        )
        # In this case, use these as testing profiles later on
        testing_profiles = X_full



    if architecture in ['MLP', 'MLP_small', 'MLP_large']:
        test_dataloader = generate_data.tensorize_profile_data_MLP(
            X,y,max_num_voters,max_num_alternatives,batch_size
        )
        if (max_num_voters_generation < max_num_voters 
            or 
            max_num_alternatives_generation < max_num_alternatives):
            test_full_dataloader = generate_data.tensorize_profile_data_MLP(
                X_full,y_full,max_num_voters,max_num_alternatives,batch_size
            )

    if architecture == 'CNN':
        if do_kendall_tau:
            X = [kendall_tau_order(profile, kendall_tau_version) 
                 for profile in X]
        test_dataloader = generate_data.tensorize_profile_data_CNN(
            X,y,max_num_voters,max_num_alternatives,batch_size
        )
        if (max_num_voters_generation < max_num_voters
            or
            max_num_alternatives_generation < max_num_alternatives):
            test_full_dataloader = generate_data.tensorize_profile_data_CNN(
                X_full,y_full,max_num_voters,max_num_alternatives,batch_size
            )



    if architecture == 'WEC':    
        test_sentences = [[models.ranking_to_string(ranking)
                            for ranking in profile.rankings]
                          for profile in X]
        test_dataloader,summary_unks=generate_data.tensorize_profile_data_WEC(
            pre_embeddings,
            test_sentences,
            y,
            max_num_voters,
            max_num_alternatives,
            batch_size,
            num_of_unks=True
        )
        # Compute number of UNKs in test set
        ratio = summary_unks['ratio']
        num_unks = summary_unks['num_unks']
        all_words = summary_unks['all_words']
        print(f'Occurrences of UNK in test data: {ratio}% ({num_unks} of {all_words} words)')
        number_of_unks['num_unks_test_set'] = summary_unks 


        if (max_num_voters_generation < max_num_voters
            or
            max_num_alternatives_generation < max_num_alternatives):
            test_full_sentences = [[models.ranking_to_string(ranking)
                                     for ranking in profile.rankings]
                                   for profile in X_full]
            test_full_dataloader,summary_unks = generate_data.tensorize_profile_data_WEC(
                pre_embeddings,
                test_full_sentences,
                y_full,
                max_num_voters,
                max_num_alternatives,
                batch_size,
                num_of_unks=True
            )
            # Compute number of UNKs in full test set
            ratio = summary_unks['ratio']
            num_unks = summary_unks['num_unks']
            all_words = summary_unks['all_words']
            print(f'Occurrences of UNK in full test data: {ratio}% ({num_unks} of {all_words} words)')
            number_of_unks['num_unks_full_test_set'] = summary_unks


        with open(f"{location}/results.json") as json_file:
            data = json.load(json_file)

        data.update({"number_of_unks": number_of_unks})

        with open(f"{location}/results.json", "w") as json_file:
            json.dump(data, json_file)





    # NEURAL NETWORK TRAINING

    #Initialize our model for the experiment
    if architecture == 'MLP':
        exp_model = MLP(max_num_voters, max_num_alternatives)
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if architecture == 'MLP_small':
        exp_model = MLP_small(max_num_voters, max_num_alternatives)
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if architecture == 'MLP_large':
        exp_model = MLP_large(max_num_voters, max_num_alternatives)
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if architecture == 'CNN':
        exp_model = CNN(
            max_num_voters,
            max_num_alternatives,
            architecture_parameters['kernel1'],
            architecture_parameters['kernel2'],
            architecture_parameters['channels']
        )
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if architecture == 'WEC':
        exp_model = WEC(pre_embeddings, max_num_voters, max_num_alternatives)
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if learning_scheduler is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            exp_optimizer,
            T_0 = learning_scheduler
        )


    print('Now starting to train')
    learning_curve = {}
    report_plot = report_intervals['plot']
    report_print = report_intervals['print']


    for step in tqdm(range(num_gradient_steps)):
        # Generate data for the batch

        X_train_profs, y_train_wins, _ = generate_profile_data(
            max_num_voters_generation,
            max_num_alternatives_generation,
            batch_size,
            election_sampling,
            [utils.dict_rules_all[rule_name] for rule_name in rule_names],
            merge=merge,
        )

        if architecture in ['MLP', 'MLP_small', 'MLP_large']:
            train_dataloader = generate_data.tensorize_profile_data_MLP(
                X_train_profs, y_train_wins,
                max_num_voters,max_num_alternatives,batch_size
            )

        if architecture == 'CNN':
            if do_kendall_tau:
                X_train_profs = [
                    kendall_tau_order(profile, kendall_tau_version)
                    for profile in X_train_profs
                ]
            train_dataloader = generate_data.tensorize_profile_data_CNN(
                X_train_profs, y_train_wins,
                max_num_voters,max_num_alternatives,batch_size
            )

        if architecture == 'WEC':
            train_sentences = [[models.ranking_to_string(ranking)
                                for ranking in profile.rankings]
                               for profile in X_train_profs]
            train_dataloader, _ = generate_data.tensorize_profile_data_WEC(
                pre_embeddings,
                train_sentences,
                y_train_wins,
                max_num_voters,max_num_alternatives,batch_size,
                num_of_unks=False
            )

        # Do step in gradient descent
        train_loss = train_and_eval.train(
            train_dataloader,
            exp_model,
            exp_loss,
            exp_optimizer
        )
        wandb_integration.log({'train_loss': train_loss}, step=step)

        if learning_scheduler is not None:
            scheduler.step()

        if step % report_plot == 0:
            dev_accuracy = train_and_eval.accuracy(exp_model, dev_dataloader)
            dev_loss = train_and_eval.loss(exp_model, exp_loss, dev_dataloader)
            learning_curve[f'{step}'] = {'dev_loss' : dev_loss,
                                        'dev_accuracy' : dev_accuracy}
            wandb_integration.log({
                'dev_loss': dev_loss,
                'dev_accuracy': dev_accuracy,
            }, step=step)
        if step % report_print == 0:
            dev_accuracy = train_and_eval.accuracy(exp_model, dev_dataloader)
            dev_loss = train_and_eval.loss(exp_model, exp_loss, dev_dataloader)
            if learning_scheduler is not None:
                current_learning_rate = scheduler.get_last_lr()[0]
            else:
                current_learning_rate = exp_optimizer.defaults['lr']
            print(f'Step {step}: dev-loss {round(dev_loss,5)}, dev-accuracy {dev_accuracy}, lr={round(current_learning_rate,5)}')


    # Test at the end of training
    test_accuracy = train_and_eval.accuracy(exp_model, test_dataloader)
    print(f'Done training. Test-accuracy: {test_accuracy}')
    test_results = {'accuracy_on_test_data':test_accuracy}
    if (max_num_voters_generation < max_num_voters 
        or 
        max_num_alternatives_generation < max_num_alternatives):
        test_full_accuracy = train_and_eval.accuracy(
            exp_model, 
            test_full_dataloader
        )
        test_results['accuracy_on_test_data_with_more_voters_or_alternatives'] = test_full_accuracy
        print(f'Test-accuracy with more voters/alternatives than during training: {test_full_accuracy}')

    num_params = summary(exp_model).total_params
    print(f'Number of parameters of the model: {num_params}')

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({
        "learning curve": learning_curve, 
        "test_accuracy":test_results, 
        "number_of_model_parameters":num_params
    })

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    if save_model:
        # We save both the model state and the optimizer state to be able to 
        # continue training later on.
        torch.save({
            'arguments' : [max_num_voters, max_num_alternatives, architecture_parameters],
            'model_state_dict': exp_model.state_dict(),
            'optimizer_state_dict': exp_optimizer.state_dict()
            }, f"{location}/model.pth")
            


    # EVALUATION

    # Define the rule computed by the model
    if architecture in ['MLP', 'MLP_small', 'MLP_large']:
        model_rule = models.MLP2rule(exp_model)
        model_rule_full = models.MLP2rule(exp_model, full=True)

    if architecture == 'CNN':
        if do_kendall_tau:
            model_rule = models.CNN2rule_kendall(
                exp_model, 
                kendall_tau_version
            )
            model_rule_full = models.CNN2rule_kendall(
                exp_model, 
                kendall_tau_version, 
                full=True
            )    
        model_rule = models.CNN2rule(exp_model)
        model_rule_full = models.CNN2rule(exp_model, full=True)

    if architecture == 'WEC':    
        model_rule = models.WEC2rule(exp_model)
        model_rule_full = models.WEC2rule(exp_model, full=True)


    # Admissibility
    admissibility_summary = train_and_eval.admissibility(
        model_rule_full,
        testing_profiles
    )
    print('Admissability:')
    for k,v in admissibility_summary.items():
        print('   ', k, v)

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"admissability": admissibility_summary})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)



    
    # Initialize dictionary with axiom satisfactions 
    axiom_satisfactions = {}

    # Axiom satisfaction of model
    axiom_satisfaction_model = {}
    for axiom_name in axioms_check_model:
        sat = train_and_eval.axiom_satisfaction(model_rule,
                utils.dict_axioms[axiom_name],
                max_num_voters,
                max_num_alternatives,
                election_sampling,
                sample_size_applicable,
                sample_size_maximal,
                utils.dict_axioms_sample[axiom_name],
                full_profile=False,
                comparison_rule=None)
        axiom_satisfaction_model[axiom_name] = sat
        cond_sat = sat['cond_satisfaction']
        print(f'The model satisfies axiom {axiom_name} to {100*cond_sat}%')
    axiom_satisfactions['learned_rule'] = axiom_satisfaction_model

    # Axiom satisfaction of rules
    for rule_name in rule_names:
        axiom_satisfaction_current_rule = {}
        for axiom_name in axioms_check_rule:
            sat = train_and_eval.axiom_satisfaction(
                    utils.dict_rules_all[rule_name],
                    utils.dict_axioms[axiom_name],
                    max_num_voters,
                    max_num_alternatives,
                    election_sampling,
                    sample_size_applicable,
                    sample_size_maximal,
                    utils.dict_axioms_sample[axiom_name],
                    full_profile=False,
                    comparison_rule=None)
            axiom_satisfaction_current_rule[axiom_name] = sat
            cond_sat = sat['cond_satisfaction']
            print(f'Rule {rule_name} satisfies axiom {axiom_name} to {100*cond_sat}%')
        axiom_satisfactions[rule_name] = axiom_satisfaction_current_rule

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"axiom_satisfaction": axiom_satisfactions})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)



    # Comparison rules
    if comparison_rules is not None:
        similarities = train_and_eval.rule_similarity(
            model_rule,
            comparison_rules,
            testing_profiles
        )

        for rule_name in comparison_rules:
            coinc = 100*similarities[rule_name]["identity_accu"]
            print(f'The learned rule coincides with the {rule_name} rule in {coinc}% of the sampled cases')

        with open(f"{location}/results.json") as json_file:
            data = json.load(json_file)

        data.update({"rule_comparison": similarities})

        with open(f"{location}/results.json", "w") as json_file:
            json.dump(data, json_file)

    # Resoluteness
    if compute_resoluteness:

        resoluteness = {}

        # Resoluteness of model
        res_coefficient_of_model = train_and_eval.resoluteness(
            model_rule, 
            testing_profiles
        )
        print(f'The learned rule has resoluteness {res_coefficient_of_model}')
        resoluteness['learned_rule'] = res_coefficient_of_model

        # Resoluteness of rules
        for rule_name in rule_names:
            res_coefficient_of_current_rule = train_and_eval.resoluteness(
                utils.dict_rules_all[rule_name], 
                testing_profiles
            )
            print(f'Rule {rule_name} has resoluteness {res_coefficient_of_current_rule}')
            resoluteness[rule_name] = res_coefficient_of_current_rule

        with open(f"{location}/results.json") as json_file:
            data = json.load(json_file)

        data.update({"resoluteness": resoluteness})

        with open(f"{location}/results.json", "w") as json_file:
            json.dump(data, json_file)




    end_time = time.time()
    duration = end_time - start_time

    print(f'Runtime (in min): {round(duration/60)}')

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"runtime_in_sec": duration})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    wandb_integration.log_summary({
        'test_accuracy': test_accuracy,
        'runtime_sec': duration,
        'num_parameters': num_params,
    })
    wandb_integration.finish_run()

    return location







def experiment1_fixed_data(
        list_of_architectures,
        rule_names,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        training_dataset_size,
        num_epochs,
        eval_dataset_size,
        sample_size_applicable,
        sample_size_maximal,
        list_of_architecture_parameters=None,
        axioms_check_model = list(utils.dict_axioms.keys()),
        axioms_check_rule = list(utils.dict_axioms_rules.keys()),
        max_num_voters_generation=None,
        max_num_alternatives_generation=None,
        merge="accumulative",
        comparison_rules = None,
        compute_resoluteness = False,
        random_seed=None,
        batch_size=64,
        learning_rate = 1e-3,
        save_training_data=False,
        save_model=False,        
    ):
    """
    Implements our experiment 1 but with fixed dataset upfront

    Inputs:
    * `list_of_architectures` is a list of strings that can be `MLP`, `CNN`, or
      `WEC`. The latter two require additional parameters which can be passed 
      via `list_of_architecture_parameters` below.

    * `rule_names`: list of names of rules used for data generation. Typically 
      just a singleton list. If multiple rules are given, their output is 
      merged in the dataset. 
    * `max_num_voters`: a positive integer describing the maximal number of 
      voters that will be considered
    * `max_num_alternatives`: a positive integer describing the maximal number 
      of alternatives that will be considered
    * `election_sampling`: a dictionary describing the parameters for the
      probability model with which profiles are generated. The most important 
      key is `probmodel`. See 
      https://pref-voting.readthedocs.io/en/latest/generate_profiles.html
    
    * `training_dataset_size`: a positive integer describing the number of 
      profiles with their corresponding winning sets that should be generated 
      for training.
    * `num_epochs`: a positive integer describing the number of epochs when 
      training the model
    * `eval_dataset_size`: a positive integer describing the number of profiles
      with their corresponding winning sets that should be used for testing.
    
    * `sample_size_applicable`: a positive integer describing the number of 
      sampled profiles on which the axioms are checked and applicable. 
    * `sample_size_maximal`: a positive integer describing how many profiles 
      are at most sampled when trying to find profiles on which the axioms are 
      applicable. 
    * `axioms_check_model` is a list of names of axioms whose satisfaction is 
      checked for the trained model. By default, it is set to all axioms. If 
      empty, no axioms are checked.
    * `axioms_check_rule` is a list of names of axioms whose satisfaction is 
      checked for the rules. By default, it is only the condorcet and the 
      independence axiom, since the other axioms are always satisfied for 
      common voting rules. If empty, no axioms are checked.   

    * `list_of_architecture_parameters` collects the parameters for each item 
      in the `list_of_architectures`, respectively. For MLP the parameters are 
      None, but CNNs and WEC they are given the following default values (for 
      explanation, see `experiment_1` function).
        * For CNN: {'kernel1':[5,1] , 'kernel2':[1,5], 'channels':64} 
        * For WEC: {'we_size':100, 'we_window':5, 'we_algorithm':1}          
    * `max_num_voters_generation` (optional): a positive integer which is
      <= max_num_voters. It describes the maximal number of voters that will be
      considered *during training*. During testing the full max_num_voters will
      be considered.
    * `max_num_alternatives_generation` (optional): a positive integer which is
      <= max_num_alternatives. It describes the maximal number of alternatives 
      that will be considered *during training*. During testing the 
      full max_num_alternatives will be considered.
    
    * `merge`: By default set to `accumulative`, but could also be `selective`.
      If `rule_names` has length 1 (i.e., only a single rule is considered), 
      both are equivalent. If there are several rules, `selective` means a 
      profile is added to the training dataset only if all rules output the 
      same winning set, while with `accumulative` all profiles are added.
    
    * `comparison_rules` is, if not None, a nonempty list of names of voting 
      rules, to which the rule found by the model is compared to with respect 
      to various measures of similarity.
    * `compute_resoluteness` is False by default. If True, the resoluteness 
      coefficient of the learned rule is computed. A rule is highly resolute 
      if it picks very few winners, and it is not resolute at all if it always 
      declares all alternatives as winners.

    * `random_seed` (optional): If not None, set all random seeds to this 
      provided value.
    * `batch_size`: a positive integer describing the size of the batches when 
      training the model. The default is 64.
    * `learning_rate`: a float number describing the learning rate when 
      training the model. The default value is 1e-3.
    * `save_training_data`: By default False, but if true, the training data 
      will be saved.
    * `save_model`: By default False, but if true, the neural network model 
      will be saved.
    """

    # SET UP BASICS

    start_time = time.time()

    assert (
        max_num_voters_generation is None or max_num_voters_generation <= max_num_voters
    ), f"max_num_voters_generation has to be <= max_num_voters"
    assert (
        max_num_alternatives_generation is None
        or max_num_alternatives_generation <= max_num_alternatives
    ), f"max_num_alternatives_generation has to be <= max_num_alternatives"

    if max_num_voters_generation is None:
        max_num_voters_generation = max_num_voters
    if max_num_alternatives_generation is None:
        max_num_alternatives_generation = max_num_alternatives


    # Set seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    # Set up saving of results
    architectures_short = ""
    for architecture_name in list_of_architectures:
        architectures_short += architecture_name
    rules_short = ""
    for rule_name in rule_names:
        rules_short += rule_name
    prob_model = election_sampling['probmodel']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    location = f"./results/exp1/Fixed/exp1_fixed_{current_time}_{architectures_short}_{rules_short}_{prob_model}"
    os.mkdir(location)
    print(f'Saving location: {location}')


    results = {
        "location": location,
        "list_of_architectures": list_of_architectures,
        "rule_names": rule_names,
        "max_num_voters": max_num_voters,
        "max_num_alternatives": max_num_alternatives,
        "election_sampling": election_sampling,
        "training_dataset_size": training_dataset_size,
        "num_epochs": num_epochs,
        "eval_dataset_size": eval_dataset_size,
        "sample_size_applicable": sample_size_applicable,
        "sample_size_maximal": sample_size_maximal,
        "list_of_architecture_parameters": list_of_architecture_parameters,
        "axioms_check_model": axioms_check_model,
        "axioms_check_rule": axioms_check_rule,
        "max_num_voters_generation": max_num_voters_generation,
        "max_num_alternatives_generation": max_num_alternatives_generation,
        "merge": merge,
        "comparison_rules": comparison_rules,
        "compute_resoluteness": compute_resoluteness,
        "random_seed": random_seed,
        "batch_size": batch_size,        
        "learning_rate": learning_rate,
        "save_training_data": save_training_data,
        "save_model": save_model,
    }

    wandb_integration.init_run("experiment1_fixed", results, location)


    with open(f"{location}/results.json", "w") as json_file:
        json.dump(results, json_file)



    # GENERATING DATA

    # Training dataset
    print("Now generating training data")
    # Generate profiles and their winning sets
    X_train_profs, y_train_wins, sample_rate = generate_profile_data(
        max_num_voters_generation,
        max_num_alternatives_generation,
        training_dataset_size,
        election_sampling,
        [utils.dict_rules_all[rule_name] for rule_name in rule_names],
        merge=merge,
        #progress_report=training_dataset_size / 10,
    )

    # Save sample rate to results
    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"profile_sample_rate": sample_rate})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    # Save training data if requested:
    if save_training_data:
        train_data = utils.voting_data_to_json(X_train_profs,y_train_wins)
        with open(f"{location}/train_data.json", "w") as json_file:
            json.dump(train_data,
                       json_file)


    # Dev dataset

    # Note: we dev on same number of voters/alternatives as for training
    X_dev, y_dev, sample_rate_dev = generate_profile_data(
        max_num_voters_generation,
        max_num_alternatives_generation,
        eval_dataset_size,
        election_sampling,
        [utils.dict_rules_all[rule_name] for rule_name in rule_names],
        merge=merge,
    )


    # Test dataset

    # We test on the same number of voters/alternatives as for training; and, 
    # if the maximum number of voters/alternatives was restricted during 
    # training, we also test on the full number
    X_test, y_test, sample_rate_test = generate_profile_data(
        max_num_voters_generation,
        max_num_alternatives_generation,
        eval_dataset_size,
        election_sampling,
        [utils.dict_rules_all[rule_name] for rule_name in rule_names],
        merge=merge,
    )
    # Save these profiles for later on
    testing_profiles = X_test

    if (max_num_voters_generation < max_num_voters or 
        max_num_alternatives_generation < max_num_alternatives):
        X_full, y_full, sample_rate = generate_profile_data(
            max_num_voters,
            max_num_alternatives,
            eval_dataset_size,
            election_sampling,
            [utils.dict_rules_all[rule_name] for rule_name in rule_names],
            merge=merge,
        )
        # In this case, use these as testing profiles later on
        testing_profiles = X_full


    # NOW LOOP OVER DIFFERENT ARCHITECTURES
    learning_curve = {}

    for i in range(len(list_of_architectures)):
        architecture = list_of_architectures[i]
        architecture_parameters = list_of_architecture_parameters[i]

        print(f'Now architecture {architecture}')

        # TENSORIZE DATA 

        # Tensorize test dataset

        if architecture == 'MLP':
            train_dataloader = generate_data.tensorize_profile_data_MLP(
                X_train_profs, y_train_wins,
                max_num_voters,max_num_alternatives,batch_size
            )

        if architecture == 'CNN':
            train_dataloader = generate_data.tensorize_profile_data_CNN(
                X_train_profs, y_train_wins,
                max_num_voters,max_num_alternatives,batch_size
            )

        if architecture == 'WEC':
            print('Now pretraining word embeddings')

            # First gather architecture parameters
            we_size = architecture_parameters['we_size']
            we_window = architecture_parameters['we_window']
            we_algorithm = architecture_parameters['we_algorithm']

            # Turn set of profiles X into a corpus (each profile a sentence)
            train_sentences = [
                [models.ranking_to_string(ranking)
                    for ranking in profile.rankings]
                for profile in X_train_profs
            ]
            # Add the 'UNK' word for future unknown words. And add 'PAD' for 
            # padding sentences to desired length. (Adding these after training 
            # the embeddings seems inefficient.)
            train_sentences_with_UNK_and_PAD=train_sentences+[['UNK'],['PAD']]

            # Pretrain an word embedding on this corpus
            pre_embeddings = Word2Vec(
                train_sentences_with_UNK_and_PAD,
                vector_size=we_size,
                window=we_window,
                min_count=1,
                workers=8,
                sg=we_algorithm
            )

            # Tensorize and create dataloader
            train_dataloader, _ = generate_data.tensorize_profile_data_WEC(
                pre_embeddings,
                train_sentences,
                y_train_wins,
                max_num_voters,max_num_alternatives,batch_size,
                num_of_unks=False
            )


        # Tensorize dev dataset

        if architecture == 'MLP':
            dev_dataloader = generate_data.tensorize_profile_data_MLP(
                X_dev,y_dev,max_num_voters,max_num_alternatives,batch_size
            )

        if architecture == 'CNN':
            dev_dataloader = generate_data.tensorize_profile_data_CNN(
                X_dev,y_dev,max_num_voters,max_num_alternatives,batch_size
            )

        if architecture == 'WEC':
            dev_sentences = [
                [models.ranking_to_string(ranking) 
                    for ranking in profile.rankings]
                for profile in X_dev
            ]
            dev_dataloader,summary_unks = generate_data.tensorize_profile_data_WEC(
                pre_embeddings,
                dev_sentences,
                y_dev,
                max_num_voters,
                max_num_alternatives,
                batch_size,
                num_of_unks=True
            )

            # Initialize computation of number of UNKs
            number_of_unks = {}
            # Compute number of UNKs in dev set
            ratio = summary_unks['ratio']
            num_unks = summary_unks['num_unks']
            all_words = summary_unks['all_words']
            print(f'Occurrences of UNK in dev data: {ratio}% ({num_unks} of {all_words} words)')
            number_of_unks['num_unks_dev_set'] = summary_unks 


        # Tensorize test dataset

        if architecture == 'MLP':
            test_dataloader = generate_data.tensorize_profile_data_MLP(
                X_test,y_test,max_num_voters,
                max_num_alternatives,batch_size
            )
            if (max_num_voters_generation < max_num_voters or
                max_num_alternatives_generation < max_num_alternatives):
                test_full_dataloader=generate_data.tensorize_profile_data_MLP(
                    X_full,y_full,max_num_voters,
                    max_num_alternatives,batch_size
                )

        if architecture == 'CNN':
            test_dataloader = generate_data.tensorize_profile_data_CNN(
                X_test,y_test,max_num_voters,
                max_num_alternatives,batch_size
            )
            if (max_num_voters_generation < max_num_voters or
                max_num_alternatives_generation < max_num_alternatives):
                test_full_dataloader=generate_data.tensorize_profile_data_CNN(
                    X_full,y_full,max_num_voters,
                    max_num_alternatives,batch_size
                )

        if architecture == 'WEC':
            test_sentences = [
                [models.ranking_to_string(ranking)
                    for ranking in profile.rankings]
                for profile in X_test
            ]
            test_dataloader,summary_unks=generate_data.tensorize_profile_data_WEC(
                pre_embeddings,
                test_sentences,
                y_test,
                max_num_voters,
                max_num_alternatives,
                batch_size,
                num_of_unks=True
            )
            # Compute number of UNKs in test set
            ratio = summary_unks['ratio']
            num_unks = summary_unks['num_unks']
            all_words = summary_unks['all_words']
            print(f'Occurrences of UNK in test data: {ratio}% ({num_unks} of {all_words} words)')
            number_of_unks['num_unks_test_set'] = summary_unks 

            if (max_num_voters_generation < max_num_voters or
                max_num_alternatives_generation < max_num_alternatives):
                test_full_sentences = [
                    [models.ranking_to_string(ranking)
                        for ranking in profile.rankings]
                    for profile in X_full
                ]
                test_full_dataloader,summary_unks=generate_data.tensorize_profile_data_WEC(
                    pre_embeddings,
                    test_full_sentences,
                    y_full,
                    max_num_voters,
                    max_num_alternatives,
                    batch_size,
                    num_of_unks=True
                )
                # Compute number of UNKs in full test set
                ratio = summary_unks['ratio'] 
                num_unks = summary_unks['num_unks']
                all_words = summary_unks['all_words']
                print(f'Occurrences of UNK in full test data: {ratio}% ({num_unks} of {all_words} words)')
                number_of_unks['num_unks_full_test_set'] = summary_unks


            with open(f"{location}/results.json") as json_file:
                data = json.load(json_file)

            data.update({
                architecture:{"number_of_unks": number_of_unks}
            })

            with open(f"{location}/results.json", "w") as json_file:
                json.dump(data, json_file)





        # NEURAL NETWORK TRAINING

        #Initialize model for the experiment
        if architecture == 'MLP':
            exp_model = MLP(max_num_voters, max_num_alternatives)
            exp_loss = nn.BCEWithLogitsLoss()
            exp_optimizer = torch.optim.AdamW(
                exp_model.parameters(), 
                lr=learning_rate
            )

        if architecture == 'CNN':
            exp_model = CNN(
                max_num_voters, 
                max_num_alternatives,
                architecture_parameters['kernel1'],
                architecture_parameters['kernel2'],
                architecture_parameters['channels']
            )
            exp_loss = nn.BCEWithLogitsLoss()
            exp_optimizer = torch.optim.AdamW(
                exp_model.parameters(),
                lr=learning_rate
            )

        if architecture == 'WEC':
            exp_model = WEC(pre_embeddings, max_num_voters, max_num_alternatives)
            exp_loss = nn.BCEWithLogitsLoss()
            exp_optimizer = torch.optim.AdamW(
                exp_model.parameters(),
                lr=learning_rate
            )


        learning_curve = {}

        for epoch in range(num_epochs):
            train_and_eval.train(
                train_dataloader,
                exp_model,
                exp_loss,
                exp_optimizer
            )
            train_accuracy = train_and_eval.accuracy(exp_model, train_dataloader)
            train_loss = train_and_eval.loss(exp_model, exp_loss, train_dataloader)
            dev_accuracy = train_and_eval.accuracy(exp_model, dev_dataloader)
            dev_loss = train_and_eval.loss(exp_model, exp_loss, dev_dataloader)
            learning_curve[f'{epoch}'] = {'train_loss' : train_loss,
                                        'train_accuracy' : train_accuracy,
                                        'dev_loss' : dev_loss,
                                        'dev_accuracy' : dev_accuracy}
            wandb_integration.log({
                f'{architecture}/train_loss': train_loss,
                f'{architecture}/train_accuracy': train_accuracy,
                f'{architecture}/dev_loss': dev_loss,
                f'{architecture}/dev_accuracy': dev_accuracy,
            }, step=epoch)
            if epoch % 5 == 0:
                print(f'Epoch {epoch}: train-loss {round(train_loss,5)}, train-accu {train_accuracy}, dev-loss {round(dev_loss,5)}, dev-accuracy {dev_accuracy}')

        test_accuracy = train_and_eval.accuracy(exp_model, test_dataloader)
        print(f'Done training. Test-accuracy: {test_accuracy}')
        test_results = {'accuracy_on_test_data':test_accuracy}
        if (max_num_voters_generation < max_num_voters or 
            max_num_alternatives_generation < max_num_alternatives):
            test_full_accuracy = train_and_eval.accuracy(
                exp_model,
                test_full_dataloader
            )
            test_results['accuracy_on_test_data_with_more_voters_or_alternatives'] = test_full_accuracy
            print(f'Test-accuracy with more voters/alternatives than during training: {test_full_accuracy}')

        num_params = summary(exp_model).total_params
        print(f'Total number of parameters of the model: {num_params}')

        with open(f"{location}/results.json") as json_file:
            data = json.load(json_file)

        data.update({
            architecture + '_training':{
                "learning curve": learning_curve, 
                "test_accuracy":test_results, 
                "number_of_model_parameters":num_params
            }
        })

        with open(f"{location}/results.json", "w") as json_file:
            json.dump(data, json_file)

        if save_model:
            # We save both the model state and the optimizer state to be able to continue 
            # training later on. 
            torch.save({
                'arguments' : [max_num_voters, max_num_alternatives, architecture_parameters],
                'epoch': epoch,
                'model_state_dict': exp_model.state_dict(),
                'optimizer_state_dict': exp_optimizer.state_dict()
                }, f"{location}/model{architecture}.pth")
                


        # EVALUATION

        # Define the rule computed by the model
        if architecture == 'MLP':
            model_rule = models.MLP2rule(exp_model)

        if architecture == 'CNN':
            model_rule = models.CNN2rule(exp_model)

        if architecture == 'WEC':    
            model_rule = models.WEC2rule(exp_model)


        # Initialize dictionary with axiom satisfactions 
        axiom_satisfactions = {}

        # Axiom satisfaction of model
        axiom_satisfaction_model = {}
        for axiom_name in axioms_check_model:
            sat = train_and_eval.axiom_satisfaction(model_rule,
                    utils.dict_axioms[axiom_name],
                    max_num_voters, 
                    max_num_alternatives,
                    election_sampling,
                    sample_size_applicable,
                    sample_size_maximal,
                    utils.dict_axioms_sample[axiom_name],
                    full_profile=False,
                    comparison_rule=None)
            axiom_satisfaction_model[axiom_name] = sat
            cond_sat = sat['cond_satisfaction']
            print(f'The model satisfies axiom {axiom_name} to {100*cond_sat}%')
        axiom_satisfactions['learned_rule'] = axiom_satisfaction_model

        # Axiom satisfaction of rules
        for rule_name in rule_names:
            axiom_satisfaction_current_rule = {}
            for axiom_name in axioms_check_rule:
                sat = train_and_eval.axiom_satisfaction(
                        utils.dict_rules_all[rule_name],
                        utils.dict_axioms[axiom_name],
                        max_num_voters,
                        max_num_alternatives,
                        election_sampling,
                        sample_size_applicable,
                        sample_size_maximal,
                        utils.dict_axioms_sample[axiom_name],
                        full_profile=False,
                        comparison_rule=None)
                axiom_satisfaction_current_rule[axiom_name] = sat
                cond_sat = sat['cond_satisfaction']
                print(f'Rule {rule_name} satisfies axiom {axiom_name} to {100*cond_sat}%')
            axiom_satisfactions[rule_name] = axiom_satisfaction_current_rule

        with open(f"{location}/results.json") as json_file:
            data = json.load(json_file)

        data.update({
            architecture + '_axioms':{"axiom_satisfaction": axiom_satisfactions}
        })

        with open(f"{location}/results.json", "w") as json_file:
            json.dump(data, json_file)


        # Comparison rules
        if comparison_rules is not None:
            similarities = train_and_eval.rule_similarity(model_rule, comparison_rules, testing_profiles)
            
            for rule_name in comparison_rules:
                coinc = 100*similarities[rule_name]["identity_accu"]
                print(f'The learned rule coincides with the {rule_name} rule in {coinc}% of the sampled cases')
        
            with open(f"{location}/results.json") as json_file:
                data = json.load(json_file)

            data.update({
                architecture + '_rules':{"rule_comparison": similarities}
            })

            with open(f"{location}/results.json", "w") as json_file:
                json.dump(data, json_file)

        # Resoluteness
        if compute_resoluteness:

            resoluteness = {}

            # Resoluteness of model
            res_coefficient_of_model = train_and_eval.resoluteness(model_rule, testing_profiles)
            print(f'The learned rule has resoluteness {res_coefficient_of_model}')
            resoluteness['learned_rule'] = res_coefficient_of_model

            # Resoluteness of rules
            for rule_name in rule_names:
                res_coefficient_of_current_rule = train_and_eval.resoluteness(utils.dict_rules_all[rule_name], testing_profiles)
                print(f'Rule {rule_name} has resoluteness {res_coefficient_of_current_rule}')
                resoluteness[rule_name] = res_coefficient_of_current_rule

            with open(f"{location}/results.json") as json_file:
                data = json.load(json_file)

            data.update({
                architecture + '_resolute':{"resoluteness": resoluteness}
            })

            with open(f"{location}/results.json", "w") as json_file:
                json.dump(data, json_file)




    end_time = time.time()
    duration = end_time - start_time

    print(f'Runtime (in min): {round(duration/60)}')

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"runtime_in_sec": duration})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    wandb_integration.log_summary({'runtime_sec': duration})
    wandb_integration.finish_run()

    return location










def experiment1_cross_validation(
        list_of_architectures,
        rule_name,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        training_dataset_size,
        num_epochs,
        list_of_architecture_parameters,
        num_folds = 10,
        random_seed=None,
        batch_size=200,
        learning_rate = 1e-3,
        save_model=False,        
    ):
    """
    Implements a cross validation for experiment 1 with fixed dataset

    Inputs:
    * `list_of_architectures` is a list of strings that can be `MLP`, `CNN`, or
      `WEC`. The latter two require additional parameters which can be passed 
      via `list_of_architecture_parameters` below.

    * `rule_name`: the rule used for data generation. 
    * `max_num_voters`: a positive integer describing the maximal number of 
      voters that will be considered
    * `max_num_alternatives`: a positive integer describing the maximal number 
      of alternatives that will be considered
    * `election_sampling`: a dictionary describing the parameters for the
      probability model with which profiles are generated. The most important 
      key is `probmodel`. See 
      https://pref-voting.readthedocs.io/en/latest/generate_profiles.html
    
    * `training_dataset_size`: a positive integer describing the number of 
      profiles with their corresponding winning sets that should be generated 
      for training.
    * `num_epochs`: a positive integer describing the number of epochs when 
      training the model
    
    * `list_of_architecture_parameters` collects the parameters for each item 
      in the `list_of_architectures`, respectively. For MLP the parameters are 
      None, but CNNs and WEC they are given the following default values (for 
      explanation, see `experiment_1` function).
        * For CNN: {'kernel1':[5,1] , 'kernel2':[1,5], 'channels':64} 
        * For WEC: {'we_size':100, 'we_window':5, 'we_algorithm':1}      

    * `num_folds`: The number k of folds used for k-fold cross validation. 
      Default is 10.
          
    
    * `random_seed` (optional): If not None, set all random seeds to this 
      provided value.
    * `batch_size`: a positive integer describing the size of the batches when 
      training the model. The default is 64.
    * `learning_rate`: a float number describing the learning rate when 
      training the model. The default value is 1e-3.
    * `save_model`: By default False, but if true, the neural network model 
      will be saved.
    """

    # SET UP BASICS

    start_time = time.time()

    # Set seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    # Set up saving of results
    architectures_short = ""
    for architecture_name in list_of_architectures:
        architectures_short += architecture_name
    prob_model = election_sampling['probmodel']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    location = f"./results/exp1/Fixed/exp1_crossval_{current_time}_{architectures_short}_{rule_name}_{prob_model}"
    os.mkdir(location)
    print(f'Saving location: {location}')


    results = {
        "location": location,
        "list_of_architectures": list_of_architectures,
        "rule_name": rule_name,
        "max_num_voters": max_num_voters,
        "max_num_alternatives": max_num_alternatives,
        "election_sampling": election_sampling,
        "training_dataset_size": training_dataset_size,
        "num_epochs": num_epochs,
        "list_of_architecture_parameters": list_of_architecture_parameters,
        "num_folds":num_folds,
        "random_seed": random_seed,
        "batch_size": batch_size,        
        "learning_rate": learning_rate,
        "save_model": save_model,
    }


    with open(f"{location}/results.json", "w") as json_file:
        json.dump(results, json_file)

    wandb_integration.init_run("experiment1_crossval", results, location)


    # GENERATING DATA

    # Training dataset
    print("Now generating training data")
    data_folds = []
    fold_size = round(training_dataset_size/num_folds)
    for fold in range(num_folds):            
        # Generate profiles and their winning sets
        X, y, sample_rate = generate_profile_data(
            max_num_voters,
            max_num_alternatives,
            fold_size,
            election_sampling,
            [utils.dict_rules_all[rule_name]],
            merge="accumulative",
        )
        data_folds.append((X,y))




    # NOW LOOP OVER DIFFERENT ARCHITECTURES
    
    for i in range(len(list_of_architectures)):
        architecture = list_of_architectures[i]
        architecture_parameters = list_of_architecture_parameters[i]
        

        print(f'Now architecture {architecture}')

        for testing_fold_number in range(num_folds):

            # Compute train and test split
            X_train = []
            y_train = []
            for fold in range(num_folds):
                if fold != testing_fold_number:
                    X_train += data_folds[fold][0]
                    y_train += data_folds[fold][1]
            X_test = data_folds[testing_fold_number][0]
            y_test = data_folds[testing_fold_number][1]

            # TENSORIZE DATA 


            # Tensorize test dataset

            if architecture == 'MLP':
                train_dataloader = generate_data.tensorize_profile_data_MLP(
                    X_train, y_train,
                    max_num_voters,max_num_alternatives,batch_size
                )

            if architecture == 'CNN':
                train_dataloader = generate_data.tensorize_profile_data_CNN(
                    X_train, y_train,
                    max_num_voters,max_num_alternatives,batch_size
                )

            if architecture == 'WEC':
                # print('Now pretraining word embeddings')

                # First gather architecture parameters
                we_size = architecture_parameters['we_size']
                we_window = architecture_parameters['we_window']
                we_algorithm = architecture_parameters['we_algorithm']

                # Turn set of profiles X into a corpus (each profile a sentence)
                train_sentences = [
                    [models.ranking_to_string(ranking) 
                        for ranking in profile.rankings]
                    for profile in X_train
                ]
                # Add the 'UNK' word for future unknown words. And add 'PAD' for 
                # padding sentences to desired length. (Adding these after training 
                # the embeddings seems inefficient.)
                train_sentences_with_UNK_and_PAD=train_sentences+[['UNK'],['PAD']]

                # Pretrain an word embedding on this corpus
                pre_embeddings = Word2Vec(
                    train_sentences_with_UNK_and_PAD, 
                    vector_size=we_size, 
                    window=we_window, 
                    min_count=1, 
                    workers=8, 
                    sg=we_algorithm
                )

                # Tensorize and create dataloader
                train_dataloader, _ = generate_data.tensorize_profile_data_WEC(
                    pre_embeddings,
                    train_sentences,
                    y_train,
                    max_num_voters,max_num_alternatives,batch_size,
                    num_of_unks=False
                )


            # Tensorize test dataset

            if architecture == 'MLP':
                test_dataloader = generate_data.tensorize_profile_data_MLP(
                    X_test,y_test,max_num_voters,max_num_alternatives,batch_size
                )

            if architecture == 'CNN':
                test_dataloader = generate_data.tensorize_profile_data_CNN(
                    X_test,y_test,max_num_voters,max_num_alternatives,batch_size
                )

            if architecture == 'WEC':
                test_sentences = [
                    [models.ranking_to_string(ranking) 
                        for ranking in profile.rankings]
                    for profile in X_test
                ]
                test_dataloader,summary_unks = generate_data.tensorize_profile_data_WEC(
                    pre_embeddings,
                    test_sentences,
                    y_test,
                    max_num_voters,
                    max_num_alternatives,
                    batch_size,
                    num_of_unks=True
                )

                # Initialize computation of number of UNKs
                number_of_unks = {}
                # Compute number of UNKs in test set
                ratio = summary_unks['ratio']
                num_unks = summary_unks['num_unks']
                all_words = summary_unks['all_words']
                # print(f'Occurrences of UNK in test data: {ratio}% ({num_unks} of {all_words} words)')
                number_of_unks['num_unks_test_set'] = summary_unks 


                with open(f"{location}/results.json") as json_file:
                    data = json.load(json_file)

                data.update({
                    architecture:{"number_of_unks": number_of_unks}
                })

                with open(f"{location}/results.json", "w") as json_file:
                    json.dump(data, json_file)


            # NEURAL NETWORK TRAINING

            #Initialize model for the experiment
            if architecture == 'MLP':
                exp_model = MLP(max_num_voters, max_num_alternatives)
                exp_loss = nn.BCEWithLogitsLoss()
                exp_optimizer = torch.optim.AdamW(
                    exp_model.parameters(), 
                    lr=learning_rate
                )

            if architecture == 'CNN':
                exp_model = CNN(
                    max_num_voters, 
                    max_num_alternatives,
                    architecture_parameters['kernel1'],
                    architecture_parameters['kernel2'],
                    architecture_parameters['channels']
                )
                exp_loss = nn.BCEWithLogitsLoss()
                exp_optimizer = torch.optim.AdamW(
                    exp_model.parameters(), 
                    lr=learning_rate
                )

            if architecture == 'WEC':    
                exp_model = WEC(pre_embeddings, max_num_voters, max_num_alternatives)
                exp_loss = nn.BCEWithLogitsLoss()
                exp_optimizer = torch.optim.AdamW(
                    exp_model.parameters(), 
                    lr=learning_rate
                )


            learning_curve = {}

            for epoch in range(num_epochs):
                train_and_eval.train(
                    train_dataloader, 
                    exp_model, 
                    exp_loss, 
                    exp_optimizer
                )
                train_accuracy = train_and_eval.accuracy(exp_model, train_dataloader)
                train_loss = train_and_eval.loss(exp_model, exp_loss, train_dataloader)
                test_accuracy = train_and_eval.accuracy(exp_model, test_dataloader)
                test_loss = train_and_eval.loss(exp_model, exp_loss, test_dataloader)
                learning_curve[f'{epoch}'] = {'train_loss' : train_loss,
                                            'train_accuracy' : train_accuracy,
                                            'test_loss' : test_loss,
                                            'test_accuracy' : test_accuracy} 
            result = {
                'train_loss' : train_loss, 
                'train_accuracy' : train_accuracy , 
                'test_loss' : test_loss,
                'test_accuracy' : test_accuracy
            }
            print(f'Fold {testing_fold_number}: train-accu {train_accuracy}, test-accu {test_accuracy}')

            num_params = summary(exp_model).total_params


            with open(f"{location}/results.json") as json_file:
                data = json.load(json_file)

            data.update({
                architecture + f'_fold_{testing_fold_number}':{
                    "learning curve": learning_curve, 
                    "result": result,
                    "number_of_model_parameters":num_params
                }
            })

            with open(f"{location}/results.json", "w") as json_file:
                json.dump(data, json_file)

            if save_model:
                # We save both the model state and the optimizer state to be 
                # able to continue training later on. 
                torch.save({
                    'arguments' : [max_num_voters, max_num_alternatives, architecture_parameters],
                    'epoch': epoch,
                    'model_state_dict': exp_model.state_dict(),
                    'optimizer_state_dict': exp_optimizer.state_dict()
                    }, f"{location}/model{architecture}.pth")
                    





    end_time = time.time()
    duration = end_time - start_time

    print(f'Runtime (in min): {round(duration/60)}')

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"runtime_in_sec": duration})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    wandb_integration.log_summary({'runtime_sec': duration})
    wandb_integration.finish_run()

    return location














def experiment1_accu_and_axioms_along_training(
        architecture,
        rule_names,
        max_num_voters,
        max_num_alternatives,
        election_sampling,
        num_gradient_steps,
        eval_dataset_size,
        sample_size_applicable,
        sample_size_maximal,
        architecture_parameters=None,
        axioms_check_model = list(utils.dict_axioms.keys()),
        merge="accumulative",
        random_seed=None,
        report_interval = 1000,
        batch_size=200,
        learning_rate = 1e-3,
        learning_scheduler = None,
        weight_decay = 0,
        save_model=False,        
    ):
    """
    Implements the development of accuracy & axiom sat. across gradient steps
    
    Inputs:
    * `architecture` can be either `MLP`, `MLP_small`, `MLP_large`, `CNN`, or 
      `WEC`. The latter two require additional parameters which can be passed 
      as `architecture_parameters` below (otherwise default values are chosen).

    * `rule_names`: list of names of rules used for data generation. Typically 
      just a singleton list. If multiple rules are given, their output is 
      merged in the dataset. 
    * `max_num_voters`: a positive integer describing the maximal number of 
      voters that will be considered
    * `max_num_alternatives`: a positive integer describing the maximal number 
      of alternatives that will be considered
    * `election_sampling`: a dictionary describing the parameters for the
      probability model with which profiles are generated. The most important 
      key is `probmodel`. See 
        https://pref-voting.readthedocs.io/en/latest/generate_profiles.html
    
    * `num_gradient_steps`: a positive integer describing the number of 
      gradient steps performed during training the model.
    * `eval_dataset_size`: a positive integer describing the number of profiles
      with their corresponding winning sets that should be used for testing.
    
    * `sample_size_applicable`: a positive integer describing the number of 
      sampled profiles on which the axioms are checked and applicable. 
    * `sample_size_maximal`: a positive integer describing how many profiles 
      are at most sampled when trying to find profiles on which the axioms are 
      applicable. 

    * `architecture_parameters` is not needed for MLPs, but CNNs and WEC they 
      are given the following default values
        * For CNN:
          {
            'kernel1':[5,1] , 
            'kernel2':[1,5], 
            'channels':32
          }
          Here `kernel1`: a list [height, width] of the dimension for the 
          kernel/filter in the first convolutional layer of the model. (The 
          height of the images is max_num_alternatives and the width of the 
          images is max_num_voters.) If max num voters/alternatives are quite 
          small, the kernel cannot be too big, otherwise error will be raised.  
          Similarly, `kernel2` is the [height, width]-dimension for the 
          kernel/filter in the second convolutional layer of the model.
          Finally, `channels` is the number of channels of the feature maps of 
          the model.
        * For WEC: 
          {
            'we_corpus_size':int(1e5), 
            'we_size':100, 
            'we_window':5, 
            'we_algorithm':1
          } 
          Here 'we_corpus_size' is the number of profiles used to pretrain the 
          word embeddings,  `we_size` is the size (i.e., length of vector) of 
          the word embeddings, `we_window` is the size of the window used when 
          training the word embeddings. Finally, `we_algorithm` is either 0 or 
          1 depending on whether one uses the CBOW algorithm or the skip gram 
          algorithm.
              
    * `axioms_check_model` is a list of names of axioms whose satisfaction is 
      checked during training. By default, it is set to all axioms. If 
      empty, no axioms are checked.
    
    * `merge`: By default set to `accumulative`, but could also be `selective`.
      If `rule_names` has length 1 (i.e., only a single rule is considered), 
      both are equivalent. If there are several rules, `selective` means a 
      profile is added to the training dataset only if all rules output the 
      same winning set, while with `accumulative` all profiles are added.

    * `random_seed` (optional): If not None, set all random seeds to this 
      provided value.
    * `report_interval` by default 1000 saying that after every 1000 gradient
      steps the accuracy and axiom satisfaction is computed on the dev-dataset.
    * `batch_size`: a positive integer describing the size of the batches when 
      training the model. The default is 200.
    * `learning_rate`: a float number describing the learning rate when 
      training the model. The default value is 1e-3.
    * `learning_scheduler`: By default None, but if given, then a positive 
      integer describing the T_0 value for the CosineAnnealingWarmRestarts 
      scheduler (i.e., the number of iterations for the first restart).
    * `weight_decay` of the optimizer which we set by default to 0 since we use 
      synthetic data and hence don't need regularization (its usual default 
      value is 0.01).  
    * `save_model`: By default False, but if true, the neural network model 
      will be saved.
    """

    # SET UP BASICS

    start_time = time.time()

    assert (
        architecture in ['MLP', 'MLP_small', 'MLP_large', 'CNN', 'WEC']
    ), f"The supported architectures are 'MLP', 'MLP_small', 'MLP_large', 'CNN', and 'WEC' but {architecture} was given"

    if architecture_parameters is None:
        if architecture == 'CNN':
            architecture_parameters = {
                'kernel1':[5,1] , 
                'kernel2':[1,5], 
                'channels':32} 
        if architecture == 'WEC':
            architecture_parameters = {
                'we_corpus_size':int(1e5),
                'we_size':100, 
                'we_window':5, 
                'we_algorithm':1}

        

    # Set seeds
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


    # Set up saving of results
    rules_short = ""
    for rule_name in rule_names:
        rules_short += rule_name
    prob_model = election_sampling['probmodel']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    location = f"./results/exp1/Evol/exp1_{current_time}_{architecture}_{rules_short}_{prob_model}"
    os.mkdir(location)
    print(f'Saving location: {location}')


    results = {
        "location": location,        
        "architecture": architecture,
        "rule_names": rule_names,
        "max_num_voters": max_num_voters,
        "max_num_alternatives": max_num_alternatives,
        "election_sampling": election_sampling,
        "num_gradient_steps": num_gradient_steps,
        "eval_dataset_size": eval_dataset_size,
        "sample_size_applicable": sample_size_applicable,
        "sample_size_maximal": sample_size_maximal,
        "architecture_parameters": architecture_parameters,
        "axioms_check_model": axioms_check_model,
        "merge": merge,
        "random_seed": random_seed,
        "report_interval": report_interval,
        "batch_size": batch_size,        
        "learning_rate": learning_rate,
        "learning_scheduler": learning_scheduler,
        "weight_decay": weight_decay,
        "save_model": save_model,
    }


    with open(f"{location}/results.json", "w") as json_file:
        json.dump(results, json_file)

    wandb_integration.init_run("experiment1_along_training", results, location)


    # GENERATING DATA

    # Training data will be generated before each training batch
    # Only WEC first needs to pretrain word embeddings

    if architecture == 'WEC':
        # First gather architecture parameters
        we_corpus = architecture_parameters['we_corpus_size']
        we_size = architecture_parameters['we_size']
        we_window = architecture_parameters['we_window']
        we_algorithm = architecture_parameters['we_algorithm']
        load_embeddings_from = architecture_parameters.get(
                'load_embeddings_from', None
            ) 


        if load_embeddings_from is None:
            print('Now pretraining word embeddings')

            print("First generate profiles and turn them into corpus")
            # Generate profiles and their winning sets
            X_train_profs, _ , _ = generate_profile_data(
                max_num_voters,
                max_num_alternatives,
                we_corpus,
                election_sampling,
                [],
                merge='empty'
            )

            # Turn set of profiles X into a corpus (each profile a sentence)
            train_sentences = [
                    [models.ranking_to_string(ranking)
                    for ranking in profile.rankings]
                for profile in X_train_profs]
            # Add the 'UNK' word for future unknown words. And add 'PAD' for 
            # padding sentences to desired length. (Adding these after training 
            # the embeddings seems inefficient.)
            train_sentences_with_UNK_and_PAD=train_sentences+[['UNK'], ['PAD']]

            # Pretrain an word embedding on this corpus
            print('Now train word embeddings')
            pre_embeddings = Word2Vec(
                    train_sentences_with_UNK_and_PAD,
                    vector_size=we_size,
                    window=we_window,
                    min_count=1,
                    workers=8,
                    sg=we_algorithm
                )
            print('Done pretraining word embeddings.')

            if save_model:
                print('Save the word embeddings.')
                # We save the pre_embedding word2vec model, so it can be used
                # when initializing a WEC model that we then load with 
                # previously trained parameters 
                pre_embeddings.save(f"{location}/pre_embeddings.bin")

        # Load back with memory-mapping = read-only, shared across processes.
        if load_embeddings_from is not None:
            print('Load the word embeddings.')
            load_embeddings_from = architecture_parameters['load_embeddings_from']
            pre_embeddings = Word2Vec.load(f"{load_embeddings_from}/pre_embeddings.bin")





    # Dev dataset
    print('Now generate dev and test data')
    # Note: we dev on same number of voters/alternatives as for training
    X, y, sample_rate = generate_profile_data(
        max_num_voters,
        max_num_alternatives,
        eval_dataset_size,
        election_sampling,
        [utils.dict_rules_all[rule_name] for rule_name in rule_names],
        merge=merge,
    )

    if architecture in ['MLP', 'MLP_small', 'MLP_large']:
        dev_dataloader = generate_data.tensorize_profile_data_MLP(
            X,y,max_num_voters,max_num_alternatives,batch_size
        )

    if architecture == 'CNN':
        dev_dataloader = generate_data.tensorize_profile_data_CNN(
            X,y,max_num_voters,max_num_alternatives,batch_size
        )

    if architecture == 'WEC':
        dev_sentences = [[models.ranking_to_string(ranking) 
                           for ranking in profile.rankings] 
                         for profile in X]
        dev_dataloader,summary_unks = generate_data.tensorize_profile_data_WEC(
            pre_embeddings,
            dev_sentences,
            y,
            max_num_voters,
            max_num_alternatives,
            batch_size,
            num_of_unks=True
        )

        # Initialize computation of number of UNKs
        number_of_unks = {}
        # Compute number of UNKs in dev set
        ratio = summary_unks['ratio']
        num_unks = summary_unks['num_unks']
        all_words = summary_unks['all_words']
        print(f'Occurrences of UNK in dev data: {ratio}% ({num_unks} of {all_words} words)')
        number_of_unks['num_unks_dev_set'] = summary_unks



    # Test dataset
    # We test on the same number of voters/alternatives as for training;
    # and, if the maximum number of voters/alternatives was restricted during
    # training, we also test on the full number
    X, y, sample_rate = generate_profile_data(
        max_num_voters,
        max_num_alternatives,
        eval_dataset_size,
        election_sampling,
        [utils.dict_rules_all[rule_name] for rule_name in rule_names],
        merge=merge,
    )
    # Save these profiles for later on
    testing_profiles = X




    if architecture in ['MLP', 'MLP_small', 'MLP_large']:
        test_dataloader = generate_data.tensorize_profile_data_MLP(
            X,y,max_num_voters,max_num_alternatives,batch_size
        )
        
        
    if architecture == 'CNN':
        test_dataloader = generate_data.tensorize_profile_data_CNN(
            X,y,max_num_voters,max_num_alternatives,batch_size
        )
        
        


    if architecture == 'WEC':    
        test_sentences = [[models.ranking_to_string(ranking) 
                            for ranking in profile.rankings] 
                          for profile in X]
        test_dataloader,summary_unks=generate_data.tensorize_profile_data_WEC(
            pre_embeddings,
            test_sentences,
            y,
            max_num_voters,
            max_num_alternatives,
            batch_size,
            num_of_unks=True
        )
        # Compute number of UNKs in test set
        ratio = summary_unks['ratio']
        num_unks = summary_unks['num_unks']
        all_words = summary_unks['all_words']
        print(f'Occurrences of UNK in test data: {ratio}% ({num_unks} of {all_words} words)')
        number_of_unks['num_unks_test_set'] = summary_unks 




        with open(f"{location}/results.json") as json_file:
            data = json.load(json_file)

        data.update({"number_of_unks": number_of_unks})

        with open(f"{location}/results.json", "w") as json_file:
            json.dump(data, json_file)





    # NEURAL NETWORK TRAINING

    #Initialize our model for the experiment
    if architecture == 'MLP':
        exp_model = MLP(max_num_voters, max_num_alternatives)
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

    if architecture == 'MLP_small':
        exp_model = MLP_small(max_num_voters, max_num_alternatives)
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

    if architecture == 'MLP_large':
        exp_model = MLP_large(max_num_voters, max_num_alternatives)
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

    if architecture == 'CNN':
        exp_model = CNN(
            max_num_voters, 
            max_num_alternatives,
            architecture_parameters['kernel1'],
            architecture_parameters['kernel2'],
            architecture_parameters['channels']
        )
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if architecture == 'WEC':
        exp_model = WEC(pre_embeddings, max_num_voters, max_num_alternatives)
        exp_loss = nn.BCEWithLogitsLoss()
        exp_optimizer = torch.optim.AdamW(
            exp_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    if learning_scheduler is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            exp_optimizer,
            T_0 = learning_scheduler
        )


    print('Now starting to train')
    evolution = {}
    
    for step in tqdm(range(num_gradient_steps)):
        # Generate data for the batch

        X_train_profs, y_train_wins, _ = generate_profile_data(
            max_num_voters,
            max_num_alternatives,
            batch_size,
            election_sampling,
            [utils.dict_rules_all[rule_name] for rule_name in rule_names],
            merge=merge,
        )

        if architecture in ['MLP', 'MLP_small', 'MLP_large']:
            train_dataloader = generate_data.tensorize_profile_data_MLP(
                X_train_profs, y_train_wins,
                max_num_voters,max_num_alternatives,batch_size
            )

        if architecture == 'CNN':
            train_dataloader = generate_data.tensorize_profile_data_CNN(
                X_train_profs, y_train_wins,
                max_num_voters,max_num_alternatives,batch_size
            )

        if architecture == 'WEC':
            train_sentences = [[models.ranking_to_string(ranking)
                                for ranking in profile.rankings]
                               for profile in X_train_profs]
            train_dataloader, _ = generate_data.tensorize_profile_data_WEC(
                pre_embeddings,
                train_sentences,
                y_train_wins,
                max_num_voters,max_num_alternatives,batch_size,
                num_of_unks=False
            )

        # Do step in gradient descent
        train_and_eval.train(
            train_dataloader,
            exp_model, 
            exp_loss, 
            exp_optimizer
        )

        if learning_scheduler is not None:
            scheduler.step()

        if step % report_interval == 0:
            
            # Dev accuracy
            dev_accuracy = train_and_eval.accuracy(exp_model, dev_dataloader)
            dev_loss = train_and_eval.loss(exp_model, exp_loss, dev_dataloader)
            

            # Dev axiom satisfaction

            # Define the rule computed by the model
            if architecture in ['MLP', 'MLP_small', 'MLP_large']:
                model_rule = models.MLP2rule(exp_model)
                model_rule_full = models.MLP2rule(exp_model, full=True)

            if architecture == 'CNN':
                model_rule = models.CNN2rule(exp_model)
                model_rule_full = models.CNN2rule(exp_model, full=True)

            if architecture == 'WEC':    
                model_rule = models.WEC2rule(exp_model)
                model_rule_full = models.WEC2rule(exp_model, full=True)


            # Admissibility
            admissibility_summary = train_and_eval.admissibility(
                model_rule_full,
                testing_profiles
            )

            
            # Initialize dictionary with axiom satisfactions 
            axiom_satisfactions = {}

            # Axiom satisfaction of model
            axiom_satisfaction_model = {}
            for axiom_name in axioms_check_model:
                sat = train_and_eval.axiom_satisfaction(model_rule,
                        utils.dict_axioms[axiom_name],
                        max_num_voters, 
                        max_num_alternatives, 
                        election_sampling, 
                        sample_size_applicable, 
                        sample_size_maximal, 
                        utils.dict_axioms_sample[axiom_name], 
                        full_profile=False,
                        comparison_rule=None)
                axiom_satisfaction_model[axiom_name] = sat

            # Gather everything
            evolution[f'{step}'] = {
                'dev_loss' : dev_loss,
                'dev_accuracy' : dev_accuracy,
                'dev_admissibility' : admissibility_summary,
                'dev_axiom_satisfaction' : axiom_satisfaction_model,
            }
            wandb_metrics = {
                'dev_loss': dev_loss,
                'dev_accuracy': dev_accuracy,
            }
            for ax_name, ax_val in axiom_satisfaction_model.items():
                wandb_metrics[f'axiom/{ax_name}'] = ax_val.get('cond_satisfaction', 0)
            for k, v in admissibility_summary.items():
                wandb_metrics[f'admissibility/{k}'] = v
            wandb_integration.log(wandb_metrics, step=step)


    num_params = summary(exp_model).total_params

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({
        "evolution": evolution, 
        "number_of_model_parameters":num_params
    })

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    if save_model:
        # We save both the model state and the optimizer state to be able to 
        # continue training later on.
        torch.save({
            'arguments' : [max_num_voters, max_num_alternatives, architecture_parameters],
            'model_state_dict': exp_model.state_dict(),
            'optimizer_state_dict': exp_optimizer.state_dict()
            }, f"{location}/model.pth")





    end_time = time.time()
    duration = end_time - start_time

    print(f'Runtime (in min): {round(duration/60)}')

    with open(f"{location}/results.json") as json_file:
        data = json.load(json_file)

    data.update({"runtime_in_sec": duration})

    with open(f"{location}/results.json", "w") as json_file:
        json.dump(data, json_file)

    wandb_integration.log_summary({'runtime_sec': duration, 'num_parameters': num_params})
    wandb_integration.finish_run()

    return location