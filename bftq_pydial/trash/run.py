# coding=utf-8
# matplotlib.use("template")
import matplotlib.pyplot as plt
import json
import configparser
import os
import bftq_pydial.algorithms.pytorch_fittedq as pftq
import bftq_pydial.algorithms.pytorch_budgeted_fittedq as pbf
import torch.nn.functional as F

import numpy as np
from sklearn.model_selection import ParameterGrid
import bftq_pydial.tools.features as feat
from bftq_pydial.tools.policies import RandomPolicy, PytorchBudgetedFittedPolicy, PytorchFittedPolicy

np.set_printoptions(precision=4)
import shutil
from bftq_pydial.tools import torch_utils as tu

import bftq_pydial.tools.utils_run_pydial as urpy

from gym_pydial.env.env_pydial import EnvPydial

import sys

print(sys.argv)

tu.set_device()
np.set_printoptions(precision=4)

if len( sys.argv) >1:
    basename_config_file =  sys.argv[1]
else:
    # basename_config_file = "run_test.cfg"
    basename_config_file = "3_run_env1.cfg"
    basename_config_file = "0_run_env6.cfg"
    basename_config_file = "0_run_env1_CAM_BFTQ.cfg"
    # basename_config_file = "test_run_env6_LAP.cfg4"
    basename_config_file = "test_run_env6_LAP.cfg"
config_file = "config_main_pydial/" + basename_config_file


config = configparser.ConfigParser()
config.read(config_file)
section = "RUN"
activation = config.get(section, "activation_type")
reset_type = config.get(section, "reset_type")
delta = config.getfloat(section, "delta")
nn_loss_stop = config.getfloat(section, "nn_loss_stop")
gamma = config.getfloat(section, "gamma")
gamma_c = config.getfloat(section, "gamma_c")
max_nn_epoch = config.getint(section, "max_nn_epoch")
normalize_reward = config.getboolean(section, "normalize_reward")
normalize = config.getboolean(section, "normalize")
reset_policy_before_regression = config.getboolean(section, "reset_policy_before_regression")
reset_weight = config.getboolean(section, "reset_weight")
weights_losses = json.loads(config.get(section, "weights_losses"))
type_beta = config.get(section, "type_beta")
data_filename = config.get(section, "data_filename")
beta_encoder_type = config.get(section, "beta_encoder_type")
NB_BATCHS = config.getint(section, "NB_BATCHS")
dialogues_by_batch = config.getint(section, "dialogues_by_batch")
batch_size_er = config.get(section, "batch_size_er")
N_dialogues_test = config.getint(section, "dialogues_test")
dialogues_test_intra_epoch = config.getint(section, "dialogues_test_intra_epoch")
# size_state = config.getint(section, "size_state")
layers = json.loads(config.get(section, "layers"))
test_bftq = config.getboolean(section, "test_bftq")
test_ftq = config.getboolean(section, "test_ftq")
clip_qc_output = config.getboolean(section, "clip_qc_output")
print_q_function = config.getboolean(section, "print_q_function")
disp = config.getboolean(section, "disp")
print_dial = config.getboolean(section, "print_dial")
seed = config.getint("GENERAL", "seed")

loss_function = config.get("RUN", "loss_function")
loss_function_c = config.get("RUN", "loss_function_c")
# reward_domain = tuple(json.loads(config.get(section, "reward_domain")))
#
size_betas_layer = tuple(json.loads(config.get(section, "size_betas_layer")))

lambda_ = tuple(json.loads(config.get(section, "lambda_")))
emax = tuple(json.loads(config.get(section, "emax")))
learning_rate = tuple(json.loads(config.get(section, "learning_rate")))
optimizer = tuple(json.loads(config.get(section, "optimizer")))
path_sample_data = tuple(json.loads(config.get(section, "path_sample_data")))
weight_decay = tuple(json.loads(config.get(section, "weight_decay")))
nb_betas = tuple(json.loads(config.get(section, "nb_betas")))
feature = tuple(json.loads(config.get(section, "feature")))
new_style = tuple(json.loads(config.get(section, "new_style")))
maxturns = config.getint("agent", "maxturns")

param_grid = {'size_betas_layer':size_betas_layer,
              'new_style': new_style,
              'layers': layers,
              'feature': feature,
              'lambda_': lambda_,
              # 'seed': seed,
              'emax': emax,
              'learning_rate': learning_rate,
              'optimizer': optimizer,
              'weight_decay': weight_decay,
              'nb_betas': nb_betas,
              'path_sample_data': path_sample_data}

grid = ParameterGrid(param_grid)

e = EnvPydial(config_file=config_file, error_rate=0.3)

path = "tmp/" + basename_config_file
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)
# print(grid)
with open("tmp/" + basename_config_file + "/grid_of_params", 'a') as infile:
    infile.write(''.join([str(param) + "\n" for param in grid]))

for il, layer in enumerate(layers):
    layers[il] = tuple(layer)

print(layers)

param_to_save = []


def go(id, size_betas_layer,new_style, layers, feature, lambda_,  emax, learning_rate, optimizer, weight_decay, nb_betas,
       path_sample_data,seed=seed,
       activation=activation, reset_type=reset_type, delta=delta, nn_loss_stop=nn_loss_stop
       , gamma=gamma, gamma_c=gamma_c, max_nn_epoch=max_nn_epoch, clamp_Qr=None,
       normalize_reward=normalize_reward, normalize=normalize,
       reset_policy_before_regression=reset_policy_before_regression,
       reset_weight=reset_weight, weights_losses=weights_losses, type_beta=type_beta,
       beta_encoder_type=beta_encoder_type,
       NB_BATCHS=NB_BATCHS, dialogues_by_batch=dialogues_by_batch, batch_size_er=batch_size_er,
       N_dialogues_test=N_dialogues_test, test_bftq=test_bftq, test_ftq=test_ftq,
       clip_qc_output=clip_qc_output,
       print_q_function=print_q_function, disp=disp,
       len_action_space=None,
       loss_function=loss_function, loss_function_c=loss_function_c):
    # e.change_style(new_style=new_style)
    e.reset()
    print("available actions : ", e.action_space())
    urpy.set_seed(seed)
    print("lambda_", lambda_)
    params = [id, new_style, layers, feature, lambda_, seed, emax, learning_rate, optimizer, weight_decay, nb_betas,
              path_sample_data]
    print ["i", "new_style", "layers", "feature", "lambda_", "seed", "emax", "learning_rate", "optimizer",
           "weight_decay", "nb_betas",
           "path_sample_data"]
    print(params)
    param_to_save.append(params)
    id_params = str(id)
    path = "tmp/" + basename_config_file + "/" + id_params
    os.makedirs(path)
    print( path)
    action_space = e.action_space()
    activation = F.relu if activation == "RELU" else None
    loss_function = F.mse_loss if loss_function == "MSE_LOSS" else None
    loss_function_c = F.mse_loss if loss_function_c == "MSE_LOSS" else None
    if feature == "feature_0":
        feature = feat.feature_0
    elif feature == "feature_1":
        feature = feat.feature_1
    elif feature == "feature_2":
        feature = feat.feature_2
    elif feature == "feature_3":
        feature = feat.feature_3
    elif feature == "feature_4":
        feature = feat.feature_4
    else:
        raise Exception("Unknown feature : {}".format(feature))

    if nb_betas == 0:
        betas = [0.]
    else:
        if type_beta == "EXPONENTIAL":
            betas = np.concatenate(
                (
                    np.array([0.]), np.exp(np.power(np.linspace(0, nb_betas, nb_betas), np.full(nb_betas, 2. / 3.))) / (
                        np.exp(np.power(nb_betas, 2. / 3.)))))
        elif type_beta == "LINSPACE":
            betas = np.linspace(0, 1, nb_betas + 1)
        else:
            raise Exception("type_beta inconnu : " + str(type_beta))
    print("betas :", betas)

    def process_between_epoch(pi):
        print("process_between_epoch ...")
        pi = PytorchFittedPolicy(pi, action_space, e, feature)
        _, results = urpy.execute_policy(e, pi, gamma, gamma_c, dialogues_test_intra_epoch, 1., False)
        return np.mean(results, axis=0)

    plt.plot(range(0, len(betas)), betas)
    plt.savefig(path + "/betas")
    plt.close()
    size_state = len(feature(e.reset(), e))
    print("neural net input size :", size_state)

    policy_network = pftq.NetFTQ(
        (size_state,) + layers + (len(action_space),),
        activation,
        normalize,
        clamp_Q=None)

    policy_network_bftq = pbf.NetBFTQ(
        size_state,
        size_betas_layer,
        layers + (2 * len(action_space),),
        activation,
        normalize,
        reset_type,
        beta_encoder_type,
        clamp_Qr=None,
        clamp_Qc=None)

    if optimizer == "RMS_PROP":
        import torch
        optimizer_ftq = torch.optim.RMSprop(policy_network.parameters(), weight_decay=weight_decay)
        optimizer_bftq = torch.optim.RMSprop(policy_network_bftq.parameters(), weight_decay=weight_decay)

    elif optimizer == "ADAM":
        import torch
        optimizer_ftq = torch.optim.Adam(policy_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_bftq = torch.optim.Adam(policy_network_bftq.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
    else:
        raise Exception("unknow optimizer {}".format(optimizer))

    ftq = pftq.PytorchFittedQ(
        policy_network=policy_network,
        gamma=gamma,
        max_ftq_epoch=emax,
        reset_policy__each_ftq_epoch=reset_policy_before_regression,  # False,  # True,
        max_nn_epoch=max_nn_epoch,
        optimizer=optimizer_ftq,
        loss_function=loss_function,
        delta=delta,
        batch_size_experience_replay=batch_size_er,
        workspace=path,
        disp_states=[],
        action_str=e.action_space_str(),
        process_between_epoch=process_between_epoch
        # optimizer=torch.optim.Adam(policy_network.parameters())
    )

    bftq = pbf.PytorchBudgetedFittedQ(
        print_q_function=print_q_function,
        policy_network=policy_network_bftq,
        betas=betas,
        betas_for_discretisation=betas,
        N_actions=len(e.action_space()),
        actions_str=e.action_space_str(),
        gamma=gamma,
        gamma_c=gamma_c,
        max_ftq_epoch=emax,
        reset_policy_after_each_ftq_epoch=reset_policy_before_regression,  # True,
        max_nn_epoch=max_nn_epoch,
        loss_function=loss_function,
        loss_function_c=loss_function_c,
        optimizer=optimizer_bftq,
        disp=disp,
        delta_stop=delta,
        batch_size_experience_replay=batch_size_er,
        nn_loss_stop_condition=nn_loss_stop,
        workspace=path,
        disp_states=[],
        weights_losses=weights_losses,
        clip_qc_output=clip_qc_output
    )
    if path_sample_data !="":
        print ("reading json data file of samples in {}".format(path_sample_data))
        with open(path_sample_data, 'r') as infile:
            datass = json.load(infile)
        datas = [None] * len(datass)
        for idata, data in enumerate(datass):
            datas[idata] = urpy.Data(**data)
    else:
        print ("Generating datas with Random policy")
        pi_epsilon_greedy = RandomPolicy()
        dialogues, _ = urpy.execute_policy(e, pi_epsilon_greedy, gamma, gamma_c,
                                                 N_dialogues=dialogues_by_batch)
        datas = urpy.dialogues_to_datas(dialogues)

    transitions_ftq, transitions_bftq = urpy.datas_to_transitions(datas, e, feature, lambda_, normalize_reward)
    # learning
    if test_ftq:
        print("[LEARNING FTQ PI GREEDY] #samples={}".format(len(transitions_ftq)))
        ftq.reset(reset_weight)
        pi = ftq.fit(transitions_ftq)
        pi_greedy = PytorchFittedPolicy(pi, action_space, e, feature)
        dialogues, results = urpy.execute_policy(e, pi_greedy, gamma, gamma_c, N_dialogues=N_dialogues_test,
                                                 print_dial=print_dial)
        urpy.print_results(results)

    results_bftq = [None] * len(betas)
    if test_bftq:
        print("[LEARNING BFTQ PI GREEDY] #samples={}".format(len(transitions_bftq)))
        bftq.reset(reset_weight)
        pi = bftq.fit(transitions_bftq)
        pi_greedy = PytorchBudgetedFittedPolicy(pi, action_space, e, feature)

        for ibeta, beta in enumerate(betas):
            dialogues, results = urpy.execute_policy(e, pi_greedy, gamma, gamma_c,
                                                     beta=beta,
                                                     N_dialogues=N_dialogues_test,
                                                     print_dial=print_dial)
            print ("beta={}".format(beta))
            urpy.print_results(results)
            results_bftq[ibeta] = results

    return params, results, results_bftq, betas


i = 0

str_params = ''

for params in grid:
    p, data_ftq, data_bftq, betas = go(i,
                                       params['size_betas_layer'],
                                       params['new_style'], params['layers'], params['feature'], params['lambda_'],
                                       params['seed'], params['emax'], params['learning_rate'], params['optimizer'],
                                       params['weight_decay'], params['nb_betas'], params['path_sample_data'])
    str_params += str(i) + ' ' + ''.join([str(param) + ' ' for param in params.values()]) + '\n'
    if test_ftq:
        np.save("tmp/{}/{}/data_ftq".format(basename_config_file, i), data_ftq)
    if test_bftq:
        for ibeta, beta in enumerate(betas):
            np.save("tmp/{}/{}/data_bftq_beta={}".format(basename_config_file, i, beta), data_bftq[ibeta])
        # np.save("tmp/{}/{}/data_bftq".format(basename_config_file, i), data_bftq)
    i += 1
    # params_str+="\n"
with open("tmp/" + basename_config_file + "/params", 'a') as infile:
    infile.write(str_params)
