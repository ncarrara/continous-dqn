# # -*- coding: utf-8 -*-
# import matplotlib
# # matplotlib.use("template")
# from algorithm.online.dqn import DQN
#
#
# import matplotlib.pyplot as plt
# import json
# import ConfigParser
# import os
# import algorithm.batch.ftq.pytorch_fittedq as pftq
# import torch.nn.functional as F
# import numpy as np
# from sklearn.model_selection import ParameterGrid
# import features as feat
# from algorithm.online.dqn import Transition
#
# np.set_printoptions(precision=4)
# import shutil
# from tools import torch_utils as tu
# from env.env_pydial.env_pydial import EnvPydial
# import utils_run_pydial as urpy
#
# import sys
#
# print sys.argv
#
# tu.set_device()
# np.set_printoptions(precision=4)
#
# if len(sys.argv) > 1:
#     basename_config_file = sys.argv[1]
# else:
#     # basename_config_file = "run_test.cfg"
#     basename_config_file = "test_dqn2.cfg"
# # basename_config_file = "run0.cfg"
# config_file = "config_main_pydial/" + basename_config_file
#
# config = ConfigParser.ConfigParser()
# config.read(config_file)
# section = "RUN"
# activation = config.get(section, "activation_type")
# reset_type = config.get(section, "reset_type")
# delta = config.getfloat(section, "delta")
# nn_loss_stop = config.getfloat(section, "nn_loss_stop")
# gamma = config.getfloat(section, "gamma")
# gamma_c = config.getfloat(section, "gamma_c")
# max_nn_epoch = config.getint(section, "max_nn_epoch")
# clamp_Qr = config.get(section, "clamp_Qr")
# normalize_reward = config.getboolean(section, "normalize_reward")
# normalize = config.getboolean(section, "normalize")
# reset_policy_before_regression = config.getboolean(section, "reset_policy_before_regression")
# reset_weight = config.getboolean(section, "reset_weight")
# weights_losses = json.loads(config.get(section, "weights_losses"))
# type_beta = config.get(section, "type_beta")
# data_filename = config.get(section, "data_filename")
# beta_encoder_type = config.get(section, "beta_encoder_type")
# NB_BATCHS = config.getint(section, "NB_BATCHS")
# dialogues_by_batch = config.getint(section, "dialogues_by_batch")
# batch_size_er = config.get(section, "batch_size_er")
# N_dialogues_test = config.getint(section, "dialogues_test")
# dialogues_test_intra_epoch = config.getint(section, "dialogues_test_intra_epoch")
# # size_state = config.getint(section, "size_state")
# layers = json.loads(config.get(section, "layers"))
# test_bftq = config.getboolean(section, "test_bftq")
# test_ftq = config.getboolean(section, "test_ftq")
# clip_qc_output = config.getboolean(section, "clip_qc_output")
# print_q_function = config.getboolean(section, "print_q_function")
# disp = config.getboolean(section, "disp")
# print_dial = config.getboolean(section, "print_dial")
# size_betas_layer = config.getint(section, "size_betas_layer")
# len_action_space = config.getint(section, "len_action_space")
# loss_function = config.get("RUN", "loss_function")
# loss_function_c = config.get("RUN", "loss_function_c")
# # reward_domain = tuple(json.loads(config.get(section, "reward_domain")))
# #
# lambda_ = tuple(json.loads(config.get(section, "lambda_")))
# emax = tuple(json.loads(config.get(section, "emax")))
# learning_rate = tuple(json.loads(config.get(section, "learning_rate")))
# optimizer = tuple(json.loads(config.get(section, "optimizer")))
# path_sample_data = tuple(json.loads(config.get(section, "path_sample_data")))
# weight_decay = tuple(json.loads(config.get(section, "weight_decay")))
# nb_betas = tuple(json.loads(config.get(section, "nb_betas")))
# seed = tuple(json.loads(config.get("GENERAL", "seed")))
# feature = tuple(json.loads(config.get(section, "feature")))
# new_style = tuple(json.loads(config.get(section, "new_style")))
# maxturns = config.getint("agent", "maxturns")
#
# param_grid = {'new_style': new_style,
#               'layers': layers,
#               'feature': feature,
#               'lambda_': lambda_,
#               'seed': seed,
#               'emax': emax,
#               'learning_rate': learning_rate,
#               'optimizer': optimizer,
#               'weight_decay': weight_decay,
#               'nb_betas': nb_betas,
#               'path_sample_data': path_sample_data}
#
# grid = ParameterGrid(param_grid)
#
# e = EnvPydial(config_file=config_file, error_rate=0.3, seed=None)
#
# path = "tmp/" + basename_config_file
# if os.path.exists(path):
#     shutil.rmtree(path)
# os.makedirs(path)
#
# with open("tmp/" + basename_config_file + "/grid_of_params", 'a') as infile:
#     infile.write(''.join([str(param) + "\n" for param in grid]))
#
# for il, layer in enumerate(layers):
#     layers[il] = tuple(layer)
#
# print layers
#
# param_to_save = []
#
#
# def go(id, new_style, layers, feature, lambda_, seed, emax, learning_rate, optimizer, weight_decay, nb_betas,
#        path_sample_data,
#        activation=activation, reset_type=reset_type, delta=delta, nn_loss_stop=nn_loss_stop
#        , gamma=gamma, gamma_c=gamma_c, max_nn_epoch=max_nn_epoch, clamp_Qr=clamp_Qr,
#        normalize_reward=normalize_reward, normalize=normalize,
#        reset_policy_before_regression=reset_policy_before_regression,
#        reset_weight=reset_weight, weights_losses=weights_losses, type_beta=type_beta,
#        beta_encoder_type=beta_encoder_type,
#        NB_BATCHS=NB_BATCHS, dialogues_by_batch=dialogues_by_batch, batch_size_er=batch_size_er,
#        N_dialogues_test=N_dialogues_test, test_bftq=test_bftq, test_ftq=test_ftq,
#        clip_qc_output=clip_qc_output,
#        print_q_function=print_q_function, disp=disp, size_betas_layer=size_betas_layer,
#        len_action_space=len_action_space,
#        loss_function=loss_function, loss_function_c=loss_function_c):
#     e.change_style(new_style=new_style)
#     e.reset()
#     print "available actions : ", e.action_space()
#     urpy.set_seed(seed)
#     print "lambda_", lambda_
#     params = [id, new_style, layers, feature, lambda_, seed, emax, learning_rate, optimizer, weight_decay, nb_betas,
#               path_sample_data]
#     print ["i", "new_style", "layers", "feature", "lambda_", "seed", "emax", "learning_rate", "optimizer",
#            "weight_decay", "nb_betas",
#            "path_sample_data"]
#     print params
#     param_to_save.append(params)
#     id_params = str(id)
#     path = "tmp/" + basename_config_file + "/" + id_params
#     os.makedirs(path)
#     print path
#     action_space = e.action_space()
#     activation = F.relu if activation == "RELU" else None
#     loss_function = F.mse_loss if loss_function == "MSE_LOSS" else None
#     if feature == "feature_0":
#         feature = feat.feature_0
#     elif feature == "feature_1":
#         feature = feat.feature_1
#     elif feature == "feature_2":
#         feature = feat.feature_2
#     elif feature == "feature_3":
#         feature = feat.feature_3
#     elif feature == "feature_4":
#         feature = feat.feature_4
#     else:
#         raise Exception("Unknown feature : {}".format(feature))
#
#     size_state = len(feature(e.reset(), e))
#     print "neural net input size :", size_state
#
#     policy_network = pftq.Net(
#         (size_state,) + layers + (len_action_space,),
#         activation,
#         normalize)
#
#     if optimizer == "RMS_PROP":
#         import torch
#         optimizer = torch.optim.RMSprop(policy_network.parameters(), weight_decay=weight_decay)
#
#     elif optimizer == "ADAM":
#         import torch
#         optimizer = torch.optim.Adam(policy_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     else:
#         raise Exception("unknow optimizer {}".format(optimizer))
#
#     def expdecay(start, decay):
#         return lambda x: np.exp(-x / (1. / decay)) * start
#
#     pi_greedy = None
#
#     decay = expdecay(0.3, 0.0015)
#     # decay = expdecay(0.3, 0.05)
#
#     plt.title("decay w.r.t episodes")
#     plt.plot(range(NB_BATCHS * dialogues_by_batch), [decay(d) for d in range(NB_BATCHS * dialogues_by_batch)])
#     plt.show()
#     plt.clf()
#
#     dqn = DQN(
#         policy_network,
#         gamma=gamma,
#         batch_size_experience_replay=batch_size_er,
#         target_update=dialogues_by_batch,
#         optimizer=optimizer,
#         loss_function=loss_function,
#         actions=action_space,
#         workspace=path,
#         action_str=e.action_space_str()
#     )
#
#     # results = np.zeros((dialogues_by_batch * NB_BATCHS,4)) #None
#     results = None
#
#     dqn.reset()
#     dialogue_tests = 0#100
#     dialogues_learn = NB_BATCHS * dialogues_by_batch
#     for episode in range(dialogues_learn + dialogue_tests):
#         if print_dial:
#             print('---------------------------------')
#         s = feature(e.reset(), e)
#         a = unicode('hello()')
#         rew_r, rew_c, ret_r, ret_c = 0., 0., 0., 0.
#         i = 0
#         s_, r_, end, info_env = e.step(a)
#         s_ = s_
#         i += 1
#         while not end:
#             s = s_
#             actions = e.action_space_executable()
#             # print s
#             sample = np.random.random_sample(1)[0]
#             if sample > decay(episode) and episode < dialogues_learn:
#                 a = np.random.choice(actions)
#             else:
#                 a = dqn.pi(feature(s, e), actions)
#             s_, r_, end, info_env = e.step(a, is_master_act=False)
#             s_ = s_
#             c_ = info_env["c_"]
#             if episode < dialogues_learn:
#                 dqn.update(
#                     Transition(
#                         feature(s, e),
#                         e.action_space().index(a),
#                         feature(s_, e),
#                         r_ / 20. if normalize_reward else r_
#                     )
#                 )
#             rew_r += r_
#             rew_c += c_
#             ret_r += r_ * (gamma ** i)
#             ret_c += c_ * (gamma_c ** i)
#             i += 1
#
#         # results[episode] = np.array([rew_r, rew_c, ret_r, ret_c])
#         rez = np.array([[rew_r, rew_c, ret_r, ret_c]])
#
#         if results is None:
#             results = rez
#         else:
#             results = np.vstack((results, rez))
#         # print results
#         if (episode + 1) % 10 == 0 and episode > 0:
#             aaa = results.reshape((len(results) / 10, -1, 4))
#             aaa = np.mean(aaa, axis=1)
#             plt.clf()
#             plt.plot(range(len(aaa)), aaa[:, 0])
#             path = "tmp" + "/" + basename_config_file + "/" + str(id_params)
#             if not os.path.exists(path):
#                 os.makedirs(path)
#             plt.savefig(path + "/dqn_episode={}.png".format(episode))
#             # plt.show()
#
#     return params, results
#
#
# i = 0
#
# str_params = ''
#
# for params in grid:
#     p, data_ftq, data_bftq, betas = go(i, params['new_style'], params['layers'], params['feature'], params['lambda_'],
#                                        params['seed'], params['emax'], params['learning_rate'], params['optimizer'],
#                                        params['weight_decay'], params['nb_betas'], params['path_sample_data'])
#     str_params += str(i) + ' ' + ''.join([str(param) + ' ' for param in params.values()]) + '\n'
#     if test_ftq:
#         np.save("tmp/{}/{}/data_ftq".format(basename_config_file, i), data_ftq)
#     if test_bftq:
#         for ibeta, beta in enumerate(betas):
#             np.save("tmp/{}/{}/data_bftq_beta={}".format(basename_config_file, i, beta), data_bftq[ibeta])
#         # np.save("tmp/{}/{}/data_bftq".format(basename_config_file, i), data_bftq)
#     i += 1
#     # params_str+="\n"
# with open("tmp/" + basename_config_file + "/params", 'a') as infile:
#     infile.write(str_params)
