import numpy as np

def transition_to_rbm_transiton(transition,n_actions):
    one_hot_action = np.zeros(n_actions)
    one_hot_action[transition.a]=1.
    rez = np.concatenate((transition.s,one_hot_action,transition.s_))
    return rez

def transtions_to_rbm_transtions(transitions,n_actions):
    t0 = transitions[0]
    rbm_transtions = np.zeros((len(transitions),len(t0.s)+len(t0.s_)+n_actions))
    for it,t in enumerate(transitions):
        rbm_transtions[it] = transition_to_rbm_transiton(t,n_actions)
    return rbm_transtions