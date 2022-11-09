import torch as th
from prettytable import PrettyTable
from abc import ABC, abstractclassmethod


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def make_counter_embedding(x, bits):
    mask = 2**th.arange(bits-1, -1, -1)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().numpy()


def get_integer(x):
    bits = len(x)
    mask = 2**th.arange(bits-1, -1, -1)
    integer = (x*mask).sum()
    return int(integer)


def get_num_bits(interger):
    return int(th.ceil(th.log2(th.tensor(interger))))


def make_partially_observed_seq(obs: th.Tensor, acts: th.Tensor, seq_len: int, act_space):
    result = th.zeros(size=[obs.shape[0], seq_len,
                      obs.shape[-1] + act_space.shape[0]], device=obs.device)
    # obs: batch, seq, dim
    result = fill_tensor(result, obs, start_dim=0)
    if acts is not None:
        result = fill_tensor(result, acts, start_dim=obs.shape[-1])
    return result


def fill_tensor(tensorA: th.Tensor, tensorB: th.Tensor, start_dim: int):
    shape = tensorB.shape
    tensorA[:shape[0], :shape[1], start_dim:start_dim+shape[2]] = tensorB
    return tensorA


def add_max_val_to_dict(dict, key, val, tm):
    if key in dict:
        dict[key] = th.max(dict[key], val)
        if val > dict[key]:
            dict[key + ' max'] = th.tensor(tm)
    else:
        dict[key] = val


def calcMSE(a, b):
    return ((a.squeeze() - b.squeeze())**2).mean()


def apply_triu(inpt:th.Tensor, diagonal:th.Tensor):
    exp_inpt = inpt.unsqueeze(1)
    shape = exp_inpt.shape
    # shape = batch, 1, seq, dims...
    exp_inpt = exp_inpt.repeat([1, shape[2], *[1]*(len(shape)-2)])
    mask = th.triu(th.ones(
        [shape[2], shape[2]], device=inpt.device), diagonal=diagonal).T
    # batch, seq, seq, dims...
    exp_out = exp_inpt * mask[None, :, :, None]
    return exp_out


def make_part_obs_data(actions:th.Tensor, observations:th.Tensor, rewards:th.Tensor):
    acts = actions.repeat([1, actions.shape[1], 1]).reshape([-1, actions.shape[1], actions.shape[2]])
    rews = rewards.repeat([1, rewards.shape[1], 1]).reshape([-1, actions.shape[1], 1])
    obsv = apply_triu(observations, diagonal=0).reshape([-1, observations.shape[-2], observations.shape[-1]])
    return acts, obsv, rews

def make_inf_seq(obs:th.Tensor, seq_len:th.Tensor):
    start_seq = th.zeros([obs.shape[0], int(seq_len/2), obs.shape[-1]], device= obs.device)
    whole_seq = th.cat((start_seq, obs), dim=1)
    result = whole_seq[:,0:seq_len]
    for i in range(len(obs[0]) - seq_len + int(seq_len/2)):
        result = th.cat((result, whole_seq[:,i+1:seq_len+i+1]), dim=0)
    return result

def get_seq_end_mask(inpt, current_step):
    mask = th.zeros_like(inpt, dtype=th.bool)
    mask[:,current_step:] = True
    return mask

def get_rew_mask(reward):
    return (reward.squeeze()>=0)


def generate_square_subsequent_mask(sz: int) -> th.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return th.triu(th.ones(sz, sz) * float('-inf'), diagonal=1)


def build_tf_horizon_mask(seq_len:int, horizon:int, device:str):
    mask = generate_square_subsequent_mask(seq_len)
    if horizon > 0:
        mask[horizon:, :-horizon] += generate_square_subsequent_mask(seq_len-horizon).T
    else:
        mask += generate_square_subsequent_mask(seq_len).T
    mask = mask.to(device)
    return mask

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()