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


def make_partially_observed_seq(obs: th.Tensor, acts: th.Tensor, seq_len: int, act_space, tokenize:bool=False, ntokens:int=1):
    if tokenize:
        result = -2*th.ones(size=[obs.shape[0], seq_len,
                        obs.shape[-1] + act_space.shape[0]*ntokens], device=obs.device)
    else:
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


def calcMSE(a:th.Tensor, b:th.Tensor) -> th.Tensor:
    return ((a.squeeze() - b.squeeze())**2).mean()


def apply_triu(inpt:th.Tensor, diagonal:th.Tensor, empty_token:float):
    exp_inpt = inpt.unsqueeze(1)
    shape = exp_inpt.shape
    # shape = batch, 1, seq, dims...
    exp_inpt = exp_inpt.repeat([1, shape[2], *[1]*(len(shape)-2)])
    mask = th.triu(th.ones(
        [shape[2], shape[2]], device=inpt.device), diagonal=diagonal).T
    # batch, seq, seq, dims...
    mask = mask.reshape([1, mask.shape[0], mask.shape[1], 1]).repeat([exp_inpt.shape[0], 1, 1, exp_inpt.shape[-1]])
    exp_out = exp_inpt * mask
    exp_out[mask==0] = empty_token
    return exp_out


def make_part_obs_data(actions:th.Tensor, observations:th.Tensor, rewards:th.Tensor, empty_token:float):
    acts = actions.repeat([1, actions.shape[1], 1]).reshape([-1, actions.shape[1], actions.shape[2]])
    rews = rewards.repeat([1, rewards.shape[1], 1]).reshape([-1, actions.shape[1], 1])
    obsv = apply_triu(observations, diagonal=0, empty_token=empty_token).reshape([-1, observations.shape[-2], observations.shape[-1]])
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

def make_indices(num, repeats):
    indices = th.arange(num).reshape([1,-1]).repeat([repeats,1]).T.reshape([-1])
    return indices

def tokenize(inpt, minimum, maximum, ntokens):
    batch_size = inpt.shape[0]
    seq_len = inpt.shape[1]
    dim = inpt.shape[2]
    scale = maximum - minimum
    rec_inpt = ((inpt - minimum) / scale)*(ntokens-1)
    rounded = th.round(rec_inpt).type(th.long).reshape([-1])
    batch_indices = make_indices(batch_size, seq_len*dim).to(inpt.device)
    seq_indices = make_indices(seq_len, dim).repeat(batch_size).to(inpt.device)
    dim_indices = make_indices(dim, 1).repeat(batch_size*seq_len).to(inpt.device)
    result = th.zeros([inpt.shape[0], inpt.shape[1], inpt.shape[2], ntokens], dtype=float, device=inpt.device)
    result[tuple((batch_indices, seq_indices, dim_indices, rounded))] = 1
    result = result.reshape([inpt.shape[0], inpt.shape[1], -1])
    return result
    
def detokenize(inpt:th.Tensor, minimum, maximum, ntokens):
    exp_inpt = inpt.reshape([inpt.shape[0], inpt.shape[0], -1, ntokens])
    values = exp_inpt.argmax(dim=-1)
    scale = maximum - minimum
    values = values/(ntokens-1) * scale
    values = values.reshape([inpt.shape[0], inpt.shape[1], -1])
    values = values + minimum
    return values