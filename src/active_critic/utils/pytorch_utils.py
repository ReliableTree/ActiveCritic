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
    '''mask[mask==0] = -2
    mask[mask==1] = 0
    exp_out = exp_out + mask[None, :, :, None]'''
    return exp_out


def make_part_obs_data(actions:th.Tensor, observations:th.Tensor, rewards:th.Tensor):
    acts = actions.repeat([1, actions.shape[1], 1]).reshape([-1, actions.shape[1], actions.shape[2]])
    rews = rewards.repeat([1, rewards.shape[1], 1]).reshape([-1, actions.shape[1], 1])
    
    obsv = observations[:, :1, :].repeat([1, observations.shape[1], 1]).reshape([-1, observations.shape[1], observations.shape[2]])
    #obsv = apply_triu(observations, diagonal=0).reshape([-1, observations.shape[-2], observations.shape[-1]])
    return actions, obsv, rewards

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

def pull_tens_to_front(src, i):
    if i > 0:
        src[i, :, :-i] = src[i,:,i:]
        src[i, :, -i:] = -1

    return src

def pull_tens_to_front_sparse(src, i):
    pulled = src[i,:,i:]
    max_pulled = pulled.max(dim=-2).values == 1
    max_pulled = max_pulled.unsqueeze(-2).repeat([1, 1, src.shape[2] - i, 1])
    if i > 0:
        src[i, :, :-i] = max_pulled
        src[i, :, -i:] = -1
    else:
        src[i, :] = max_pulled

    return src

def repeat_along_seq_td(src):
    src = src.repeat([src.shape[1], 1, 1]).reshape([src.shape[1], src.shape[0], src.shape[1], src.shape[2]])
    return src

def repeat_along_seq(src, seq_len):
    src = src.repeat([1, seq_len, 1])
    return src


def make_dense_seq_encoding_data(actions, obsv, rewards):
    actions = repeat_along_seq_td(actions)
    obsv = repeat_along_seq_td(obsv)
    rewards = repeat_along_seq_td(rewards)
    for i in range(len(obsv)):
        obsv[i] = obsv[i,:,i].unsqueeze(1)

        actions = pull_tens_to_front(actions, i)
        rewards = pull_tens_to_front(rewards, i)
    return actions.reshape([-1, actions.shape[-2], actions.shape[-1]]), obsv.reshape([-1, obsv.shape[-2], obsv.shape[-1]]), rewards.reshape([-1, rewards.shape[-2], rewards.shape[-1]])

def generate_partial_observed_mask(reward, nheads):
    device = reward.device
    inv_result_mask = reward.reshape([reward.shape[0], -1]) == -1
    result_mask = ~inv_result_mask
    args = th.argwhere(inv_result_mask)
    restructured_args = args.repeat([1, reward.shape[1]]).reshape([-1, 2])
    exp_ind = th.arange(reward.shape[1], device=device).repeat(args.shape[0])
    full_ind = th.cat((restructured_args[:, :1], exp_ind.unsqueeze(1), restructured_args[:, 1:]), dim=-1)
    attention_mask = th.zeros([reward.shape[0], reward.shape[1], reward.shape[1]], device=device)
    attention_mask[tuple(full_ind.T)] = -float('inf')
    attention_mask = attention_mask.repeat([1,1,nheads]).reshape([nheads*attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2]])
    result_mask = th.clone(result_mask.detach())
    attention_mask = th.clone(attention_mask.detach())
    return attention_mask, result_mask