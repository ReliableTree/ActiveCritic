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


def calcMSE(a, b, return_tensor = False):
    l2_dist = (a.reshape([-1]) - b.reshape([-1]))**2
    loss = (l2_dist).mean()
    if return_tensor:
        return loss, l2_dist
    else:
        return loss


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

def pad_observations(obs: th.Tensor) -> th.Tensor:
    batch_size, seq_len, dim = obs.size()
    padded_obs = th.full((batch_size * seq_len, seq_len, dim), -2, device=obs.device, dtype=obs.dtype)
    for i in range(batch_size):
        for j in range(seq_len):
            padded_obs[i*seq_len + j, 0] = obs[i, j]
            for k in range(j):
                padded_obs[i*seq_len + j, -(k+1)] = -3
    return padded_obs

def pad_actions(acts: th.Tensor) -> th.Tensor:
    batch_size, seq_len, dim = acts.size()
    padded_acts = th.full((batch_size * seq_len, seq_len, dim), -3, device=acts.device, dtype=acts.dtype)
    for i in range(batch_size):
        for j in range(seq_len):
            padded_acts[i*seq_len + j, :seq_len-j] = acts[i, j:]
    return padded_acts

def make_autoregressive_obs_data(actions:th.Tensor, observations:th.Tensor, rewards:th.Tensor, expert_trjs:th.Tensor):
    padded_act = pad_actions(actions)
    padded_obsv = pad_observations(observations)
    padded_rews = pad_actions(rewards)
    exp_trjs = expert_trjs.unsqueeze(1).repeat([1, actions.shape[1]]).reshape([-1])
    steps = get_steps_from_actions(actions=actions)
    return padded_act, padded_obsv, padded_rews, steps, exp_trjs

def make_part_obs_data(actions:th.Tensor, observations:th.Tensor, rewards:th.Tensor, expert_trjs:th.Tensor):
    acts = actions.repeat([1, actions.shape[1], 1]).reshape([-1, actions.shape[1], actions.shape[2]])
    exp_trjs = expert_trjs.unsqueeze(1).repeat([1, actions.shape[1]]).reshape([-1])
    rews = rewards.repeat([1, rewards.shape[1], 1]).reshape([-1, actions.shape[1], 1])
    obsv = apply_triu(observations, diagonal=0).reshape([-1, observations.shape[-2], observations.shape[-1]])
    steps = get_steps_from_actions(actions=actions)
    return acts, obsv, rews, steps, exp_trjs

def get_steps_from_actions(actions):
    steps = th.arange(0, actions.shape[1], device=actions.device).reshape([1, -1]).repeat([actions.shape[0], 1]).reshape([actions.shape[0], -1])
    return steps

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

def pick_action_from_history(action_histories, steps):
    result = th.zeros([steps.shape[0] * steps.shape[1], action_histories.shape[1], action_histories.shape[-1]], device=action_histories.device, dtype=th.float)
    for batch in range(steps.shape[0]):
        for i, timestep in enumerate(steps[batch]):
            result_index = batch * len(steps[batch]) + i
            result[result_index] = action_histories[batch, timestep]
    return result

def diff_boundaries(actions, low, high):
    high_mask = actions > high
    low_mask = actions < low
    high_bound = th.square(th.where(high_mask, actions - high, th.zeros_like(actions))).sum()
    low_bound = th.square(th.where(low_mask, actions - low, th.zeros_like(actions))).sum()
    normalizing = th.sum(high_mask) + th.sum(low_mask)
    if normalizing > 0:
        com = (high_bound + low_bound) / normalizing
    else:
        com = th.zeros([1], device=actions.device, requires_grad=True)
    return com

def max_mask_after_ind(x, ind):
    x = x.reshape([x.shape[0], x.shape[1]])
    max_values, _ = x[:, ind:].max(dim=1)

    # create mask
    mask = x[:, ind:] == max_values.unsqueeze(1)

    # pad with zeros for indices before start index
    mask = th.cat([th.zeros((x.size(0), ind), dtype=th.bool, device=x.device), mask], dim=1)
    return mask.unsqueeze(-1)

def max_mask_before_ind(x, ind):
    x = x.reshape([x.shape[0], x.shape[1]])
    max_values, _ = x[:, :ind].max(dim=1)

    # create mask
    mask = x[:, :ind] == max_values.unsqueeze(1)

    # pad with zeros for indices before start index
    mask = th.cat([mask, th.zeros((x.size(0), x.size(1) - ind), dtype=th.bool, device=x.device)], dim=1)
    return mask.unsqueeze(-1)

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return th.triu(th.ones(sz, sz) * float('-inf'), diagonal=1)

def linear_interpolation(total_steps, current_step, start, end):
    if current_step <= 0:
        return start
    elif current_step >= total_steps:
        return end

    step_size = (end - start) / total_steps
    interpolated_value = start + (current_step * step_size)
    return int(interpolated_value)

def sample_gauss(mean, variance):
    gaussian = th.distributions.MultivariateNormal(mean, th.diag(variance))
    sampled_vector = gaussian.sample()
    return sampled_vector