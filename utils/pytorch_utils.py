import torch as th
from prettytable import PrettyTable


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
        assert obs.shape[1]-1  == acts.shape[1], "Observations and actions don't match."
        result = fill_tensor(result, acts, start_dim=obs.shape[-1])
    else:
        assert obs.shape[1] == 1, 'No actions have not been given to the model.'
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
