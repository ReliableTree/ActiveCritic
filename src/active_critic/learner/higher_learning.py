import torch as th
import higher
from active_critic.model_src.whole_sequence_model import CriticSequenceModel, WholeSequenceModelSetup, WholeSequenceModel
from active_critic.model_src.transformer import (
    ModelSetup)
from active_critic.utils.pytorch_utils import calcMSE
import copy

class module_parameter(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()


def high_step(critic:CriticSequenceModel, plan_enc, inf_opt_steps, inf_opt_lr, obsvs, acts, rewards, goal_rewards, get_critic_inpt, w_forward, w_meta):
    init_plan_enc = copy.deepcopy(plan_enc)
    meta_plan_enc = copy.deepcopy(plan_enc)
    inpt_optim = th.optim.SGD(meta_plan_enc.parameters(), lr=inf_opt_lr)
    if critic.model is None:
        with th.no_grad():
            test_inpt = get_critic_inpt(obsvs, acts, plan_enc.param)
            critic.forward(test_inpt)

    critic_losses = []
    meta_losses = []
    for step in range(inf_opt_steps):
        with higher.innerloop_ctx(critic, critic.optimizer) as (higher_critic, higher_optimizer):
            with higher.innerloop_ctx(meta_plan_enc, inpt_optim) as (higher_meta_plan_enc, higher_inpt_optim):
                meta_inpt = get_critic_inpt(obsvs, acts, higher_meta_plan_enc.param)
                init_meta_inpt = get_critic_inpt(obsvs, acts, init_plan_enc.param)
                meta_res_1 = higher_critic.forward(meta_inpt)
                forward_result = higher_critic.forward(init_meta_inpt)
                forward_loss = calcMSE(forward_result, rewards)
                meta_loss_1 = calcMSE(meta_res_1, goal_rewards)
                higher_inpt_optim.step(meta_loss_1)

                meta_inpt = get_critic_inpt(obsvs, acts, higher_meta_plan_enc.param)
                meta_res_2 = higher_critic.forward(meta_inpt)
                meta_loss_2 = calcMSE(goal_rewards, meta_res_2)

                meta_plan_enc.load_state_dict(higher_meta_plan_enc.state_dict())
                grad_of_grads = th.autograd.grad(
                    w_meta * meta_loss_2 + w_forward * forward_loss, higher_critic.parameters(time=0))

        critic_param_list = list(critic.parameters())
        for index in range(len(grad_of_grads)):
            critic_param_list[index].grad = grad_of_grads[index]
        critic.optimizer.step()

        critic_losses.append(forward_loss.detach())
        meta_losses.append(meta_loss_2.detach())

    return critic_losses, meta_losses

def critic_step(
        critic:CriticSequenceModel, 
        planner:WholeSequenceModel, 
        inf_opt_steps, 
        inf_opt_lr, 
        obsvs, 
        acts, 
        rewards, 
        goal_rewards,
        get_critic_inpt, 
        get_planner_inpt,
        expert_trjs,
        w_forward,
        w_meta
        ):
    planner_inpt = get_planner_inpt(acts=acts, obsvs=obsvs)
    with th.no_grad():
        plans = planner.forward(planner_inpt)
        plans[expert_trjs] = 0
    plans_param = th.nn.parameter.Parameter(plans.detach())
    plans_module = module_parameter()
    plans_module.register_parameter('param', param=plans_param)
    critic_losses, meta_losses = high_step(
        critic=critic, 
        plan_enc=plans_module, 
        inf_opt_steps=inf_opt_steps, 
        inf_opt_lr=inf_opt_lr, 
        obsvs=obsvs, 
        acts=acts, 
        rewards=rewards, 
        goal_rewards=goal_rewards, 
        get_critic_inpt=get_critic_inpt,
        w_forward=w_forward,
        w_meta=w_meta)
    debug_dict = {}
    for step in range(len(critic_losses)):
        debug_dict[f'']

def actor_step(actor:WholeSequenceModel, planner:WholeSequenceModel, obsvs:th.Tensor, acts:th.Tensor, expert_trjs:th.Tensor, get_planner_inpt, get_actor_inpt):

    planner_inpt = get_planner_inpt(acts=acts, obsvs=obsvs)
    plans = planner.forward(planner_inpt)
    plans[expert_trjs] = 0

    actor_input = get_actor_inpt(plans=plans, obsvs=obsvs)
    actor_result = actor.forward(actor_input)
    loss = ((actor_result.reshape(-1) - acts.reshape(-1))**2).mean()
    actor.optimizer.zero_grad()
    planner.optimizer.zero_grad()
    loss.backward()
    actor.optimizer.step()
    planner.optimizer.step()
    return loss.detach()