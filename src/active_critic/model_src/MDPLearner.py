class MDPData(th.utils.data.Dataset):
    def __init__(self, device) -> None:
        super().__init__()
        self.obsv = None
        self.action = None
        self.reward = None
        self.done = None
        self.device = device

    def add_step(self, obsv:th.Tensor, action:th.Tensor, reward:th.Tensor, done:th.Tensor):
        if self.obsv is None:
            self.obsv = obsv.reshape([1, -1]).to(self.device)
        else:
            self.obsv = th.cat((self.obsv, obsv.to(self.device).reshape([1, -1])), dim=0)

        if self.action is None:
            self.action = action.to(self.device).reshape([1, -1])
        else:
            self.action = th.cat((self.action, action.to(self.device).reshape([1, -1])), dim=0)

        if self.reward is None:
            self.reward = reward.to(self.device).reshape([1, -1])
        else:
            self.reward = th.cat((self.reward, reward.to(self.device).reshape([1, -1])), dim=0)

        if self.done is None:
            self.done = done.to(self.device).reshape([1])
        else:
            self.done = th.cat((self.done, done.to(self.device).reshape([-1])), dim=0)

    def __len__(self):
        return len(self.obsv)

    def __getitem__(self, index):
        done = self.done[index]

        if done:
            return self.obsv[index], th.zeros_like(self.obsv[index]), self.action[index], th.zeros_like(self.action[index]), self.reward[index], th.zeros_like(self.reward[index]), done
        else:
            return self.obsv[index], self.obsv[index+1], self.action[index], self.action[index+1], self.reward[index], self.reward[index+1], done

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, quantisation=0, activation=nn.ReLU(), dropout=0, use_batch_norm=False):
        super(MLP, self).__init__()
        
        # create a sequential container to hold the layers
        self.layers = nn.Sequential()
        
        # create the input layer
        self.layers.add_module("input", nn.Linear(input_size, hidden_sizes[0]))
        
        # create the hidden layers
        for i, size in enumerate(hidden_sizes[1:]):
            self.layers.add_module(f"hidden_{i+1}", nn.Linear(hidden_sizes[i], size))
            if use_batch_norm:
                self.layers.add_module(f"batch_norm_{i+1}", nn.BatchNorm1d(size))
            self.layers.add_module(f"activation_{i+1}", activation)
            if dropout > 0:
                self.layers.add_module(f"dropout_{i+1}", nn.Dropout(dropout))
        
        # create the output layer
        self.layers.add_module("output", nn.Linear(hidden_sizes[-1], output_size))
        self.quantisation = quantisation
    
    def forward(self, x):
        x_shape = x.shape
        quantized = len(x_shape) == 4
        if quantized: #quantized input
            x = x.reshape([x.shape[0], x.shape[1], -1])
        # forward pass through the layers

        result = self.layers(x)
        if self.quantisation != 0:
            result = result.reshape([x_shape[0], x_shape[1], -1, self.quantisation])
        return result
        
class QuantzedMDP(gym.Wrapper):
    def __init__(self, env: gym.Env, ntokens_obsv, ntokens_act, obsv_low, obsv_high, action_low, action_high, device) -> None:
        super().__init__(env)
        self.ntokens_obsv= ntokens_obsv
        self.ntokens_act = ntokens_act

        min_obsv = self.observation_space.low
        min_obsv = np.maximum(min_obsv, obsv_low)
        self.min_obsv = th.tensor(min_obsv)
        max_obsv = self.observation_space.high
        max_obsv = np.minimum(max_obsv, obsv_high)
        self.max_obsv = th.tensor(max_obsv)

        min_action = self.action_space.low
        min_action = np.maximum(min_action, action_low)
        self.min_action = th.tensor(min_action)
        max_action = self.action_space.high
        max_action = np.minimum(max_action, action_high)
        self.max_action = th.tensor(max_action)

        self.max_recoreded_obsv = -float("inf")
        self.min_recoreded_obsv = float("inf")

        self.replay_data = MDPData(device)

        self.current_obsv = None

        

    def quantize(self, inpt, min, max, ntokens):
        th_inpt = th.tensor(inpt).reshape([1,1,-1])
        th_inpt = tokenize(inpt=th_inpt, minimum=min, maximum=max, ntokens=ntokens)
        th_inpt = detokenize(inpt=th_inpt, minimum=min, maximum=max, ntokens=ntokens)
        return th_inpt.numpy().squeeze()

    def reset(self) -> Any:
        obsv = super().reset()
        if max(obsv) > self.max_recoreded_obsv:
            self.max_recoreded_obsv = max(obsv)

        if min(obsv) < self.min_recoreded_obsv:
            self.min_recoreded_obsv = min(obsv)

        q_obsv = self.quantize(inpt=obsv, min=self.min_obsv, max=self.max_obsv, ntokens=self.ntokens_obsv)
        self.current_obsv = q_obsv
        return q_obsv

    def step(self, action):
        q_act = self.quantize(inpt=action, min=self.min_action, max=self.max_action, ntokens=self.ntokens_act)
        obsv, reward, dones, info = super().step(q_act)
        if max(obsv) > self.max_recoreded_obsv:
            self.max_recoreded_obsv = max(obsv)
            
        if min(obsv) < self.min_recoreded_obsv:
            self.min_recoreded_obsv = min(obsv)
            
        q_obsv = self.quantize(inpt=obsv, min=self.min_obsv, max=self.max_obsv, ntokens=self.ntokens_obsv)
        self.replay_data.add_step(th.tensor(self.current_obsv), th.tensor(q_act), th.tensor(reward), th.tensor(dones))
        self.current_obsv = q_obsv

        return q_obsv, reward, dones, info

    def learn(self):
        pass

class MDPLearner(nn.Module):
    def __init__(self, embbed_size, env:QuantzedMDP, embedding_decimals:int,  device:str, max_batch_size = 64) -> None:
        super().__init__()
        ntokens_obsv = env.ntokens_obsv
        ntokens_act = env.ntokens_act
        obsv_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        self.emitter = MLP(input_size=obsv_size, hidden_sizes=[2048, 2048, 2048], output_size=embbed_size, quantisation=0).to(device)
        self.predictor = MLP(input_size=(embbed_size+action_size), hidden_sizes=[2048, 2048, 2048], output_size=embbed_size, quantisation=0).to(device)
        self.reward_model = MLP(input_size=(embbed_size+action_size), hidden_sizes=[2048, 2048, 2048], output_size=1, quantisation=0).to(device)
        
        self.optimizer = th.optim.Adam(params=list(self.emitter.parameters()) + list(self.predictor.parameters())+ list(self.reward_model.parameters()), lr=1e-4)
        self.scheduler = th.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=300000/32, gamma=0.95)
        self.env = env
        self.obs_minimum = env.min_obsv.to(device)
        self.obs_maximum = env.max_obsv.to(device)
        self.action_minimum = env.min_action.to(device)
        self.action_maximum = env.max_action.to(device)
        self.ntokens_obsv = ntokens_obsv
        self.ntokens_act = ntokens_act
        self.embbed_size = embbed_size

        self.embedding_decimals = embedding_decimals

        self.obs_minimum = self.obs_minimum.reshape([1,1,-1]).repeat([max_batch_size, 1, 1]).to(device)
        self.obs_maximum = self.obs_maximum.reshape([1,1,-1]).repeat([max_batch_size, 1, 1]).to(device)
        self.action_minimum = self.action_minimum.reshape([1,1,-1]).repeat([max_batch_size, 1, 1]).to(device)
        self.action_maximum = self.action_maximum.reshape([1,1,-1]).repeat([max_batch_size, 1, 1]).to(device)

    def qemb_qact_f_obsv(self, obsvs, actions):
        batch_size = actions.shape[0]
        qobsvs = quantize(obsvs, minimum=self.obs_minimum[:batch_size], maximum=self.obs_maximum[:batch_size], nquants=self.ntokens_act)
        embeddings = self.emitter.forward(qobsvs)
        return self.get_q_emb_q_act(embeddings=embeddings, actions=actions)

    def get_q_emb_q_act(self, embeddings, actions):
        batch_size = actions.shape[0]
        qactions = quantize(actions, minimum=self.action_minimum[:batch_size], maximum=self.action_maximum[:batch_size], nquants=self.ntokens_act)
        #qembeddings = th.round(embeddings, decimals=self.embedding_decimals)
        qembeddings = embeddings

        emb_act = th.cat((qembeddings, qactions), dim=2)
        return emb_act, qembeddings


    def step(self, obsvs:th.Tensor, n_obsvs:th.Tensor, actions:th.Tensor, n_actions:th.Tensor, rewards:th.Tensor, n_rewards:th.Tensor, dones:th.Tensor):
        #Inputs are step wise, so seq_len = 1
        obsvs = obsvs.unsqueeze(1)
        n_obsvs = n_obsvs.unsqueeze(1)
        actions = actions.unsqueeze(1)
        n_actions = n_actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        n_rewards = n_rewards.unsqueeze(1)
        batch_size = obsvs.shape[0]

        #Reshape the maximum and minimum according to batch size

        nd_n_actions = n_actions[~dones]
        nd_n_observations = n_obsvs[~dones]
        nd_nrewards = n_rewards[~dones]

        rew1_loss, emb_act1, embeddings1, expected_rewards1 = self.step_reward_model(actions=actions, observations=obsvs, rewards=rewards, do_print=False)
        rew2_loss, emb_act2, q_embeddings2, expected_rewards2 = self.step_reward_model(actions=nd_n_actions, observations=nd_n_observations, rewards=nd_nrewards, do_print=False)

        nd_emb_act1 = emb_act1[~dones]

        pred_loss, pred_n_embeddings = self.step_predictor(emb_act=nd_emb_act1, n_embeddings=q_embeddings2)
        loss = rew1_loss + rew2_loss + pred_loss


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        #q_pred_embeddings2 = th.round(pred_n_embeddings, decimals=self.embedding_decimals)
        q_pred_embeddings2 = pred_n_embeddings
        return rew1_loss.detach(), rew2_loss.detach(), pred_loss.detach(), q_embeddings2.detach(), q_pred_embeddings2.detach()

    def step_reward_model(self, actions:th.Tensor, observations:th.Tensor, rewards:th.Tensor, do_print:bool):
        emb_act, qembeddings = self.qemb_qact_f_obsv(obsvs=observations, actions=actions)
        expected_rewards = self.reward_model.forward(emb_act)

        loss = calcMSE(expected_rewards, rewards)
        return loss, emb_act, qembeddings, expected_rewards

    def step_predictor(self, emb_act:th.Tensor, n_embeddings:th.Tensor):

        pred_n_embeddings = self.predictor(emb_act)
        pred_loss = calcMSE(pred_n_embeddings, n_embeddings)

        pred_loss = pred_loss.mean()
        return pred_loss, pred_n_embeddings

    def predict_step(self, embeddings, actions):
        q_emb_q_act, _ = self.get_q_emb_q_act(embeddings=embeddings, actions=actions)

        pred_n_embeddings = self.predictor(q_emb_q_act)
        #q_pred_n_embeddings = th.round(pred_n_embeddings, decimals=self.embedding_decimals)
        q_pred_n_embeddings = pred_n_embeddings
        return q_pred_n_embeddings

    def pred_n_steps(self, obsv, actions):
        rewards = []
        embeddings = []
        steps = actions.shape[1]

        obsv = obsv.unsqueeze(1)


        q_emb_q_act, qembeddings = self.qemb_qact_f_obsv(obsvs=obsv, actions=actions[:, :1])
        for i in range(steps):
            print(f'qembeddings.shape: {qembeddings.shape}')
            emb_act, qembeddings = self.get_q_emb_q_act(embeddings=qembeddings, actions=actions[:, i:i+1])
            pred_reward = self.reward_model(emb_act)
            rewards.append(pred_reward)

            q_pred_n_embeddings  = self.predict_step(embeddings=qembeddings, actions=actions[:, i:i+1])
            embeddings.append(q_pred_n_embeddings.detach())
            qembeddings = q_pred_n_embeddings.detach()

        return rewards, embeddings

    def pred_rewards(self, obsvs, actions):
        self.eval()
        q_emb_q_act, qembeddings = self.qemb_qact_f_obsv(obsvs=obsvs, actions=actions)
        rewards = self.reward_model.forward(q_emb_q_act)
        return rewards

    def learn(self, max_steps, rew1_thr, rew2_thr, embedd_thr, dataloader:th.utils.data.DataLoader, tboard:TBoardGraphs):
        self.train()
        steps = 0
        l2_rew1 = float('inf')
        l2_rew2 = float('inf')
        n_equal_embedding = float('inf')
        
        while (steps <= max_steps) and ((l2_rew1 > rew1_thr) or (l2_rew2 > rew2_thr) or (n_equal_embedding > embedd_thr)):

            r1l = 0
            r2l = 0
            el2 = 0
            nem = 0
            iteration_counter = 0
            for obsv, nobsv, action, naction, reward, nreward, done in dataloader:
                rew1_loss, rew2_loss, pred_loss, q_embeddings2, q_pred_embeddings2 = self.step(
                    obsvs=obsv, 
                    n_obsvs=nobsv, 
                    actions=action, 
                    n_actions=naction, 
                    rewards=reward,
                    n_rewards=nreward,
                    dones=done)
                steps += obsv.shape[0]
                iteration_counter+= 1
                r1l = r1l + rew1_loss
                r2l = r2l + rew2_loss
                el2 = el2 + pred_loss

                nee = (q_embeddings2 != q_pred_embeddings2).sum()
                nem = nem + nee
                

            l2_rew1 = r1l / iteration_counter
            l2_rew2 = r2l / iteration_counter
            el2 = el2 / iteration_counter
            n_equal_embedding = nem
            tboard.addTrainScalar('l2_rew1', value=l2_rew1.to('cpu'), stepid=steps)
            tboard.addTrainScalar('l2_rew2', value=l2_rew2.to('cpu'), stepid=steps)
            tboard.addTrainScalar('pred_loss', value=el2.to('cpu'), stepid=steps)
            tboard.addTrainScalar('lr', value=th.tensor(self.optimizer.state_dict()['param_groups'][0]['lr']), stepid=steps)
            #tboard.addTrainScalar('n_equal_embedding', n_equal_embedding.to('cpu'), stepid=steps)
            
