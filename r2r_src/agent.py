
import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer
import utils
import model
from param import args
from collections import defaultdict
import clip

top_prompt = "Break the following instruction into multiple detailed sub-instructions and list them numerically, each sub-instruction must be written in English and end with a period, do not output other information: "

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        if args.seed is not None:
            random.seed(args.seed)
        self.results = {}
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Models
        enc_hidden_size = args.rnn_dim//2 if args.bidir else args.rnn_dim
        self.encoder = model.EncoderLSTM(tok.vocab_size(), args.wemb, enc_hidden_size, padding_idx,
                                         args.dropout, bidirectional=args.bidir).cuda()
        self.decoder = model.AttnDecoderLSTM(args.aemb, args.rnn_dim, args.dropout, feature_size=self.feature_size + args.angle_feat_size).cuda()
        self.critic = model.Critic().cuda()

        # self.top_tokenizer = AutoTokenizer.from_pretrained("/home/wsco/DILLM/r2r_src/THUDM/chatglm3-6b", trust_remote_code=True)
        # # FP16（no quantization）with two GPU
        # # self.decoder_top = load_model_on_gpus("/home/wsco/DILLM/r2r_src/THUDM/chatglm3-6b", num_gpus=2)
        # # INT4 with single GPU
        # self.decoder_top =  AutoModel.from_pretrained("/home/wsco/DILLM/r2r_src/THUDM/chatglm3-6b", trust_remote_code=True).quantize(4).half().cuda()
        # self.decoder_top = self.decoder_top.eval()

        self.discriminator = model.FFNet().cuda()
        self.clip_model, _ = clip.load("RN50x4", device="cuda", download_root = "/home/wsco/DILLM2/img_features/")
        # CLIP is used only for inference; disable gradients and set eval mode permanently
        self.clip_model.requires_grad_(False)
        self.clip_model.eval()
        self.models = (self.encoder, self.decoder, self.critic, self.discriminator)
        
        # Optimizers
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        critic_lr = args.critic_lr if args.critic_lr is not None else args.lr
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=critic_lr)
        self.discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=args.lr)
        self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer, self.discriminator_optimizer)
        # Evaluations
        self.losses = []
        # self.mse = nn.MSELoss(reduce=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

        # Parameter count (per module + total)
        model_names = ('encoder', 'decoder', 'critic', 'discriminator')
        total_params = 0
        print('=' * 50)
        print('Model parameter count:')
        for name, mod in zip(model_names, self.models):
            n = sum(p.nelement() for p in mod.parameters())
            total_params += n
            print('  %-20s %10d  (%.2fM)' % (name, n, n / 1e6))
        print('  %-20s %10d  (%.2fM)' % ('TOTAL', total_params, total_params / 1e6))
        print('=' * 50)
        self.total_params = total_params

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_instr = np.array([ob['instructions'] for ob in obs])
        seq_subgoals = [ob['subgoals'] for ob in obs]
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        # instr,subgoal need to be reordered
        sorted_instr = seq_instr[perm_idx]
        sorted_subgoals = [seq_subgoals[i] for i in perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]    # seq_lengths[0] is the Maximum length

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(),  \
               list(seq_lengths), list(perm_idx), sorted_instr, sorted_subgoals

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        obj_features = np.empty((len(obs), args.views, args.obj_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
            obj_features[i, :, :] = ob['obj']   # Obj feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda(), Variable(torch.from_numpy(obj_features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        candidate_obj = np.zeros((len(obs), max(candidate_leng), args.obj_dim), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                # batch * cand_num * 2048+128
                candidate_feat[i, j, :] = c['feature']                         # Image feat
                candidate_obj[i, j, :] = ob['obj'][c['pointId']]               # Obj feat from same viewpoint
        return (torch.from_numpy(candidate_feat).cuda(),
                torch.from_numpy(candidate_obj).cuda(),
                candidate_leng)

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t, obj_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_obj, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_obj, candidate_leng, obj_t

    def _teacher_action(self, obs, ended):

        a = np.zeros(len(obs), dtype=np.int64)
        rel_heading = np.zeros(len(obs), dtype=np.float32)
        rel_elevation = np.zeros(len(obs), dtype=np.float32)
        for i, ob in enumerate(obs):
            if ended[i]:
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:
                        a[i] = k
                        rel_heading[i] = candidate['heading']
                        rel_elevation[i] = candidate['elevation']
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda(), torch.from_numpy(rel_heading).cuda(), torch.from_numpy(rel_elevation).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view 
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        if speaker is not None:         # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.feature_size).cuda())
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))

        # Reorder the language input for the encoder (do not ruin the original code)
        seq, seq_mask, seq_lengths, perm_idx, seq_instr, seq_subgoals = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx_mask = seq_mask

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # For test result submission
        visited = [set() for _ in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        # Global synchronous subgoal switching with gamma aligned to t % option_step

        option_step = args.option_step

        # using pre-processed sub-instructions
        all_subgoal_list = seq_subgoals

        # number of sub-instruction decomposed by instruction in each batch (generally 2-5)
        all_subgoal_length = [len(item) for item in all_subgoal_list]
        all_subgoal_length_arr = np.array(all_subgoal_length, dtype=np.int32)
        max_subgoal_length = int(all_subgoal_length_arr.max())
        current_subgoal_index = [-1]*batch_size
        finish_flag = np.array([0]*batch_size)

        # Pre-compute CLIP text features for all subgoals (no grad needed, CLIP is frozen)
        flat_subgoals = []
        flat_indices = []
        for i in range(batch_size):
            for j in range(all_subgoal_length_arr[i]):
                flat_subgoals.append(all_subgoal_list[i][j])
                flat_indices.append((i, j))

        flat_clip_tokens = clip.tokenize(flat_subgoals).to("cuda")
        with torch.no_grad():
            flat_clip_features = self.clip_model.encode_text(flat_clip_tokens).float()

        precomp_clip = [[None] * int(all_subgoal_length_arr[i]) for i in range(batch_size)]
        for k, (i, j) in enumerate(flat_indices):
            precomp_clip[i][j] = flat_clip_features[k]             # (feat_dim,)

        cached_text_features = None      # current batch CLIP text features
        rollout_start_time = time.time()
        num_steps = 0
        for t in range(self.episode_len):
            step_start_time = time.time()
            input_a_t, f_t, candidate_feat, candidate_obj, candidate_leng, obj_t = self.get_input_feat(perm_obs)
            if speaker is not None:
                candidate_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise

            # --- Global synchronous subgoal switching (original DILLM design) ---
            # All samples switch together at fixed intervals OR when all finish flags are set.
            # This keeps switching aligned with the global periodic A2C gamma schedule.
            if (t % option_step == 0 or finish_flag.all()) and max(current_subgoal_index) < max_subgoal_length - 1:
                # reset flag
                finish_flag = np.array([0]*batch_size)

                current_subgoal_index = [index+1 for index in current_subgoal_index]
                for i in range(batch_size):
                    if current_subgoal_index[i] > all_subgoal_length[i]-1:
                        current_subgoal_index[i] = all_subgoal_length[i]-1

                # Encode current subgoals with encoder (with gradients, matching DILLM_light)
                subgoal_encoding_list = []
                current_subgoal_list = []
                for i in range(batch_size):
                    subgoal = all_subgoal_list[i][current_subgoal_index[i]]
                    current_subgoal_list.append(subgoal)
                    subgoal_encoding_list.append(self.env.tok.encode_sentence(subgoal))
                subgoal_encoding = np.array(subgoal_encoding_list)

                subgoal_lengths = np.argmax(subgoal_encoding == padding_idx, axis=1)
                subgoal_lengths[subgoal_lengths == 0] = subgoal_encoding.shape[1]
                subgoal_lengths = torch.from_numpy(subgoal_lengths)

                subgoal_mask = (subgoal_encoding == padding_idx)[:,:max(subgoal_lengths).item()]
                subgoal_mask = torch.from_numpy(subgoal_mask).byte().cuda()

                subgoal_encoding = torch.from_numpy(subgoal_encoding)
                subgoal_encoding = Variable(subgoal_encoding, requires_grad=False).long().cuda()
                subgoal_ctx, _, _ = self.encoder(subgoal_encoding, subgoal_lengths, enforce_sorted=False)

                # Stack pre-computed CLIP features
                cached_text_features = torch.stack([precomp_clip[i][current_subgoal_index[i]] for i in range(batch_size)])
            h_t, c_t, logit, attn_feat, weighted_context = self.decoder(f_t, candidate_feat,
                                               h_t, c_t, subgoal_ctx, subgoal_mask,
                                               ctx, ctx_mask,
                                               already_dropfeat=(speaker is not None),
                                               obj_feat=obj_t if args.fusion_proj else None,
                                               cand_obj_feat=candidate_obj if args.fusion_proj else None)

            hidden_states.append(h_t.clone())
            candidate_mask = utils.length2mask(candidate_leng)
            if args.submit:     # Avoding cyclic path
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask.bool(), -float('inf'))

            # Supervised training
            # target, _, _ = self._teacher_action(perm_obs, ended)
            # ml_loss += self.criterion(logit, target)
            
            if self.feedback == 'teacher':
                # a_t = target
                pass
            elif self.feedback == 'argmax': 
                _, a_t = logit.max(1)
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())
                entropys.append(c.entropy())
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            cpu_a_t = a_t.cpu().numpy()
            candidate_leng_arr = np.array(candidate_leng)
            is_end = (cpu_a_t == (candidate_leng_arr - 1)) | (cpu_a_t == args.ignoreid) | ended
            cpu_a_t[is_end] = -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            # Get features for discriminator (no caching, matching DILLM_light)
            input_a_t, f_t, _, _, _, obj_t = self.get_input_feat(perm_obs)

            # Use cached CLIP text features (recomputed only on subgoal switch, not every step)
            text_features = cached_text_features
            image_features = f_t[:,:,0:args.feature_size]
            object_features = obj_t

            # normalized features
            image_features = image_features / image_features.norm(dim=2, keepdim=True) # bs x 36 x image_f_dim
            object_features = object_features / object_features.norm(dim=2, keepdim=True) # bs x 36 x obj_f_dim
            text_features_norm = text_features / text_features.norm(dim=1, keepdim=True) # bs x text_f_dim
            text_features_norm_unsqueeze = text_features_norm.unsqueeze(2)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * torch.bmm(image_features, text_features_norm_unsqueeze)
            logits_per_image = logits_per_image.squeeze(2)
            image_text_match_probs = logits_per_image.softmax(dim=-1)

            logits_per_obj = logit_scale * torch.bmm(object_features, text_features_norm_unsqueeze)
            logits_per_obj = logits_per_obj.squeeze(2)
            obj_text_match_probs = logits_per_obj.softmax(dim=-1)

            finish_or_not = self.discriminator(obj_text_match_probs, image_text_match_probs, input_a_t, text_features_norm)

            # Compute distances first (needed for reward)
            dist = np.array([ob['distance'] for ob in perm_obs], dtype=np.float32)

            # Use discriminator output for global synchronous subgoal switching
            finish_flag = finish_or_not.squeeze(-1).detach().cpu().numpy() > 0.5

            # Calculate the mask and reward (dist already computed above, vectorized)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            mask[ended] = 0.

            active = ~ended
            is_stop = (cpu_a_t == -1) & active
            is_move = (cpu_a_t != -1) & active

            # Stop action rewards
            reward[is_stop & (dist < 3)] = 2.0
            reward[is_stop & (dist >= 3)] = -2.0

            # Move action rewards: sign of distance change
            if is_move.any():
                delta = -(dist[is_move] - last_dist[is_move])
                delta_sign = np.sign(delta)
                if (delta_sign == 0).any():
                    raise NameError("The action doesn't change the move")
                reward[is_move] = delta_sign
            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            num_steps += 1
            self.logs['step_time'].append(time.time() - step_start_time)

            # Early exit if all ended
            if ended.all():
                break

        self.logs['rollout_time'].append(time.time() - rollout_start_time)
        self.logs['rollout_steps'].append(num_steps)

        if train_rl:
            # Last action in A2C
            input_a_t, f_t, candidate_feat, candidate_obj, candidate_leng, obj_t = self.get_input_feat(perm_obs)
            if speaker is not None:
                candidate_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise
            last_h_, _, _, _, _ = self.decoder(f_t, candidate_feat,
                                            h_t, c_t, subgoal_ctx, subgoal_mask,
                                            ctx, ctx_mask,
                                            already_dropfeat=(speaker is not None),
                                            obj_feat=obj_t if args.fusion_proj else None,
                                            cand_obj_feat=candidate_obj if args.fusion_proj else None)
            rl_loss = 0

            # A2C
            last_value__ = self.critic(last_h_).detach()
            discount_reward = np.zeros(batch_size, np.float32)
            discount_reward[~ended] = last_value__[~ended].cpu().numpy()

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                # Global periodic gamma: 0 at subgoal boundary, 1 otherwise (original DILLM design)
                if t % option_step == option_step - 1:
                    worker_gamma = 0
                else:
                    worker_gamma = 1
                discount_reward = discount_reward * worker_gamma + rewards[t]
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                r_ = Variable(torch.from_numpy(discount_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = r_ - v_.detach()

                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5
                if self.feedback == 'sample':
                    rl_loss += (- args.entropy_coef * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)
            self.logs['reward'].append(discount_reward.sum())

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        return traj

    def _dijkstra(self):
        """
        The dijkstra algorithm.
        Was called beam search to be consistent with existing work.
        But it actually finds the Exact K paths with smallest listener log_prob.
        :return:
        [{
            "scan": XXX
            "instr_id":XXX,
            'instr_encoding": XXX
            'dijk_path': [v1, v2, ..., vn]      (The path used for find all the candidates)
            "paths": {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        }]
        """
        def make_state_id(viewpoint, action):     # Make state id
            return "%s_%s" % (viewpoint, str(action))
        def decompose_state_id(state_id):     # Make state id
            viewpoint, action = state_id.split("_")
            action = int(action)
            return viewpoint, action

        # Get first obs
        obs = self.env._get_obs()

        # Prepare the state id
        batch_size = len(obs)
        results = [{"scan": ob['scan'],
                    "instr_id": ob['instr_id'],
                    "instr_encoding": ob["instr_encoding"],
                    "dijk_path": [ob['viewpoint']],
                    "paths": []} for ob in obs]

        # Encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        recover_idx = np.zeros_like(perm_idx)
        for i, idx in enumerate(perm_idx):
            recover_idx[idx] = i

        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx, h_t, c_t, ctx_mask = ctx[recover_idx], h_t[recover_idx], c_t[recover_idx], seq_mask[recover_idx]    # Recover the original order

        # Dijk Graph States:
        id2state = [
            {make_state_id(ob['viewpoint'], -95):
                 {"next_viewpoint": ob['viewpoint'],
                  "running_state": (h_t[i], h_t[i], c_t[i]),
                  "location": (ob['viewpoint'], ob['heading'], ob['elevation']),
                  "feature": None,
                  "from_state_id": None,
                  "score": 0,
                  "scores": [],
                  "actions": [],
                  }
             }
            for i, ob in enumerate(obs)
        ]    # -95 is the start point
        visited = [set() for _ in range(batch_size)]
        finished = [set() for _ in range(batch_size)]
        graphs = [utils.FloydGraph() for _ in range(batch_size)]        # For the navigation path
        ended = np.array([False] * batch_size)

        h_t_top = h_t
        c_t_top = c_t
        h1_top = h_t
        z_t = torch.zeros([64,8]).cuda()
        option_step = args.option_step

        # Dijk Algorithm
        for step in range(300):
            # Get the state with smallest score for each batch
            # If the batch is not ended, find the smallest item.
            # Else use a random item from the dict  (It always exists)
            smallest_idXstate = [
                max(((state_id, state) for state_id, state in id2state[i].items() if state_id not in visited[i]),
                    key=lambda item: item[1]['score'])
                if not ended[i]
                else
                next(iter(id2state[i].items()))
                for i in range(batch_size)
            ]

            # Set the visited and the end seqs
            for i, (state_id, state) in enumerate(smallest_idXstate):
                assert (ended[i]) or (state_id not in visited[i])
                if not ended[i]:
                    viewpoint, action = decompose_state_id(state_id)
                    visited[i].add(state_id)
                    if action == -1:
                        finished[i].add(state_id)
                        if len(finished[i]) >= args.candidates:     # Get enough candidates
                            ended[i] = True

            # Gather the running state in the batch
            h_ts, h1s, c_ts = zip(*(idXstate[1]['running_state'] for idXstate in smallest_idXstate))
            h_t, h1, c_t = torch.stack(h_ts), torch.stack(h1s), torch.stack(c_ts)

            # Recover the env and gather the feature
            for i, (state_id, state) in enumerate(smallest_idXstate):
                next_viewpoint = state['next_viewpoint']
                scan = results[i]['scan']
                from_viewpoint, heading, elevation = state['location']
                self.env.env.sims[i].newEpisode(scan, next_viewpoint, heading, elevation) # Heading, elevation is not used in panoramic
            obs = self.env._get_obs()

            # Update the floyd graph
            # Only used to shorten the navigation length
            # Will not effect the result
            for i, ob in enumerate(obs):
                viewpoint = ob['viewpoint']
                if not graphs[i].visited(viewpoint):    # Update the Graph
                    for c in ob['candidate']:
                        next_viewpoint = c['viewpointId']
                        dis = self.env.distances[ob['scan']][viewpoint][next_viewpoint]
                        graphs[i].add_edge(viewpoint, next_viewpoint, dis)
                    graphs[i].update(viewpoint)
                results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], viewpoint))

            input_a_t, f_t, candidate_feat, candidate_obj, candidate_leng, obj_t = self.get_input_feat(obs)

            # Run one decoding step
            if step % option_step == 0:
                h_t_top, c_t_top, h1_top, seg_pos = self.decoder_top(f_t.clone(),
                                               h1_top, c_t_top,
                                               ctx, ctx_mask,
                                               False) 
            
            h_t, c_t, logit, _ = self.decoder(f_t, candidate_feat,
                                               h_t, c_t, z_t,
                                               ctx, ctx_mask,
                                               False)

            # Update the dijk graph's states with the newly visited viewpoint
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask.bool(), -float('inf'))
            log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            _, max_act = log_probs.max(1)

            for i, ob in enumerate(obs):
                current_viewpoint = ob['viewpoint']
                candidate = ob['candidate']
                current_state_id, current_state = smallest_idXstate[i]
                old_viewpoint, from_action = decompose_state_id(current_state_id)
                assert ob['viewpoint'] == current_state['next_viewpoint']
                if from_action == -1 or ended[i]:       # If the action is <end> or the batch is ended, skip it
                    continue
                for j in range(len(ob['candidate']) + 1):               # +1 to include the <end> action
                    # score + log_prob[action]
                    modified_log_prob = log_probs[i][j].detach().cpu().item() 
                    new_score = current_state['score'] + modified_log_prob
                    if j < len(candidate):                        # A normal action
                        next_id = make_state_id(current_viewpoint, j)
                        next_viewpoint = candidate[j]['viewpointId']
                        trg_point = candidate[j]['pointId']
                        heading = (trg_point % 12) * math.pi / 6
                        elevation = (trg_point // 12 - 1) * math.pi / 6
                        location = (next_viewpoint, heading, elevation)
                    else:                                                 # The end action
                        next_id = make_state_id(current_viewpoint, -1)    # action is -1
                        next_viewpoint = current_viewpoint                # next viewpoint is still here
                        location = (current_viewpoint, ob['heading'], ob['elevation'])

                    if next_id not in id2state[i] or new_score > id2state[i][next_id]['score']:
                        id2state[i][next_id] = {
                            "next_viewpoint": next_viewpoint,
                            "location": location,
                            "running_state": (h_t[i], h1[i], c_t[i]),
                            "from_state_id": current_state_id,
                            "feature": (f_t[i].detach().cpu(), candidate_feat[i][j].detach().cpu()),
                            "score": new_score,
                            "scores": current_state['scores'] + [modified_log_prob],
                            "actions": current_state['actions'] + [len(candidate)+1],
                        }

            # The active state is zero after the updating, then setting the ended to True
            for i in range(batch_size):
                if len(visited[i]) == len(id2state[i]):     # It's the last active state
                    ended[i] = True

            # End?
            if ended.all():
                break

        # Move back to the start point
        for i in range(batch_size):
            results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], results[i]['dijk_path'][0]))
        """
            "paths": {
                "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        """
        # Gather the Path
        for i, result in enumerate(results):
            assert len(finished[i]) <= args.candidates
            for state_id in finished[i]:
                path_info = {
                    "trajectory": [],
                    "action": [],
                    "listener_scores": id2state[i][state_id]['scores'],
                    "listener_actions": id2state[i][state_id]['actions'],
                    "visual_feature": []
                }
                viewpoint, action = decompose_state_id(state_id)
                while action != -95:
                    state = id2state[i][state_id]
                    path_info['trajectory'].append(state['location'])
                    path_info['action'].append(action)
                    path_info['visual_feature'].append(state['feature'])
                    state_id = id2state[i][state_id]['from_state_id']
                    viewpoint, action = decompose_state_id(state_id)
                state = id2state[i][state_id]
                path_info['trajectory'].append(state['location'])
                for need_reverse_key in ["trajectory", "action", "visual_feature"]:
                    path_info[need_reverse_key] = path_info[need_reverse_key][::-1]
                result['paths'].append(path_info)

        return results

    def beam_search(self, speaker):
        """
        :param speaker: The speaker to be used in searching.
        :return:
        {
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "speaker_scores": [log_prob_word1, log_prob_word2, ..., ],
            }]
        }
        """
        self.env.reset()
        results = self._dijkstra()
        """
        return from self._dijkstra()
        [{
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                    "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }]
        }]
        """

        # Compute the speaker scores:
        for result in results:
            lengths = []
            num_paths = len(result['paths'])
            for path in result['paths']:
                assert len(path['trajectory']) == (len(path['visual_feature']) + 1)
                lengths.append(len(path['visual_feature']))
            max_len = max(lengths)
            img_feats = torch.zeros(num_paths, max_len, 36, self.feature_size + args.angle_feat_size)
            can_feats = torch.zeros(num_paths, max_len, self.feature_size + args.angle_feat_size)
            for j, path in enumerate(result['paths']):
                for k, feat in enumerate(path['visual_feature']):
                    img_feat, can_feat = feat
                    img_feats[j][k] = img_feat
                    can_feats[j][k] = can_feat
            img_feats, can_feats = img_feats.cuda(), can_feats.cuda()
            features = ((img_feats, can_feats), lengths)
            insts = np.array([result['instr_encoding'] for _ in range(num_paths)])
            seq_lengths = np.argmax(insts == self.tok.word_to_index['<EOS>'], axis=1)   # len(seq + 'BOS') == len(seq + 'EOS')
            insts = torch.from_numpy(insts).cuda()
            speaker_scores = speaker.teacher_forcing(train=True, features=features, insts=insts, for_listener=True)
            for j, path in enumerate(result['paths']):
                path.pop("visual_feature")
                path['speaker_scores'] = -speaker_scores[j].detach().cpu().numpy()[:seq_lengths[j]]
        return results

    def beam_search_test(self, speaker):
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        looped = False
        self.results = {}
        while True:
            for traj in self.beam_search(speaker):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj
            if looped:
                break

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.critic.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.critic.eval()
        with torch.no_grad():
            super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            # self.feedback = 'teacher'
            # self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), 40.)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.critic_optimizer.step()
        self.discriminator_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.encoder.train()
        self.decoder.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                # if args.ml_weight != 0:
                #     self.feedback = 'teacher'
                #     self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                # self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False
            self.loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), 40.)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.critic_optimizer.step()
            self.discriminator_optimizer.step()

    def save(self, epoch, path, schedulers=None, best_val=None):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer=None):
            entry = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }
            if optimizer is not None:
                entry['optimizer'] = optimizer.state_dict()
            states[name] = entry
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer),
                     ("discriminator", self.discriminator)]  # weights saved, no optimizer
        for param in all_tuple:
            create_state(*param)
        if schedulers is not None:
            states['schedulers'] = [s.state_dict() for s in schedulers]
        if best_val is not None:
            states['best_val'] = best_val
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer=None):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            # Filter to only keys that exist in current model (backward compat)
            state.update({k: v for k, v in states[name]['state_dict'].items() if k in model_keys})
            model.load_state_dict(state)
            if optimizer is not None and args.loadOptim and 'optimizer' in states[name]:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer),
                     ("discriminator", self.discriminator)]  # no optimizer to load
        for param in all_tuple:
            recover_state(*param)
        epoch = states['encoder']['epoch'] - 1
        scheduler_states = states.get('schedulers', None)
        best_val = states.get('best_val', None)
        return epoch, scheduler_states, best_val

