
import torch
import random

import os
import time
import json
import logging
import numpy as np
from collections import defaultdict
from speaker import Speaker

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter


log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# File logger: save all output to DILLM/log/, one file per run named by timestamp
_run_timestamp = time.strftime('%Y%m%d_%H%M%S')
_file_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
os.makedirs(_file_log_dir, exist_ok=True)
_log_file_path = os.path.join(_file_log_dir, '%s_%s.log' % (_run_timestamp, args.name))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(_log_file_path, encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)
# Redirect built-in print to logger so all existing print() calls are captured
import builtins as _builtins
_original_print = _builtins.print
def _logging_print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    logger.info(msg)
_builtins.print = _logging_print

logger.info('Log file: %s' % _log_file_path)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'
VITB32_FEATURES = 'img_features/CLIP-512.tsv'
RN50X4_FEATURES = 'img_features/CLIP-ResNet-50x4-views.tsv'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'vitb32':
    features = VITB32_FEATURES
elif args.features == 'rn50x4':
    features = RN50X4_FEATURES

if args.fast_train:
    name, ext = os.path.splitext(features)
    features = name + "-fast" + ext

feedback_method = args.feedback # teacher or sample

print(args)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
                continue

            print("............ Evaluating %s ............." % env_name)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            path_id = next(iter(path2inst.keys()))
            print("Inference: ", tok.decode_sentence(path2inst[path_id]))
            print("GT: ", evaluator.gt[str(path_id)]['instructions'])
            bleu_score, precisions = evaluator.bleu_score(path2inst)

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            # Save the model according to the bleu score
            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
                speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

            # Screen print out
            print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    
    '''
    if args.load:
        checkpoint = torch.load(args.load)
        listner.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']  # 如果需要的话加载损失
        print("Resuming training from epoch", start_epoch)
    '''
    '''
    # 1. 加载模型和优化器状态
    if args.load:
        print("Loading model from %s" % args.load)
        checkpoint = torch.load(args.load)
        listner.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['epoch']
        loss = checkpoint['loss']
        print("Resuming training from epoch", start_iter)
    else:
        start_iter = 0
    '''
    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)

    start_iter = 0
    scheduler_states = None
    loaded_best_val = None
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter, scheduler_states, loaded_best_val = listner.load(os.path.join(args.load))

    start = time.time()

    # LR schedulers: cosine annealing (optional)
    schedulers = None
    if args.cosine_annealing:
        # Always use T_max=n_iters so the curve shape is identical regardless of resume point
        schedulers = [
            CosineAnnealingLR(listner.encoder_optimizer, T_max=n_iters, eta_min=args.lr_min),
            CosineAnnealingLR(listner.decoder_optimizer, T_max=n_iters, eta_min=args.lr_min),
            CosineAnnealingLR(listner.critic_optimizer, T_max=n_iters, eta_min=args.lr_min),
        ]
        # Restore scheduler state from checkpoint if available; otherwise fast-forward
        if start_iter > 0:
            if scheduler_states is not None:
                for s, sd in zip(schedulers, scheduler_states):
                    s.load_state_dict(sd)
                print('Restored LR schedulers from checkpoint (iter %d)' % start_iter)
            else:
                # Legacy checkpoint without scheduler states: fast-forward
                for s in schedulers:
                    for _ in range(start_iter):
                        s.step()
                print('Fast-forwarded LR schedulers %d steps (legacy checkpoint)' % start_iter)
        print('LR schedule: cosine annealing %g -> %g over %d iters (resuming at %d)' % (
            args.lr, args.lr_min, n_iters, start_iter))
    else:
        print('LR schedule: constant lr=%g (no scheduler)' % args.lr)

    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
    if loaded_best_val is not None:
        # New checkpoint: restore directly
        for env_name in best_val:
            if env_name in loaded_best_val:
                best_val[env_name]['accu'] = loaded_best_val[env_name]['accu']
                best_val[env_name]['state'] = loaded_best_val[env_name]['state']
        print('Restored best_val from checkpoint: val_seen=%.4f, val_unseen=%.4f'
              % (best_val['val_seen']['accu'], best_val['val_unseen']['accu']))
    # Command-line overrides (always take precedence, useful for legacy checkpoints)
    if args.best_val_seen is not None:
        best_val['val_seen']['accu'] = args.best_val_seen
        best_val['val_seen']['state'] = '(from --bestValSeen)'
        print('best_val_seen set to %.4f via --bestValSeen' % args.best_val_seen)
    if args.best_val_unseen is not None:
        best_val['val_unseen']['accu'] = args.best_val_unseen
        best_val['val_unseen']['state'] = '(from --bestValUnseen)'
        print('best_val_unseen set to %.4f via --bestValUnseen' % args.best_val_unseen)
    print('Total trainable parameters: %d (%.2fM)' % (listner.total_params, listner.total_params / 1e6))
    if args.fast_train:
        log_every = 40
    for idx in range(start_iter, n_iters, log_every):
        
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:     # The default training process
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)   # Train interval iters
        else:
            if args.accumulate_grad:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    args.ml_weight = 0.2
                    listner.accumulate_gradient(feedback_method)
                    listner.env = aug_env

                    # Train with Back Translation
                    args.ml_weight = 0.6        # Sem-Configuration
                    listner.accumulate_gradient(feedback_method, speaker=speaker)
                    listner.optim_step()
            else:
                # for _ in range(interval // 2):
                for _ in range(interval // 2 // 8 + 1):
                    # Train with GT data
                    listner.env = train_env
                    args.ml_weight = 0.2
                    listner.train(1, feedback=feedback_method)

                    # Train with Back Translation
                    listner.env = aug_env
                    args.ml_weight = 0.6
                    listner.train(1, feedback=feedback_method, speaker=speaker)

        # Step LR schedulers (one step per training iteration)
        if schedulers is not None:
            for s in schedulers:
                for _ in range(interval):
                    s.step()

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        writer.add_scalar("total_actions", total, idx)
        length = max(len(listner.logs['critic_loss']), 1)
        writer.add_scalar("max_length", length, idx)

        critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
        writer.add_scalar("loss/critic", critic_loss, idx)

        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        writer.add_scalar("policy_entropy", entropy, idx)

        reward = sum(listner.logs['reward'])/max(len(listner.logs['reward']), 1)
        writer.add_scalar("reward", reward, idx)
        if schedulers is not None:
            writer.add_scalar("lr/encoder_decoder", schedulers[0].get_last_lr()[0], idx)
            writer.add_scalar("lr/critic", schedulers[2].get_last_lr()[0], idx)

        print("total_actions", total)
        print("max_length", length)

        # Run validation
        loss_str = ""
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

            # Clear step_time logs before validation to get per-env inference speed
            listner.logs['step_time'].clear()
            listner.logs['rollout_time'].clear()
            listner.logs['rollout_steps'].clear()

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.3f' % (metric, val)

            # Inference speed stats for this env
            if listner.logs['step_time']:
                avg_step_ms = np.mean(listner.logs['step_time']) * 1000
                steps_per_sec = 1.0 / np.mean(listner.logs['step_time'])
                avg_rollout_sec = np.mean(listner.logs['rollout_time'])
                avg_steps = np.mean(listner.logs['rollout_steps'])
                loss_str += ', step_time: %.1fms' % avg_step_ms
                loss_str += ', steps/s: %.1f' % steps_per_sec
                writer.add_scalar("speed/step_time_ms_%s" % env_name, avg_step_ms, idx)
                writer.add_scalar("speed/steps_per_sec_%s" % env_name, steps_per_sec, idx)
                writer.add_scalar("speed/avg_rollout_sec_%s" % env_name, avg_rollout_sec, idx)
                print('[%s] Inference speed — avg step: %.1f ms (%.1f steps/s), avg rollout: %.2f s (%.1f steps)'
                      % (env_name, avg_step_ms, steps_per_sec, avg_rollout_sec, avg_steps))

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)), schedulers=schedulers, best_val=best_val)

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

        if iter % 5000 == 0:
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)), schedulers=schedulers, best_val=best_val)
        '''
        if iter % 500 == 0:
            checkpoint_path = os.path.join("snap", args.name, "state_dict", "Iter_%06d.pth" % (iter))
            # 保存模型、优化器以及其他训练信息
            torch.save({
                'epoch': iter,
                'model_state_dict': listner.state_dict(),
                'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),  # 保存encoder优化器的状态
                'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),  # 保存decoder优化器的状态
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),  # 保存critic优化器的状态
                }, checkpoint_path)
        '''


    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)), schedulers=schedulers, best_val=best_val)


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load)[0], args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def beam_valid(train_env, tok, val_envs={}):
    listener = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = Speaker(train_env, listener, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    print("Loaded the listener model at iter % d" % listener.load(args.load)[0])

    final_log = ""
    for env_name, (env, evaluator) in val_envs.items():
        listener.logs = defaultdict(list)
        listener.env = env

        listener.beam_search_test(speaker)
        results = listener.results

        def cal_score(x, alpha, avg_speaker, avg_listener):
            speaker_score = sum(x["speaker_scores"]) * alpha
            if avg_speaker:
                speaker_score /= len(x["speaker_scores"])
            # normalizer = sum(math.log(k) for k in x['listener_actions'])
            normalizer = 0.
            listener_score = (sum(x["listener_scores"]) + normalizer) * (1-alpha)
            if avg_listener:
                listener_score /= len(x["listener_scores"])
            return speaker_score + listener_score

        if args.param_search:
            # Search for the best speaker / listener ratio
            interval = 0.01
            logs = []
            for avg_speaker in [False, True]:
                for avg_listener in [False, True]:
                    for alpha in np.arange(0, 1 + interval, interval):
                        result_for_eval = []
                        for key in results:
                            result_for_eval.append({
                                "instr_id": key,
                                "trajectory": max(results[key]['paths'],
                                                  key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                                  )['trajectory']
                            })
                        score_summary, _ = evaluator.score(result_for_eval)
                        for metric,val in score_summary.items():
                            if metric in ['success_rate']:
                                print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                                      (avg_speaker, avg_listener, alpha, val))
                                logs.append((avg_speaker, avg_listener, alpha, val))
            tmp_result = "Env Name %s\n" % (env_name) + \
                    "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f\n" % max(logs, key=lambda x: x[3])
            print(tmp_result)
            # print("Env Name %s" % (env_name))
            # print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
            #       max(logs, key=lambda x: x[3]))
            final_log += tmp_result
            print()
        else:
            avg_speaker = True
            avg_listener = True
            alpha = args.alpha

            result_for_eval = []
            for key in results:
                result_for_eval.append({
                    "instr_id": key,
                    "trajectory": [(vp, 0, 0) for vp in results[key]['dijk_path']] + \
                                  max(results[key]['paths'],
                                   key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                  )['trajectory']
                })
            # result_for_eval = utils.add_exploration(result_for_eval)
            score_summary, _ = evaluator.score(result_for_eval)

            if env_name != 'test':
                loss_str = "Env Name: %s" % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                              (avg_speaker, avg_listener, alpha, val))
                    loss_str += ",%s: %0.4f " % (metric, val)
                print(loss_str)
            print()

            if args.submit:
                json.dump(
                    result_for_eval,
                    open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    print(final_log)


def set_seed(seed):
    """Set all RNG seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup():
    if args.seed is not None:
        print("Random seed: %d (deterministic)" % args.seed)
    else:
        print("Random seed: None (non-deterministic)")
    set_seed(args.seed)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(features)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        # pass
        val_env_names.append('train')

    if not args.beam:
        pass
        # val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, val_envs=val_envs)
        else:
            valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False


def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))


def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False

