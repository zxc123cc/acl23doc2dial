import torch
from modelscope.trainers.nlp.document_grounded_dialog_rerank_trainer import Reporting, LossHistory
from modelscope.utils.logger import get_logger
from modelscope.trainers import EpochBasedTrainer
import random
import numpy as np
from modelscope.preprocessors.nlp.document_grounded_dialog_rerank_preprocessor import \
    DocumentGroundedDialogRerankPreprocessor
import torch.nn.functional as F
import os

from transformers import AdamW, get_linear_schedule_with_warmup

from rerank_src.rerank_model import RerankModel
from torch import amp


class EMA(object):
    """
    Maintains (exponential) moving average of a set of parameters.
    使用ema累积模型参数
    Args:
        parameters (:obj:`list`): 需要训练的模型参数
        decay (:obj:`float`): 指数衰减率
        use_num_updates (:obj:`bool`, optional, defaults to True): Whether to use number of updates when computing averages
    Examples::
        >>> ema = EMA(module.parameters(), decay=0.995)
        >>> # Train for a few epochs
        >>> for _ in range(epochs):
        >>>     # 训练过程中，更新完参数后，同步update shadow weights
        >>>     optimizer.step()
        >>>     ema.update(module.parameters())
        >>> # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
        >>> ema.store(module.parameters())
        >>> ema.copy_to(module.parameters())
        >>> # evaluate
        >>> ema.restore(module.parameters())
    Reference:
        [1]  https://github.com/fadel/pytorch_ema
    """  # noqa: ignore flake8"

    def __init__(
        self,
        parameters,
        decay,
        use_num_updates=True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)



class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.emb_name = 'embeddings.word_embeddings.'
        self.epsilon = 1.0
        # self.epsilon = 0.5

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = 1.0
        # self.epsilon = 0.5
        self.emb_name = 'embeddings.word_embeddings.'
        self.alpha = 0.3

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]




logger = get_logger()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def block_shuffle(iter, *, block_size=20000, rand=random):
    """
    shuffle the possibly endless iterator by blocks
    Good shuffling over multiple files:
    block_shuffle(read_lines(files, shuffled_files=rand), rand=rand, block_size=100000)
    :param iter: the iterator we will yield shuffled items from
    :param block_size: size of memory to use for block shuffling
    :param rand: rand.shuffle will be used on the list block
    :return:
    """
    assert block_size >= 4
    block = []
    for item in iter:
        block.append(item)
        if len(block) >= block_size:
            rand.shuffle(block)
            for _ in range(block_size // 2):
                yield block.pop(-1)
    rand.shuffle(block)
    for bi in block:
        yield bi

class TransformerOptimize:
    """
    Collects standard steps to train transformer
    call step_loss after computing each loss
    """

    def __init__(self, hypers, num_instances_to_train_over: int, model):
        self.step = 0
        self.global_step = 0
        self.hypers = hypers
        self.model = model
        instances_per_step = hypers['full_train_batch_size'] // hypers['gradient_accumulation_steps']
        self.reporting = Reporting(recency_weight=0.0001 * instances_per_step)
        args = self.hypers

        self.t_total = num_instances_to_train_over // args['full_train_batch_size']

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                args['weight_decay'],
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            },
        ]

        warmup_instances = args['warmup_instances']
        if hasattr(
                args, 'warmup_fraction'
        ) and args['warmup_fraction'] > 0 >= args['warmup_instances']:
            warmup_instances = \
                args['warmup_fraction'] * num_instances_to_train_over
        if warmup_instances < 0:
            warmup_instances = 0

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args['learning_rate'],
            eps=args['adam_epsilon'])
        num_warmup_steps = warmup_instances // args['full_train_batch_size']
        # num_warmup_steps = int(args['warmup_fraction']*self.t_total)
        print('num_warmup_steps',num_warmup_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.t_total)

        # Check if saved optimizer or scheduler states exist
        if args['resume_from'] and os.path.isfile(os.path.join(args['resume_from'], 'optimizer.pt')) and \
                os.path.isfile(os.path.join(args['resume_from'], 'scheduler.pt')):
            resume_from = args['resume_from']
        # elif os.path.isfile(os.path.join(args['model_name_or_path'], "optimizer.pt")) and \
        #         os.path.isfile(os.path.join(args['model_name_or_path'], "scheduler.pt")):
        #     resume_from = args['model_name_or_path']
        else:
            resume_from = None
        if resume_from is not None:
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(resume_from, 'optimizer.pt'),
                    map_location='cpu'))
            self.scheduler.load_state_dict(
                torch.load(
                    os.path.join(resume_from, 'scheduler.pt'),
                    map_location='cpu'))
            logger.info(f'loaded optimizer and scheduler from {resume_from}')

        if args['fp16']:
            self.model, optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=args['fp16_opt_level'])

        # multi-gpu training (should be after apex fp16 initialization)
        if args['n_gpu'] > 1:
            # NOTE: won't work at O2, only O1
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(args['n_gpu'])))

        # Distributed training (should be after apex fp16 initialization)
        # if args.local_rank != -1:
        #     self.model = torch.nn.parallel.DistributedDataParallel(
        #         self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        #     )
        # set_seed(args)
        # assert args.per_gpu_train_batch_size * (args.n_gpu if args.n_gpu > 0 else 1) * \
        #        args.world_size * args.gradient_accumulation_steps == args.full_train_batch_size
        logger.info('***** Running training *****')
        logger.info('  Instantaneous batch size per GPU = %d', args['per_gpu_train_batch_size'])
        logger.info('  Total train batch size (w. parallel, distributed & accumulation) = %d', args['full_train_batch_size'])
        logger.info('  Gradient Accumulation steps = %d', args['gradient_accumulation_steps'])
        logger.info('  Total optimization steps = %d', self.t_total)

    def should_continue(self):
        return self.global_step < self.t_total

    def backward_on_loss(self, loss, **moving_averages):
        if self.hypers['n_gpu'] > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        loss_val = loss.item()
        if self.hypers['gradient_accumulation_steps'] > 1:
            loss = loss / self.hypers['gradient_accumulation_steps']
        if self.hypers['fp16']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.reporting.moving_averages(loss=loss_val, **moving_averages)
        return loss_val

    def optimizer_step(self):
        if self.global_step >= self.t_total:
            logger.warning(
                f'Warning, exceeded total steps! {self.global_step} step of {self.t_total}'
            )
            return False
        if (self.step + 1) % self.hypers['gradient_accumulation_steps'] == 0:
            if self.hypers['max_grad_norm'] > 0:
                if self.hypers['fp16']:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer),
                        self.hypers['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.hypers['max_grad_norm'])

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1
        self.step += 1

        if self.reporting.is_time():
            self.reporting.display()
            inst_count = \
                self.hypers['world_size'] * self.hypers['n_gpu'] * self.hypers[
                    'per_gpu_train_batch_size'] * self.reporting.check_count
            learning_rate_scalar = self.scheduler.get_lr()[0]
            logger.info(
                f'{inst_count / self.reporting.elapsed_seconds()} instances per second; '
                f'{inst_count} total ({learning_rate_scalar} learn rate)')
        return True

    def step_loss(self, loss, **moving_averages):
        loss_val = self.backward_on_loss(loss, **moving_averages)
        if self.optimizer_step():
            return loss_val
        else:
            return None



class DocumentGroundedDialogRerankTrainer(EpochBasedTrainer):

    def __init__(self, model_path, dataset, **args):
        args = args['args']
        self.positive_pids = ''
        self.instances_size = 1
        # load id to positive pid map
        self.inst_id2pos_pids = dict()
        self.inst_id2pos_passages = dict()
        self.dataset = dataset
        self.model = RerankModel(model_path)
        self.preprocessor = DocumentGroundedDialogRerankPreprocessor(
            model_path, **args)
        self.tokenizer = self.preprocessor.tokenizer
        if args['model_resize']:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = self.preprocessor.device
        self.model.to(self.device)
        for jobj in self.dataset:
            self.inst_id2pos_pids[jobj['id']] = eval(jobj['positive_pids'])
            assert isinstance(eval(jobj['positive_pids']), list)
        logger.info( f'gathered positive pids for {len(self.inst_id2pos_pids)} instances')

        # remove out-of-recall
        instance_count = 0
        for jobj in self.dataset:
            inst_id = jobj['id']
            if inst_id not in self.inst_id2pos_pids:
                continue
            passages = eval(jobj['passages'])
            positive_pids = self.inst_id2pos_pids[inst_id]
            target_mask = [p['pid'] in positive_pids for p in passages]
            if not any(target_mask) or all(target_mask):
                del self.inst_id2pos_pids[inst_id]
            else:
                instance_count += 1
        if instance_count != len(self.inst_id2pos_pids):
            logger.error(
                f'!!! Mismatch between --positive_pids and --initial_retrieval! '
                f'{len(self.inst_id2pos_pids)} vs {instance_count}')

        # transformer_optimize
        if args['train_instances'] <= 0:
            args['train_instances'] = instance_count
        # MARK
        instances_to_train_over = args['train_instances'] * args['num_train_epochs'] // args['instances_size']
        self.optimizer = TransformerOptimize(args, instances_to_train_over, self.model)
        logger.info('  Num Epochs = %d', args['num_train_epochs'])
        self.optimizer.model.zero_grad()
        # MARK
        train_batch_size = \
            args['full_train_batch_size'] // args['gradient_accumulation_steps']
        self.loss_history = \
            LossHistory(
                args['train_instances'] // train_batch_size // args['instances_size']
            )
        self.args = args
        self.max_length_count = 0

        self.use_ema = args['ema']
        self.adv = args['adv']

    def one_instance(self, query, passages):
        # model = self.model
        input_dict = {'query': query, 'passages': passages}
        inputs = self.preprocessor(input_dict)
        logits = self.model(inputs).logits
        logits = F.log_softmax(logits,dim=-1)[:, 1]  # log_softmax over the binary classification
        logprobs = F.log_softmax(logits, dim=0)  # log_softmax over the passages
        # we want the logits rather than the logprobs as the teacher labels
        return logprobs

    def limit_gpu_sequences_binary(self, passages, target_mask, rand):
        if len(passages) > self.args['max_num_seq_pairs_per_device']:
            num_pos = min(
                sum(target_mask),
                self.args['max_num_seq_pairs_per_device'] // 2)
            num_neg = self.args['max_num_seq_pairs_per_device'] - num_pos
            passage_and_pos = list(zip(passages, target_mask))
            rand.shuffle(passage_and_pos)
            pos_count = 0
            neg_count = 0
            passages = []
            target_mask = []
            for passage, mask in passage_and_pos:
                if mask and pos_count < num_pos:
                    passages.append(passage)
                    target_mask.append(mask)
                    pos_count += 1
                elif not mask and neg_count < num_neg:
                    passages.append(passage)
                    target_mask.append(mask)
                    neg_count += 1
        return passages, target_mask

    def limit_gpu_sequences(self, passages, correctness, rand):
        if len(passages) > self.args['max_num_seq_pairs_per_device']:
            num_pos = min(
                sum([c > 0 for c in correctness]),
                self.args['max_num_seq_pairs_per_device'] // 2)
            num_neg = self.args['max_num_seq_pairs_per_device'] - num_pos
            passage_and_pos = list(zip(passages, correctness))
            rand.shuffle(passage_and_pos)
            pos_count = 0
            neg_count = 0
            passages = []
            correctness = []
            for passage, pos in passage_and_pos:
                if pos > 0 and pos_count < num_pos:
                    passages.append(passage)
                    correctness.append(pos)
                    pos_count += 1
                elif pos == 0 and neg_count < num_neg:
                    passages.append(passage)
                    correctness.append(pos)
                    neg_count += 1
        return passages, correctness

    def passage_correctness(self, pid, positive_pids, positive_dids):
        if pid in positive_pids:
            return 1.0
        elif positive_dids and pid[:pid.index('::')] in positive_dids:
            return self.args['doc_match_weight']
        else:
            return 0

    def train(self):
        rand = random.Random()
        if self.use_ema:
            ema = EMA(self.model.parameters(), decay=0.999)
        if self.adv:
            fgm = FGM(self.model)

        while self.optimizer.should_continue():
            self.model.train()
            dataset = block_shuffle(self.dataset, block_size=100000, rand=rand)
            last_passages = []
            for line_ndx, jobj in enumerate(dataset):
                # print(jobj)
                inst_id = jobj['id']
                if inst_id not in self.inst_id2pos_pids:
                    continue
                if line_ndx % self.args['world_size'] != self.args['global_rank']:
                    continue
                query = jobj['input'] if 'input' in jobj else jobj['query']

                if self.args['add_neg']:
                    passages = eval(jobj['passages']) + last_passages
                else:
                    passages = eval(jobj['passages'])

                last_passages = eval(jobj['passages'])
                positive_pids = self.inst_id2pos_pids[inst_id]
                if self.args['doc_match_weight'] > 0:
                    positive_dids = [pid[:pid.index('::')] for pid in positive_pids]
                else:
                    positive_dids = None
                correctness = [self.passage_correctness(p['pid'], positive_pids,positive_dids) for p in passages]
                passages, correctness = self.limit_gpu_sequences(passages, correctness, rand)
                logits = self.one_instance(query, passages)
                # nll = -(logits[target_mask].sum())  # TODO: instead take the weighted sum
                nll = -(logits.dot(torch.tensor(correctness).to(logits.device)))   # loss

                loss_val = self.optimizer.step_loss(nll)   # 误差传播和参数更新
                # loss_val = self.optimizer.backward_on_loss(nll)

                self.loss_history.note_loss(loss_val)

                if self.adv:
                    fgm.attack()
                    logits = self.one_instance(query, passages)
                    # nll = -(logits[target_mask].sum())  # TODO: instead take the weighted sum
                    nll = -(logits.dot(torch.tensor(correctness).to(logits.device)))
                    loss_val_adv = self.optimizer.step_loss(nll)
                    # loss_val_adv = self.optimizer.backward_on_loss(nll)
                    fgm.restore()

                # self.optimizer.optimizer_step()

                if self.use_ema:
                    ema.update(self.model.parameters())

                # if self.loss_history.batch_count % self.loss_history.record_loss_every == 0 and \
                #         len(self.loss_history.loss_history)%2==0 and len(self.loss_history.loss_history) >= 30:
                #     save_transformer(self.args, self.optimizer.model, self.tokenizer,
                #                      save_dir=f'./model_storage/rerank_output_infoxlm-large_ckpt{len(self.loss_history.loss_history)}')

                if self.use_ema:
                    ema.restore(self.model.parameters())

                if not self.optimizer.should_continue():
                    break


        get_length = self.args['max_seq_length']
        logger.info(f'loss_history = {self.loss_history.loss_history}')
        logger.info(f'truncated to max length ({get_length}) {self.max_length_count} times')
        if self.use_ema:
            ema.store(self.model.parameters())
            ema.copy_to(self.model.parameters())

        save_transformer(self.args, self.optimizer.model, self.tokenizer)



def save_transformer(hypers, model, tokenizer, *, save_dir=None):
    if hypers['global_rank'] == 0:
        if save_dir is None:
            save_dir = hypers['output_dir']
        # Create output directory if needed
        os.makedirs(save_dir, exist_ok=True)
        logger.info('Saving model checkpoint to %s', save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, 'module') else model)  # Take care of distributed/parallel training
        torch.save(hypers, os.path.join(save_dir, 'training_args.bin'))
        model_to_save.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)