import argparse
import logging
import random
import os
import torch
import pickle
import yaml
import json
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim
from data import SQuADDataset
from model import QANet
from math import log2
from metrics import *


def evaluation(model, dataset, eval_file, num_batches):
    answer_dict = {}
    losses = []
    for i in random.sample(range(0, len(dataset)), num_batches):
        passage_w, passage_c, question_w, question_c, y1, y2, ids = dataset[i]
        passage_w, passage_c = passage_w.to(device), passage_c.to(device)
        question_w, question_c = question_w.to(device), question_c.to(device)
        y1, y2 = y1.to(device), y2.to(device)
        loss, p1, p2 = model.evaluate([passage_w, passage_c, question_w, question_c], y1, y2)
        losses.append(loss)
        yp1 = torch.argmax(p1, 1)
        yp2 = torch.argmax(p2, 1)
        yps = torch.stack([yp1, yp2], dim=1)
        ymin, _ = torch.min(yps, 1)
        ymax, _ = torch.max(yps, 1)
        answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
        answer_dict.update(answer_dict_)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    return metrics, answer_dict


def train(model_params, launch_params):
    with open(launch_params['word_emb_file'], "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(launch_params['char_emb_file'], "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(launch_params['train_eval_file'], "r") as fh:
        train_eval_file = json.load(fh)
    with open(launch_params['dev_eval_file'], "r") as fh:
        dev_eval_file = json.load(fh)

    writer = SummaryWriter(os.path.join(launch_params['log'], launch_params['prefix']))
    
    lr = launch_params['learning_rate']
    base_lr = 1.0
    warm_up = launch_params['lr_warm_up_num']
    model_params['word_mat'] = word_mat
    model_params['char_mat'] = char_mat
    
    logging.info('Load dataset and create model.')
    dev_dataset = SQuADDataset(launch_params['dev_record_file'], launch_params['test_num_batches'], 
                               launch_params['batch_size'], launch_params['word2ind_file'])
    if launch_params['fine_tuning']:
        train_dataset = SQuADDataset(launch_params['train_record_file'], launch_params['fine_tuning_steps'], 
                                    launch_params['batch_size'], launch_params['word2ind_file'])
        model_args = pickle.load(open(launch_params['args_filename'], 'rb'))
        model = QANet(**model_args)
        model.load_state_dict(torch.load(launch_params['dump_filename']))
        model.to(device)
    else:
        train_dataset = SQuADDataset(launch_params['train_record_file'], launch_params['num_steps'], 
                                    launch_params['batch_size'], launch_params['word2ind_file'])
        model = QANet(**model_params).to(device)
        launch_params['fine_tuning_steps'] = 0
    
    params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=base_lr, betas=(launch_params['beta1'], launch_params['beta2']), eps=1e-7, weight_decay=3e-7)
    cr = lr / log2(warm_up)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * log2(ee + 1) if ee < warm_up else lr)
    qt = False
    logging.info('Start training.')
    for iter in range(launch_params['num_steps']):
        try:
            passage_w, passage_c, question_w, question_c, y1, y2, ids = train_dataset[iter]
            passage_w, passage_c = passage_w.to(device), passage_c.to(device)
            question_w, question_c = question_w.to(device), question_c.to(device)
            y1, y2 = y1.to(device), y2.to(device)
            loss, p1, p2 = model.train_step([passage_w, passage_c, question_w, question_c], y1, y2, optimizer, scheduler)
            if iter % launch_params['train_interval'] == 0:
                logging.info('Iteration %d; Loss: %f', iter+launch_params['fine_tuning_steps'], loss)
                writer.add_scalar('Loss', loss, iter+launch_params['fine_tuning_steps'])
            if iter % launch_params['train_sample_interval'] == 0:
                start = torch.argmax(p1[0, :]).item()
                end = torch.argmax(p2[0, start:]).item()+start
                passage = train_dataset.decode(passage_w)
                question = train_dataset.decode(question_w)
                generated_answer = train_dataset.decode(passage_w[:, start:end+1])
                real_answer = train_dataset.decode(passage_w[:, y1[0]:y2[0]+1])
                logging.info('Train Sample:\n Passage: %s\nQuestion: %s\nOriginal answer: %s\nGenerated answer: %s',
                        passage, question, real_answer, generated_answer)
            if iter % launch_params['test_interval'] == 0:
                metrics, _ = evaluation(model, train_dataset, train_eval_file, launch_params['val_num_batches'])
                logging.info("VALID loss %f F1 %f EM %f", metrics['loss'], metrics['f1'], metrics['exact_match'])
                writer.add_scalar('Valid_loss', metrics['loss'], iter)
                writer.add_scalar('Valid_f1', metrics['f1'], iter)
                writer.add_scalar('Valid_em', metrics['exact_match'], iter)
            if iter % launch_params['test_interval'] == 0:
                metrics, _ = evaluation(model, dev_dataset, dev_eval_file, launch_params['test_num_batches'])
                logging.info("TEST loss %f F1 %f EM %f", metrics['loss'], metrics['f1'], metrics['exact_match'])
                writer.add_scalar('Test_loss', metrics['loss'], iter)
                writer.add_scalar('Test_f1', metrics['f1'], iter)
                writer.add_scalar('Test_em', metrics['exact_match'], iter)
        except RuntimeError as e:
            logging.error(str(e))
        except KeyboardInterrupt:
            break
    torch.save(model.cpu().state_dict(), launch_params['dump_filename'])
    pickle.dump(model_params, open(launch_params['args_filename'], 'wb'))
    logging.info('Model has been saved.')


def test(cfg):
    logging.info('Model is loading...')
    with open(cfg['dev_eval_file'], "r") as fh:
        dev_eval_file = json.load(fh)
    dev_dataset = SQuADDataset(cfg['dev_record_file'], -1, cfg['batch_size'], cfg['word2ind_file'])
    model_args = pickle.load(open(cfg['args_filename'], 'rb'))
    model = QANet(**model_args)

    model.load_state_dict(torch.load(cfg['dump_filename']))
    model.to(device)
    
    metrics, answer_dict = evaluation(model, dev_dataset, dev_eval_file, len(dev_dataset))
    with open('logs/answers.json', 'w') as f:
        json.dump(answer_dict, f)
    logging.info("TEST loss %f F1 %f EM %f\n", metrics["loss"], metrics["f1"], metrics["exact_match"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='C', type=str, default='config/train_config.yml')
    parser.add_argument("--mode", action="store", dest="mode", default="train", help="train/test/debug/fine-tuning")
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    model_params = cfg['model_params']
    launch_params = cfg['launch_params']
    home = os.path.expanduser('.')
    log_filename = os.path.join(home, launch_params['log'], launch_params['prefix'] + '.log')
    dump_filename = os.path.join(home, 'model_dumps', launch_params['prefix'], launch_params['prefix'] + '.model')
    args_filename = os.path.join(home, 'model_dumps', launch_params['prefix'], launch_params['prefix'] + '.args')
    for key, item in launch_params.items():
        if 'file' in key:
            launch_params[key] = os.path.join(home, item)
            os.makedirs(os.path.dirname(launch_params[key]), exist_ok=True)
    launch_params['dump_filename'] = dump_filename
    launch_params['args_filename'] = args_filename
    launch_params['word2ind_file'] = os.path.join(home, launch_params['word2ind_file'])
    for path in [log_filename, dump_filename]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('Current device is %s.', device)

    launch_params['fine_tuning'] = args.mode == 'fine_tuning' ? True : False
    
    if args.mode == 'train' or args.mode == 'fine_tuning':
        train(model_params, launch_params)
    elif args.mode == 'debug':
        launch_params['batch_size'] = 2
        launch_params['num_steps'] = 32
        launch_params['test_num_batches'] = 2
        launch_params['val_num_batches'] = 2
        
        train(model_params, launch_params)
    elif args.mode == 'test':
        test(launch_params)
    else:
        logging.error('Unknown mode')
