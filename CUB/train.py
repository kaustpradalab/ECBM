"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import pdb
import os
import sys
import time
from datetime import timedelta
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy

from CUB import probe, tti, gen_cub_synthetic, hyperopt
from CUB.dataset import load_data, find_class_imbalance
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY


def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            #inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        
        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters
    return loss_meter, acc_meter

def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels, index = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels = attr_labels.T.float().cuda()
            attr_labels_var = torch.autograd.Variable(attr_labels)
            #attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        inputs_var = torch.autograd.Variable(inputs).cuda()
        #inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        #labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append((
                        1.0 * attr_criterion[i](outputs[i+out_start].squeeze(), attr_labels_var[:, i]) + \
                        0.4 * attr_criterion[i](aux_outputs[i+out_start].squeeze(), attr_labels_var[:, i])
                    ) * args.attr_loss_weight)
        else: #testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))

        if args.bottleneck: #attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(outputs[0], labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses)/ args.n_attributes
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else: #finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, acc_meter

def train(model, args, excluded_attr_indexes=set(), suffix='', logger=None):
    time_start = time.time()

    # Determine imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)

    if logger is None:
        if os.path.exists(args.log_dir): # job restarted by cluster
            raise FileExistsError(args.log_dir)
            for f in os.listdir(args.log_dir):
                os.remove(os.path.join(args.log_dir, f))
        else:
            os.makedirs(args.log_dir)
        logger = Logger(os.path.join(args.log_dir, 'log.txt'))

    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert(imbalance is not None)
            for ratio in imbalance:
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
        tmp = attr_criterion
        attr_criterion = [x for i,x in enumerate(tmp) if i not in excluded_attr_indexes]
    else:
        attr_criterion = None

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    logger.write(f"Stop epoch: {stop_epoch}\n")

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    if args.ckpt: #retraining
        train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling, excluded_attr_indexes=excluded_attr_indexes)
        val_loader = None
    else:
        train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling, excluded_attr_indexes=excluded_attr_indexes)
        val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr, excluded_attr_indexes=excluded_attr_indexes)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True)
        else:
            train_loss_meter, train_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True)
 
        if not args.ckpt: # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
        
            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
                else:
                    val_loss_meter, val_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion, args, is_training=False)

        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, f'best_model_{suffix}.pth'))
            #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\tTime: %s\n'
                % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch, str(timedelta(seconds=time.time() - time_start))[:-7])) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            logger.write(f'Current lr: {scheduler.get_lr()}\n')

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            logger.write("Early stopping because of low accuracy\n")
            break
        if epoch - best_val_epoch >= 100:
            logger.write("Early stopping because acc hasn't improved for a long time\n")
            break


TOP_CONCEPTS = [13, 36, 43, 97, 75, 83, 94, 41, 18]


def train_concept_level(args):
    excluded = int(os.environ['CONCEPT_LEVEL'])
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    excluded = set({TOP_CONCEPTS[excluded]})
    suffix = '_'.join(map(str, excluded))
    logger = Logger(os.path.join(args.log_dir, f'log_{suffix}.txt'))

    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                        num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes-len(excluded),
                        expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    logger.write(f"Training concept level {excluded}\n")
    train(model, args, excluded_attr_indexes=excluded, suffix=suffix, logger=logger)


def train_data_level(args):
    rate = float(os.environ['DATA_LEVEL'])
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    dataset_len = 4796  # CUB
    cnt = int(rate * dataset_len)

    logger = Logger(os.path.join(args.log_dir, f'log_{cnt:05}.txt'))
    logger.write(f"Training data level {cnt} samples\n")
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                        num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                        expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args, excluded_attr_indexes={f'REMOVE_DATA_{cnt}'}, suffix=f'{cnt:05}', logger=logger)


def train_single_level(args):
    rate = float(os.environ['SINGLE_LEVEL'])
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    dataset_len = 4796  # CUB
    cnt = int(rate * dataset_len)

    logger = Logger(os.path.join(args.log_dir, f'log_{cnt:05}.txt'))
    logger.write(f"Training single level {cnt} samples\n")
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                        num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                        expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args, excluded_attr_indexes={f'REMOVE_DATA_{cnt}S'}, suffix=f'{cnt:05}', logger=logger)


@torch.no_grad
def run_eval_epoch(args, task, ckp_suffix, excluded_attr_indexes=set()):
    val_data_path = os.path.join(BASE_DIR, args.data_dir, 'val.pkl')
    loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr, excluded_attr_indexes=excluded_attr_indexes)
    ckp = os.path.join(args.log_dir, f'best_model_{ckp_suffix}.pth').replace('Eval', task)
    model = torch.load(ckp, weights_only=False).eval()

    assert not args.bottleneck
    all_preds, all_labels, all_concepts, all_names = list(), list(), list(), list()
    all_concepts_logits = list()
    all_concepts_truths = list()

    for _, (inputs, labels, attr_labels, index) in enumerate(loader):
        if args.n_attributes > 1:
            attr_labels = [i.long() for i in attr_labels]
            attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
        else:
            if isinstance(attr_labels, list):
                attr_labels = attr_labels[0]
            attr_labels = attr_labels.unsqueeze(1)
        attr_labels = attr_labels.T.float().cuda()
        attr_labels_var = torch.autograd.Variable(attr_labels)

        inputs_var = torch.autograd.Variable(inputs).cuda()
        labels_var = torch.autograd.Variable(labels).cuda()

        outputs = model(inputs_var)
        #print(len(outputs))
        all_concepts_logits.append(torch.cat(outputs[1:], -1).cpu())
        all_concepts_truths.append(attr_labels.cpu())

        concepts = torch.sigmoid(torch.cat(outputs[1:], -1))
        concept_sign = torch.where(torch.round(concepts) == attr_labels, 1, -1)

        name = map(lambda i: loader.dataset.data[i]['img_path'], index)
        _, pred = outputs[0].topk(1, 1, True, True)
        all_preds += pred[:, 0].tolist()
        all_labels += labels_var.tolist()
        all_concepts.append((concepts * concept_sign).cpu())
        all_names += list(name)

    all_concepts_logits = torch.cat(all_concepts_logits, 0)
    all_concepts_truths = torch.cat(all_concepts_truths, 0)
    ce = torch.nn.functional.cross_entropy(all_concepts_logits, all_concepts_truths)
    acc = (torch.sigmoid(all_concepts_logits).round() == all_concepts_truths).float().mean()
    print(f'{acc=:.4%} ce={float(ce)}')

    all_concepts = torch.cat(all_concepts, 0).numpy()
    csv_result = list()
    for name, pred, label, concept in zip(all_names, all_preds, all_labels, all_concepts):
        csv_result.append(f'{name},{label},{pred},{",".join(map(lambda x: f"{x:.6f}", concept))}')
    csv_result = ['name,label,pred,' + ','.join(map(str, range(all_concepts.shape[1])))] + csv_result + ['']
    with open(os.path.join(args.log_dir, f'{task}_{ckp_suffix}.csv'), 'w') as file:
        file.write('\n'.join(csv_result))


def run_eval_data(args, task):
    dataset_len = 4796  # CUB
    rate = float(os.environ['EVAL'])
    cnt = int(rate / 100 * dataset_len)
    excluded_attr_indexes = {f'REMOVE_DATA_{cnt}' + ('' if task == 'DataLevel' else 'S')}
    run_eval_epoch(args, task, f'{cnt:05}', excluded_attr_indexes)


def run_eval_concept(args):
    task = 'ConceptLevel'
    concept = float(os.environ['CONCEPT'])
    excluded = set({concept})
    suffix = '_'.join(map(str, excluded))
    run_eval_epoch(args, task, suffix, excluded)


def run_eval(args):
    level = os.environ.get('EVAL', None)
    if level == 'Joint':
        run_eval_epoch(args, 'Joint', '')
    elif level == 'ConceptLevel':
        run_eval_concept(args)
    else:
        assert level in {'DataLevel', 'SingleLevel'}
        run_eval_data(args, level)


def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                      n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C_to_y(args):
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args)

def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux)
    train(model, args)

def train_X_to_Cy(args):
    model = ModelXtoCY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                       n_attributes=args.n_attributes, three_class=args.three_class, connect_CY=args.connect_CY)
    train(model, args)

def train_probe(args):
    probe.run(args)

def test_time_intervention(args):
    tti.run(args)

def robustness(args):
    gen_cub_synthetic.run(args)

def hyperparameter_optimization(args):
    hyperopt.run(args)


def parse_arguments(experiment, seed, argv):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    #parser.add_argument('dataset', type=str, help='Name of the dataset.')
    #parser.add_argument('exp', type=str,
    #                    choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
    #                             'Standard', 'Multitask', 'Joint', 'Probe',
    #                             'TTI', 'Robustness', 'HyperparameterSearch'],
    #                    help='Name of experiment to run.')
    #parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')

    if experiment == 'Probe':
        return (probe.parse_arguments(parser),)

    elif experiment == 'TTI':
        return (tti.parse_arguments(parser),)

    elif experiment == 'Robustness':
        return (gen_cub_synthetic.parse_arguments(parser),)

    elif experiment == 'HyperparameterSearch':
        return (hyperopt.parse_arguments(parser),)

    else:
        parser.add_argument('-log_dir', default='./outputs', help='where the trained model is saved')
        parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
        parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
        parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
        parser.add_argument('-lr', type=float, help="learning rate")
        parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
        parser.add_argument('-pretrained', '-p', action='store_true',
                            help='whether to load pretrained model & just fine-tune')
        parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
        parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
        parser.add_argument('-use_attr', action='store_true',
                            help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
        parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
        parser.add_argument('-no_img', action='store_true',
                            help='if included, only use attributes (and not raw imgs) for class prediction')
        parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
        parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                            help='Whether to use weighted loss for single attribute or multiple ones')
        parser.add_argument('-uncertain_labels', action='store_true',
                            help='whether to use (normalized) attribute certainties as labels')
        parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                            help='whether to apply bottlenecks to only a few attributes')
        parser.add_argument('-expand_dim', type=int, default=0,
                            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
        parser.add_argument('-n_class_attr', type=int, default=2,
                            help='whether attr prediction is a binary or triary classification')
        parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
        parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
        parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
        parser.add_argument('-end2end', action='store_true',
                            help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
        parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
        parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
        parser.add_argument('-scheduler_step', type=int, default=1000,
                            help='Number of steps before decaying current learning rate by half')
        parser.add_argument('-normalize_loss', action='store_true',
                            help='Whether to normalize loss by taking attr_loss_weight into account')
        parser.add_argument('-use_relu', action='store_true',
                            help='Whether to include relu activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-use_sigmoid', action='store_true',
                            help='Whether to include sigmoid activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-connect_CY', action='store_true',
                            help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')
        args = parser.parse_args(argv)
        args.three_class = (args.n_class_attr == 3)
        args.log_dir = os.path.join(args.log_dir, 'CUB'+str(seed), experiment + os.environ.get('EXEXP_LEVEL', ''))
        return (args,)
