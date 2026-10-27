import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import math
import time
import random
import shutil
import logging
import warnings
import argparse
import builtins
from transformers import BertModel, BertTokenizer
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from utils.model import PNCS
from utils.resnet import *
from utils.utils_algo import *
from utils.utils_loss import PNCSLoss
from utils.fmnist import load_fmnist
from utils.cifar10 import load_cifar10
from utils.cifar100 import load_cifar100



parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['fmnist', 'cifar10', 'cifar100'])

parser.add_argument('--exp-dir', default='./experiment/cifar10_time', type=str)

parser.add_argument('--data-dir', default='./data', type=str)

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')

parser.add_argument('-j', '--workers', default=0, type=int)

parser.add_argument('--epochs', default=500, type=int)

parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('-b', '--batch-size', default=128, type=int)

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)

parser.add_argument('-lr_decay_epochs', type=str, default='99,199,299')

parser.add_argument('-lr_decay_rate', type=float, default=0.1)

parser.add_argument('--cosine', action='store_true', default=True)

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default= 1, type=int)

parser.add_argument('--rank', default= 0, type=int,
                    help='node rank for distributed training')

parser.add_argument('--dist-url', default='tcp://localhost:10016', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('--seed', default=None, type=int)

parser.add_argument('--gpu', default=None, type=int)

parser.add_argument('--multiprocessing-distributed', default=True, action='store_true')

parser.add_argument('--cuda_VISIBLE_DEVICES', default='2', type=str)

parser.add_argument('--partial_rate', default=0.5, type=float,
                    help='ambiguity level (q)')

parser.add_argument('--mu', type=float)
parser.add_argument('--beta_amc', type=float)

parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')

parser.add_argument('--tau_proto', type=float, default=0.3,
                    help='temperature for prototype')

parser.add_argument('--alpha_mixup', type=float, default=8.0,
                    help='alpha for beta distribution')

parser.add_argument('--conf_th', type=float, default=0.5,
                    help='prototype update threshold')

parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')

parser.add_argument('--latent-dim', default=512, type=int,
                    help='latent embedding dimension')

parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')

parser.add_argument('--alpha_weight', default=1.0, type=float,
                    help='contrastive loss weight')

parser.add_argument('--pseudo_label_weight_range', default='0.95, 0.8', type=str,
                    help='pseudo target updating coefficient')

parser.add_argument('--pro_weight_range', default='0.9, 0.5', type=str,
                    help='prototype updating coefficient')

parser.add_argument('--hierarchical', action='store_true',
                    help='for CIFAR100-H training')

parser.add_argument('--n_embedding', help='the dim of label embedding', default=768, type=int)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_VISIBLE_DEVICES

torch.set_printoptions(precision=2, sci_mode=False)

logging.basicConfig(format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('./result/result_' + args.dataset + '_bs' + str(args.batch_size) + '_p' \
                            + str(args.partial_rate) + '_seed' + str(args.seed) + '_dis_' + args.dist_url[-5:] + '.log'),
        logging.StreamHandler()
])


def main():

    args.pseudo_label_weight_range = [float(item) for item in args.pseudo_label_weight_range.split(',')]
    args.pro_weight_range = [float(item) for item in args.pro_weight_range.split(',')]

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')
        print()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
        print()

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed


    model_path = 'ds_{ds}_lr_{lr}_ep_{ep}_aw_{aw}_arch_{arch}_pr_{pr}_sd_{seed}_dis_{dis}'.format(
                                            ds=args.dataset,
                                            lr=args.lr,
                                            ep=args.epochs,
                                            aw=args.alpha_weight,
                                            arch=args.arch,
                                            pr=args.partial_rate,
                                            seed=args.seed,
                                            dis=args.dist_url[-5:]
    )


    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    logging.info(args)
    print()

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):

    cudnn.benchmark = True

    args.gpu = gpu

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        print("Use GPU: {} for training\n".format(args.gpu))

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.gpu==0:
        print("=> creating model '{}'\n".format(args.arch))

    if args.dataset == 'fmnist':
        train_loader, train_partialY_matrix, train_sampler, test_loader, class_name = load_fmnist(args)

    elif args.dataset == 'cifar10':
        train_loader, train_partialY_matrix, train_sampler, test_loader, class_name = load_cifar10(args)

    elif args.dataset == 'cifar100':
        train_loader, train_partialY_matrix, train_sampler, test_loader, class_name = load_cifar100(args)


    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

    if args.gpu == 0:
        logging.info('\nAverage candidate num: {}\n'.format(train_partialY_matrix.sum(1).mean()))

    model = PNCS(args, PNCSNet, get_class_embedding(class_name))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    tempY = train_partialY_matrix.sum(dim=1, keepdim=True)
    uniform_confidence = train_partialY_matrix.float() / tempY
    uniform_confidence = uniform_confidence.cuda()

    PNCS_loss = PNCSLoss(predicted_score_cls=uniform_confidence, pseudo_label_weight=0.99)

    sim_criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()


    best_acc = 0


    for epoch in range(args.start_epoch, args.epochs):

        is_best = False

        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(args, optimizer, epoch)

        PNCS_loss.set_alpha(epoch, args)


        acc_train_cls, loss_cls_log, loss_PNCS_log, avg_batch_time = train(train_loader, model, PNCS_loss, optimizer, epoch, args, sim_criterion)


        PNCS_loss.set_pseudo_label_weight(epoch, args)
        model.module.set_prototype_update_weight(epoch, args)


        acc_test, _ = test(model, test_loader, args, epoch)


        if acc_test.item() > best_acc:
            best_acc = acc_test.item()

        if args.gpu==0:
            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write('Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}. (lr:{})\n'.format(\
                 epoch, acc_train_cls.avg, acc_test.item(), best_acc, optimizer.param_groups[0]['lr']))

            logging.info('Epoch {}: Train_Acc {}, Test_Acc {}, Best_Acc {}. (lr:{})\n'.format(\
                  epoch, acc_train_cls.avg, acc_test.item(), best_acc, optimizer.param_groups[0]['lr']))



def train(train_loader, model, PNCS_loss, optimizer, epoch, args, sim_criterion):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@cls', ':2.2f')
    loss_PNCS_log = AverageMeter('Loss@PNCS', ':2.2f')
    metric_updates = DeferredMeterUpdates(loss_cls_log, loss_PNCS_log, acc_cls)

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, loss_cls_log, loss_PNCS_log],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()

    end = time.time()

    for i, (images_1, images_2, labels, true_labels, index) in enumerate(train_loader):

        data_time.update(time.time() - end)

        X_1 = images_1.cuda(non_blocking=True)
        X_2 = images_2.cuda(non_blocking=True)
        Y = labels.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        Y_true = true_labels.long().cuda(non_blocking=True)

        Lambda = np.random.beta(args.alpha_mixup, args.alpha_mixup)

        # 单GPU原始
        idx_rp = torch.randperm(X_1.size(0), device=X_1.device)
        # 多GPU
        # idx_rp = torch.randperm(X_1.size(0))

        X_1_rp = X_1[idx_rp]
        X_2_rp = X_2[idx_rp]

        X_1_mix = Lambda * X_1 + (1 - Lambda) * X_1_rp
        X_2_mix = Lambda * X_2 + (1 - Lambda) * X_2_rp

        cls_out_1, cls_out_2, cls_out_mix1, cls_out_mix2,logits_prot_1, logits_prot_2, logits_prot_1_mix, logits_prot_2_mix = \
        model(img_q=X_1, img_k=X_2, img_q_mix=X_1_mix, img_k_mix=X_2_mix, partial_Y=Y, Y_true=Y_true, args=args, run_epoch=epoch)

        batch_size = cls_out_1.shape[0]

        PNCS_loss.update_weight_byclsout1(cls_predicted_score=cls_out_1, batch_index=index, batch_partial_Y=Y, args=args)

        cls_loss_1, ada_loss, kl_loss, alpha_td = PNCS_loss(cls_out_1,cls_out_mix1, cls_out_mix2, logits_prot_1, logits_prot_1_mix, logits_prot_2_mix, idx_rp, Lambda, index, args, sim_criterion, epoch)

        loss = cls_loss_1 + args.beta_amc * ada_loss + alpha_td * kl_loss

        acc = accuracy(cls_out_1, Y_true)
        metric_updates.update(cls_loss_1, loss, acc[0])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        log_batch = (i % 50 == 0) or ((i + 1) % len(train_loader) == 0)
        if log_batch:
            metric_updates.flush()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.gpu==0:
            if log_batch:
                logging.info('Epoch:[{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'A_cls {Acc_cls.val:.4f} ({Acc_cls.avg:.4f})\t'
                    'L_cls {Loss_cls.val:.4f} ({Loss_cls.avg:.4f})\t'
                    'L_all {Loss_PNCS.val:.4f} ({Loss_PNCS.avg:.4f})\t'.format(
                        epoch, i + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, Acc_cls=acc_cls,
                        Loss_cls=loss_cls_log, Loss_PNCS=loss_PNCS_log
                    )
                )

    metric_updates.flush()
    return acc_cls, loss_cls_log, loss_PNCS_log, batch_time.avg



def test(model, test_loader, args, epoch):
    with torch.inference_mode():

        print('\n=====> Evaluation...\n')
        model.eval()

        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")

        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs, outputs_pro = model(img_q=images, args=args, eval_only=True)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

            if args.gpu==0:
                if (batch_idx % 10 == 0) or ((batch_idx + 1) % len(test_loader) == 0):
                    logging.info(
                        'Test:[{0}/{1}]\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'\
                        .format(batch_idx + 1, len(test_loader), top1=top1_acc, top5=top5_acc)
                    )

        # average across all processes
        acc_tensors = torch.stack((top1_acc.avg.detach(), top5_acc.avg.detach()))

        dist.all_reduce(acc_tensors)

        acc_tensors /= args.world_size

        if args.gpu==0:
            logging.info('Top1 Accuracy is %.2f%%, Top5 Accuracy is %.2f%%\n'%(acc_tensors[0], acc_tensors[1]))

    return acc_tensors[0], acc_tensors[1]

def get_class_embedding(class_name=[]):
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    model = BertModel.from_pretrained('./bert-base-uncased')
    model.eval()

    all_words = []
    label_word_ranges = []
    for label in class_name:
        start = len(all_words)
        all_words.extend(label.split(' ')) # a label with multiple words
        label_word_ranges.append((start, len(all_words)))

    if len(all_words) == 0:
        return torch.empty(0, args.n_embedding).cuda()

    with torch.no_grad():
        text_dict = tokenizer(
            all_words,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            return_tensors='pt'
        )
        res = model(
            input_ids=text_dict['input_ids'],
            attention_mask=text_dict['attention_mask'],
            token_type_ids=text_dict['token_type_ids']
        )
        word_vectors = res[0][:, 1, :].detach()

    all_class_embedding = []
    for start, end in label_word_ranges:
        all_class_embedding.append(word_vectors[start:end].sum(dim=0))

    class_embedding = torch.stack(all_class_embedding).cuda()
    class_embedding = torch.nn.functional.normalize(class_embedding, dim=1).float()
    return class_embedding

if __name__ == '__main__':
    main()

