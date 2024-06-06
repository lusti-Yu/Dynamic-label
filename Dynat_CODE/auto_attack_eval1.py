from __future__ import print_function

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

from torchvision import datasets, transforms
from models.wideresnet import *
from autoattack_modified.autoattack import AutoAttack


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./',
                    help='source model for black-box attack evaluation')
parser.add_argument('--source2-model-path',
                    default='./',
                    help='source model for black-box attack evaluation')

parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--mark', default=None, type=str,
                            help='log file name')
parser.add_argument('--widen_factor', default=None, type=int,
                            help='widen_factor for wideresnet')
parser.add_argument('--num_classes', default=10, type=int,
                            help='cifar10 or cifar100')
parser.add_argument('--dataparallel', default=False, type=bool,
                            help='whether model is trained with dataparallel')

parser.add_argument('-teacher_model', type=str)

parser.add_argument('-teacher', type=str)


args = parser.parse_args()

if args.epsilon == 8.0:
    args.epsilon = 8.0 / 255

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)

def normalize(X):
    return (X - mu) / std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])

if args.num_classes == 100:
   testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test)
   test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
else:
   testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
   test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, model2,t_model,t_model2,attack_iters, restarts, epsilon=(8 / 255.) / std):
    print(epsilon)
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)


    return pgd_loss / n, pgd_acc / n

def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


def _pgd_whitebox(model,
                  X,
                  y,
                  adversary,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = adversary.run_standard_evaluation(X, y, bs=X.size(0))
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd


def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def evaluate_pgd_cw(test_loader, model, attack_iters, restarts):
    alpha = (2 / 255.) / std
    epsilon = (8 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters=attack_iters, restarts=restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n

def eval_adv_test_whitebox(model, device, test_loader, adverary):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, adverary)
        robust_err_total += err_robust
        natural_err_total += err_natural

    open("Logs/"+args.mark,"a+").write("robust_err_total: "+str(robust_err_total)+"\n")
    open("Logs/"+args.mark,"a+").write("natural_err_total: "+str(natural_err_total)+"\n")


def attack_fgsm(model, X, y, epsilon, alpha, restarts):
    attack_iters = 1
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_fgsm(test_loader, model, restarts):
    epsilon = (8 / 255.) / std
    alpha = (8 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_fgsm(model, X, y, epsilon, alpha, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n



def main():

    if args.white_box_attack:
        # white-box attack
        open("Logs/"+args.mark,"a+").write('pgd white-box attack\n')
        model = WideResNet(num_classes=args.num_classes, widen_factor=args.widen_factor).cuda()

        if args.dataparallel:
           model = nn.DataParallel(model).to(device)
        else:
           model = model.to(device)



        # print(model)
        model.load_state_dict(torch.load(args.model_path), strict=False)

        epsilon = args.epsilon
        # epsilon=float(epsilon)/255.
        print(epsilon)



        model.eval()

        #
        # AT_models_test_loss, AT_models_test_acc = evaluate_standard(test_loader5, model,model2,t_model,t_model2)
        # print('AT_models_test_acc:', AT_models_test_acc)
        # AT_fgsm_loss, AT_fgsm_acc = evaluate_fgsm(test_loader, model, 1)
        # print('AT_fgsm_acc:', AT_fgsm_acc)
        # AT_pgd_loss_10, AT_pgd_acc_10 = evaluate_pgd(test_loader, model, 10, 1, epsilon=(8 / 255.) / std)
        # print('AT_models_test_acc:', AT_pgd_acc_10)
        # AT_pgd_loss_20, AT_pgd_acc_20 = evaluate_pgd(test_loader, model, 20, 1, epsilon=(8 / 255.) / std)
        # print('AT_pgd_acc_20:', AT_pgd_acc_20)
        # AT_pgd_loss_50, AT_pgd_acc_50 = evaluate_pgd(test_loader, model, 50, 1, epsilon=(8 / 255.) / std)
        # print('AT_pgd_acc_50:', AT_pgd_acc_50)
        # AT_CW_loss_20, AT_pgd_cw_acc_20 = evaluate_pgd_cw(test_loader, model, 20, 1)
        # print('AT_pgd_cw_acc_20:', AT_pgd_cw_acc_20)
        # # #


        # adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard', log_path = "Logs/"+args.mark)
        adversary1 = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard', log_path = "Logs/"+args.mark)
        # l = [x for (x, y) in test_loader]
        # x_test = torch.cat(l, 0)
        # l = [y for (x, y) in test_loader]
        # y_test = torch.cat(l, 0)
        x_test = torch.cat([x for (x, y) in test_loader], 0)
        y_test = torch.cat([y for (x, y) in test_loader], 0)
        adv_complete = adversary1.run_standard_evaluation(x_test[:1000], y_test[:1000],
                        bs=128)
        #
        print(adv_complete)
        # adversary1.run_standard_evaluation(x_test[:10000], y_test[:10000],
        #                 bs=128).to(torch.device('cuda'))
        # adversary1.seed = 0
        # eval_adv_test_whitebox(model, device, test_loader, adversary)
    

if __name__ == '__main__':
    main()
