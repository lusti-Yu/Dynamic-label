import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def dynat_loss(model, model_teacher,
                x_natural,
                y,
                epoch,
                optimizer,
                awp_adversary,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    mse=torch.nn.MSELoss()
    ce=torch.nn.CrossEntropyLoss()
    softmax=torch.nn.Softmax(dim=1)

    model.eval()

    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # print(x_adv.shape)
                # print(x_natural.shape)
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
                #
                # out_adv = model(x_adv)
                # out = model_teacher(x_natural)
                # train_label = out.argmax(dim=1)
                # loss_kl = ce(out_adv, train_label)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            for idx_batch in range(batch_size):
                grad_idx = grad[idx_batch]
                grad_idx_norm = l2_norm(grad_idx)
                grad_idx /= (grad_idx_norm + 1e-8)
                x_adv[idx_batch] = x_adv[idx_batch].detach() + step_size * grad_idx
                eta_x_adv = x_adv[idx_batch] - x_natural[idx_batch]
                norm_eta = l2_norm(eta_x_adv)
                if norm_eta > epsilon:
                    eta_x_adv = eta_x_adv * epsilon / l2_norm(eta_x_adv)
                x_adv[idx_batch] = x_natural[idx_batch] + eta_x_adv
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    if epoch >= 10:
        awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                     inputs_clean=x_natural,
                                     targets=y,
                                     beta=6)
        awp_adversary.perturb(awp)

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    out_adv=model(x_adv)
    out_natural=model(x_natural)
    out=model_teacher(x_natural)

    # train_label = out.argmax(dim=1)
    # loss_ce = ce(out_adv, train_label)
    def kl_loss(a, b):
        loss = -a * b + torch.log(b + 1e-5) * b
        return loss

    kl_Loss1 = kl_loss(F.log_softmax(out, dim=1),
                       F.softmax(out_natural, dim=1))
    kl_Loss2 = kl_loss(F.log_softmax(out_natural, dim=1),
                       F.softmax(out, dim=1))

    kl_Loss1 = torch.mean(kl_Loss1)
    kl_Loss2 = torch.mean(kl_Loss2)




    kl_Loss3 = kl_loss(F.log_softmax(out_adv, dim=1),
                       F.softmax(out_natural.detach(), dim=1))
    kl_Loss4 = kl_loss(F.log_softmax(out_natural, dim=1),
                       F.softmax(out_adv, dim=1))


    kl_Loss5 = kl_loss(F.log_softmax(out_adv, dim=1),
                       F.softmax(out_natural.detach(), dim=1))

    kl_Loss3 = torch.mean(kl_Loss3)
    kl_Loss4 = torch.mean(kl_Loss4)

    loss_2klloss = 10*(out.size(0)) * torch.abs(kl_Loss3 + kl_Loss4)
    # print('kl_Loss1:', kl_Loss1)
    # print('kl_Loss2:', kl_Loss2)
    # print('loss_klloss', loss_klloss)
    # print('kl_Loss3:', kl_Loss3)
    # print('kl_Loss4:', kl_Loss4)
    # print('loss_2klloss:', loss_2klloss)
    #
    # lossmse = mse(out_adv, out)
    #
    # print('lossmse:', lossmse)

    softout = F.softmax(out, dim=1)
    softadv= F.softmax(out_adv, dim=1)


    max_out, predicted_classes = torch.max(softout, dim=1)


    max_adv, predicted_adv = torch.max(softadv, dim=1)

    average_adv= torch.mean(max_adv)

    # print(average_adv)

    average_value = torch.mean(max_out)

    out = model_teacher(x_natural)
    train_label = out.argmax(dim=1)
    # dot_product = torch.dot(train_label.float, max_out)
    # print("Dot Product:", dot_product)
    # max_adv, predicted_classes = torch.max(softadv, dim=1)

    # print(out)
    loss_klloss = torch.abs(kl_Loss1 - kl_Loss2)



    # if loss_klloss > 0.001:
    #     loss_klloss = 0

    loss_mse = ce(out, y) + 0.2*ce(out_adv,train_label)

    # print(loss_mse)

    # loss_mse = ce(out,y) + mse(out_adv, out)+ loss_klloss

    # print('mse(out_natural.detach(), out):',mse(out_natural.detach(), out))
    #
    # print('mse(out_adv, out):', mse(out_adv, out))
    # print('loss_mse:', loss_mse)

    # print(out_natural)
    # print(out)
    #
    #
    # print(loss_mse)
    #
    # print(mse(out,out_natural))
    #
    # loss_mse = 5*average_value*ce(out, y) + 5*average_adv*ce(out_adv, train_label)

    # print(average_adv)
    # print(average_value)

    # print(ce(out, y))
    # print(mse(out_adv, out))
    # print(10 *loss_klloss)


    loss_test = mse(out_adv, out)
    loss_kl = (1.0 / out.size(0)) * criterion_kl(F.log_softmax(out_adv, dim=1),F.softmax(out_natural, dim=1))



    loss = loss_mse + loss_kl

    # print('loss_2klloss:',loss)

    # print('loss_test:', loss_test)

    loss.backward()
    optimizer.step()
    #
    if epoch >= 10:
        awp_adversary.restore(awp)

    return loss
