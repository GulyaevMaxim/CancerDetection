from torch.autograd import Variable
import torch
import torch.nn
from tqdm import tqdm
import tensorboardX


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def train(model, writer,
          use_cuda, train_dloader,
          L, optimizer, epoch):
    train_loss = 0.0
    model.train()

    pbar = tqdm(train_dloader)
    acc = 0

    for batch_idx, data in enumerate(pbar):
        if use_cuda:
            data, target = Variable(data[0].cuda(),
                                    requires_grad=False), \
                           Variable(data[1].cuda(),
                                    requires_grad=False)
        else:
            data, target = Variable(data[0],
                                    requires_grad=False), \
                           Variable(data[1],
                                    requires_grad=False)

        optimizer.zero_grad()
        y_predicted = model(data)

        loss = L(y_predicted, target)

        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        train_loss += loss.item() / batch_size

        batch_acc = accuracy(y_predicted, target, topk=(1, ))
        acc += batch_acc[0]

        pbar.set_description('Loss {}'.format(
            loss.cpu().data.numpy() / batch_size))

        del loss, data, target, y_predicted

    print('Accuracy {}'.format(acc[0] / len(train_dloader)))
    writer.add_scalars('data/train_accuracy',
                       {'train_accuracy': acc[0] / len(train_dloader)},
                       epoch)

    print('Loss {}'.format(train_loss / len(train_dloader)))
    writer.add_scalars('data/train_loss_by_epoch',
                       {'loss_epoch': train_loss / len(train_dloader)},
                       epoch)


def validate(model, writer,
          use_cuda, valid_dloader,
          L, epoch):

    valid_loss = 0
    acc = 0
    model.eval()
    pbar = tqdm(valid_dloader)
    with torch.no_grad():
        for batch_idx, data in enumerate(pbar):
            if use_cuda:
                data, target = Variable(data[0].cuda(),
                                        requires_grad=False), \
                               Variable(data[1].cuda(),
                                        requires_grad=False)
            else:
                data, target = Variable(data[0],
                                        requires_grad=False), \
                               Variable(data[1],
                                        requires_grad=False)

            y_predicted = model(data)

            loss = L(y_predicted, target)
            batch_size = target.size(0)
            valid_loss += loss.item() / batch_size

            pbar.set_description('Loss {}'.format(loss))

            batch_acc = accuracy(y_predicted, target, topk=(1,))
            acc += batch_acc[0]

    print('Loss_validate {}'.format(valid_loss / len(valid_dloader)))
    writer.add_scalars('data/validate_loss_avg',
                       {'loss': valid_loss / len(valid_dloader)},
                       epoch)

    print('Accuracy_validate {}'.format(acc[0] / len(valid_dloader)))
    writer.add_scalars('data/validate_accuracy',
                       {'accuracy': acc[0] / len(valid_dloader)}, epoch)

    return acc / len(valid_dloader)
