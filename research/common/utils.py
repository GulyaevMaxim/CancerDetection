from torch.autograd import Variable
import torch
import torch.nn
from tqdm import tqdm
import tensorboardX
from sklearn.metrics import roc_auc_score
import pandas
import numpy


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


def roc_auc(output, target):
    y_true = target.numpy()
    y_pred = output.numpy()
    return roc_auc_score(y_true, y_pred)


def train(model, writer,
          use_cuda, train_dloader,
          L, optimizer, epoch):
    train_loss = 0.0
    model.train()

    pbar = tqdm(train_dloader)
    acc = 0
    roc_auc_acc = 0

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

        loss = L(y_predicted[:, 0], target.float())

        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        train_loss += loss.item() / batch_size

        batch_acc = accuracy(y_predicted, target, topk=(1, ))
        acc += batch_acc[0]

        try:
            batch_roc_acc = roc_auc(y_predicted[0:].detach().cpu(), target.cpu())
            roc_auc_acc += batch_roc_acc
        except Exception as e:
            pass

        pbar.set_description('Loss {}'.format(
            loss.cpu().data.numpy() / batch_size))

        del loss, data, target, y_predicted

    print('Accuracy {}'.format(acc[0] / len(train_dloader)))
    writer.add_scalars('data/train_accuracy',
                       {'train_accuracy': acc[0] / len(train_dloader)},
                       epoch)

    print('ROC_AUC Accuracy {}'.format(roc_auc_acc / len(train_dloader)))
    writer.add_scalars('data/roc_train_accuracy',
                       {'roc_train_accuracy': roc_auc_acc / len(train_dloader)},
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
    roc_auc_acc = 0

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

            loss = L(y_predicted[:, 0], target.float())
            batch_size = target.size(0)
            valid_loss += loss.item() / batch_size

            pbar.set_description('Loss {}'.format(loss))

            batch_acc = accuracy(y_predicted, target, topk=(1,))
            acc += batch_acc[0]
            try:
                batch_roc_acc = roc_auc(y_predicted[0:].detach().cpu(), target.cpu())
                roc_auc_acc += batch_roc_acc
            except Exception as e:
                pass

    print('Loss_validate {}'.format(valid_loss / len(valid_dloader)))
    writer.add_scalars('data/validate_loss_avg',
                       {'loss': valid_loss / len(valid_dloader)},
                       epoch)

    print('ROC_AUC Accuracy {}'.format(roc_auc_acc / len(valid_dloader)))
    writer.add_scalars('data/roc_validate_accuracy',
                       {'roc_validate_accuracy': roc_auc_acc / len(valid_dloader)},
                       epoch)

    print('Accuracy_validate {}'.format(acc[0] / len(valid_dloader)))
    writer.add_scalars('data/validate_accuracy',
                       {'accuracy': acc[0] / len(valid_dloader)}, epoch)

    return roc_auc_acc / len(valid_dloader)


def generate_submission(id, labels, path):
    df = pandas.DataFrame({'id': id, 'label': labels})
    df.to_csv('{}.gz'.format(path), index=False,
              compression='gzip')