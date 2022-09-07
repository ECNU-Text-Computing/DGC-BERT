import argparse
import datetime
import numpy as np
import pandas as pd

from torch import nn
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, \
    precision_recall_curve, auc
from scripts.warmup import TriangularScheduler


class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, pad_size=1500, word2vec=None, keep_prob=0.5,
                 model_path=None, **kwargs):
        super(BaseModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not model_path:
            if word2vec is not None:
                self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False)
            else:
                self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False, padding_idx=pad_index)
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.pad_size = pad_size
        self.model_name = 'base_model'
        self.adaptive_lr = False
        self.warmup = False
        self.T = kwargs['T'] if 'T' in kwargs.keys() else 400
        self.seed = None
        if 'seed' in kwargs.keys():
            self.seed = kwargs['seed']
        if self.warmup:
            print('warm up T', self.T)

    def init_weights(self):
        print('init')

    def forward(self, text, lengths, masks, **kwargs):
        return 'forward'

    def train_model(self, dataloader, epoch, criterion, optimizer, epoch_log):
        """
        single epoch train
        """
        self.train()
        total_acc, total_count = 0, 0
        log_interval = 10
        start_time = time.time()
        loss_list = []
        all_predicted_result = []
        all_true_label = []

        for idx, (contents, labels, lengths, masks, indexes) in enumerate(dataloader):
            # if idx == 0:
            #     # print('contents', contents.shape)
            #     # print('labels', labels.shape)
            #     # print('lengths', lengths.shape)
            #     # print('masks', masks.shape)
            #     print(contents)
            # data trans
            start_time = time.time()
            if type(contents) == list:
                contents = [content.to(self.device) for content in contents]
            else:
                contents = contents.to(self.device)
            # print('io-time:', time.time()-start_time)
            labels = labels.to(self.device)
            # lengths = lengths.to(self.device)
            masks = masks.to(self.device)
            # print('io-time:', time.time() - start_time)

            optimizer.zero_grad()
            predicted_result = self(contents, lengths, masks)
            # print('forward-time:', time.time() - start_time)
            loss = criterion(predicted_result, labels.long())
            # print('loss-time:', time.time() - start_time)
            loss.backward()
            # print('backforward-time:', time.time() - start_time)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            if self.warmup:
                self.scheduler.step()
            optimizer.step()
            # print('optimizer-time:', time.time() - start_time)
            all_predicted_result += F.softmax(predicted_result, dim=1).detach().cpu().numpy().tolist()
            all_true_label += labels.cpu().numpy().tolist()
            total_acc += (predicted_result.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            loss_list.append(loss.item())
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                                                 total_acc / total_count, loss.item()))
                epoch_log.write('| epoch {:3d} | {:5d}/{:5d} batches '
                                '| accuracy {:8.3f} | loss {:8.3f}\n'.format(epoch, idx, len(dataloader),
                                                                             total_acc / total_count, loss.item()))
                total_acc, total_count = 0, 0
                start_time = time.time()
            # print('val-time:', time.time() - start_time)

        avg_loss = np.mean(loss_list)
        all_predicted_result = np.array(all_predicted_result)
        train_results = [avg_loss] + cal_metrics(all_true_label, all_predicted_result)

        train_str = '-' * 59 + '\n' + \
                    get_format_str(train_results, 'train') + \
                    '-' * 59
        print(train_str)
        if epoch_log:
            epoch_log.write(train_str)

        return train_results

    def train_batch(self, dataloaders, epochs, lr=10e-4, criterion='CrossEntropyLoss', optimizer='ADAM',
                    scheduler=False, record_path=None, save_path=None):
        """
        batch training
        """
        final_results = []
        train_dataloader, val_dataloader, test_dataloader = dataloaders
        total_accu = None
        val_accu_list = []
        if criterion == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss()
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=1e-3)
        elif optimizer == 'ADAM':
            if self.adaptive_lr:
                optimizer = torch.optim.Adam(self.get_group_parameters(lr), lr=lr, weight_decay=0)
            else:
                optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-3)
                # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
                # print('change the weight decay to 5e-4!!!!')
        elif optimizer == 'SPARSE_ADAM':
            optimizer = torch.optim.SparseAdam(self.parameters(), lr=lr)
        elif optimizer == 'ADAMW':
            if self.adaptive_lr:
                optimizer = torch.optim.AdamW(self.get_group_parameters(lr), lr=lr)
            else:
                optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        if self.warmup:
            self.scheduler = TriangularScheduler(optimizer, cut_frac=0.1, T=self.T, ratio=32)
            optimizer.zero_grad()
            optimizer.step()

        if self.seed:
            print('{}_records_{}.csv'.format(self.model_name, self.seed))
            fw = open(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), 'w')
        else:
            fw = open(record_path + '{}_records.csv'.format(self.model_name), 'w')
        fw.write('epoch,'
                 'loss_train,accu_train,roc_auc_train,log_loss_train,prec_train,recall_train,f1_train,pr_auc_train,'
                 'prec_neg_train,recall_neg_train,f1_neg_train,pr_auc_neg_train,'
                 'loss_val,accu_val,roc_auc_val,log_loss_val,prec_val,recall_val,f1_val,pr_auc_val,'
                 'prec_neg_val,recall_neg_val,f1_neg_val,pr_auc_neg_val,'
                 'loss_test,accu_test,roc_auc_test,log_loss_test,prec_test,recall_test,f1_test,pr_auc_test,'
                 'prec_neg_test,recall_neg_test,f1_neg_test,pr_auc_neg_test\n')
        for epoch in range(1, epochs + 1):
            epoch_log = open(record_path + '{}_epoch_{}.log'.format(self.model_name, epoch), 'w')
            epoch_start_time = time.time()
            # train(model, train_dataloader)
            train_results = self.train_model(train_dataloader, epoch, self.criterion, optimizer, epoch_log)
            val_results = self.evaluate(val_dataloader)
            # acc, prec, recall, maf1, f1, auc, log_loss_value = val_results
            # val_accu_list.append(round(acc, 3))
            acc = val_results[1]
            if save_path:
                self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, epoch))

            if scheduler:
                if total_accu is not None and total_accu > acc:  # 总准确率大于验证准确率时调整学习率
                    scheduler.step()
            else:
                total_accu = acc

            val_str = '-' * 59 + '\n' \
                      + '| end of val | time: {:5.2f}s |\n'.format(time.time() - epoch_start_time) \
                      + get_format_str(val_results, 'val') \
                      + '-' * 59
            print(val_str)
            if epoch_log:
                epoch_log.write(val_str)
            # print('-' * 59)

            test_results = self.test(test_dataloader, epoch_log=epoch_log)
            epoch_log.close()
            all_results = ['{:5d}'.format(epoch)] + ['{:.5f}'.format(val) for val in train_results] + \
                          ['{:.5f}'.format(val) for val in val_results] + ['{:.5f}'.format(val) for val in test_results]
            final_results.append(all_results)
            # print(len(all_results))
            fw.write(','.join(all_results) + '\n')
        fw.close()
        return final_results

    def save_model(self, path):
        # torch.save(self, path)
        torch.save({'state_dict': self.state_dict()}, path)
        print('Save successfully!')

    def load_model(self, path):
        # model = torch.load(path)
        state_dict = torch.load(path)['state_dict']
        model = self.load_state_dict(state_dict)
        print('Load successfully!')
        return model

    def evaluate(self, dataloader, phase='train'):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.eval()
        all_predicted_result = []
        all_true_label = []

        with torch.no_grad():
            loss_list = []
            for idx, (contents, labels, lengths, masks, indexes) in enumerate(dataloader):
                if type(contents) == list:
                    contents = [content.to(self.device) for content in contents]
                else:
                    contents = contents.to(self.device)
                labels = labels.to(self.device)
                # lengths = lengths.to(self.device)
                masks = masks.to(self.device)

                model_result = self(contents, lengths, masks)

                if type(model_result) == list:
                    model_result = model_result[0]
                loss = self.criterion(model_result, labels.long())  # 计算损失函数
                loss_list.append(loss.item())

                predicted_result = F.softmax(model_result, dim=1).detach().cpu().numpy()
                true_label = labels.cpu().numpy().tolist()

                all_predicted_result += predicted_result.tolist()
                all_true_label += true_label

            all_predicted_result = np.array(all_predicted_result)
            all_predicted_label = all_predicted_result.argmax(1)
            avg_loss = np.mean(loss_list)

        if phase == 'benchmark':
            print(all_predicted_label)
            print(all_true_label)
        results = [avg_loss] + cal_metrics(all_true_label, all_predicted_result)

        return results

    def test(self, test_dataloader, phase='test', epoch_log=None):
        # print('-' * 59)
        test_start_time = time.time()
        all_results = self.evaluate(test_dataloader, phase)
        test_str = '-' * 59 + '\n' \
                  + '| end of test | time: {:5.2f}s |\n'.format(time.time() - test_start_time) \
                  + get_format_str(all_results, phase) \
                  + '-' * 59

        print(test_str)
        if epoch_log:
            epoch_log.write(test_str)

        return all_results

    def get_mistake_results(self, test_dataloader):
        self.test(test_dataloader)
        self.eval()
        mistake_results = []

        with torch.no_grad():
            for idx, (contents, labels, lengths, masks, indexes) in enumerate(test_dataloader):
                if type(contents) == list:
                    contents = [content.to(self.device) for content in contents]
                else:
                    contents = contents.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                masks = masks.to(self.device)

                model_result = self(contents, lengths, masks)
                if type(model_result) == list:
                    model_result = model_result[0]
                predicted_result = F.softmax(model_result, dim=1).detach().cpu().numpy()
                predicted_label = np.array(predicted_result).argmax(1)
                true_label = labels.cpu().numpy().tolist()
                result = zip(predicted_result, predicted_label, true_label, indexes.cpu().numpy().tolist(),
                             (predicted_label == true_label))
                mistake_results.extend(list(filter(lambda x: x[-1] == False, result)))

        return mistake_results


def cal_metrics(all_true_label, all_predicted_result):
    all_predicted_label = np.argmax(all_predicted_result, axis=1)
    # print(len(all_true_label))
    # print(all_true_label)
    # print(all_predicted_label)
    # all
    acc = accuracy_score(all_true_label, all_predicted_label)
    roc_auc = roc_auc_score(all_true_label, all_predicted_result[:, 1])
    log_loss_value = log_loss(all_true_label, all_predicted_result)
    # pos
    prec = precision_score(all_true_label, all_predicted_label)
    recall = recall_score(all_true_label, all_predicted_label)
    f1 = f1_score(all_true_label, all_predicted_label, average='binary')
    pr_p, pr_r, _ = precision_recall_curve(all_true_label, all_predicted_result[:, 1])
    pr_auc = auc(pr_r, pr_p)
    # neg
    prec_neg = precision_score(all_true_label, all_predicted_label, pos_label=0)
    recall_neg = recall_score(all_true_label, all_predicted_label, pos_label=0)
    f1_neg = f1_score(all_true_label, all_predicted_label, average='binary', pos_label=0)
    pr_np, pr_nr, _ = precision_recall_curve(all_true_label, all_predicted_result[:, 0], pos_label=0)
    pr_auc_neg = auc(pr_nr, pr_np)

    results = [acc, roc_auc, log_loss_value, prec, recall, f1, pr_auc, prec_neg, recall_neg, f1_neg, pr_auc_neg]
    return results


def get_format_str(results, phase):
    avg_loss, acc, roc_auc, log_loss_value, prec, recall, f1, pr_auc, prec_neg, recall_neg, f1_neg, pr_auc_neg = results

    format_str = '| avg loss {:8.3f} | {:9} {:7.3f} |\n' \
                 '| roc_auc {:9.3f} | log_loss {:8.3f} |\n' \
                 'pos:\n' \
                 '| precision {:7.3f} | recall {:10.3f} |\n' \
                 '| f1 {:14.3f} | pr_auc {:10.3f} |\n' \
                 'neg:\n' \
                 '| precision {:7.3f} | recall {:10.3f} |\n' \
                 '| f1 {:14.3f} | pr_auc {:10.3f} |\n'.format(avg_loss, phase + ' acc', acc, roc_auc, log_loss_value,
                                                              prec, recall, f1,
                                                              pr_auc, prec_neg, recall_neg, f1_neg, pr_auc_neg)
    return format_str


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done base_model!')
