import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.adamw import AdamW


class TriangularScheduler(_LRScheduler):
    '''
    确保输入最后的参数为BERT外参数，不采用预热
    '''
    def __init__(self, optimizer, cut_frac, T, ratio):
        self.cut_frac = cut_frac
        self.T = T
        self.ratio = ratio
        self.cut = self.T * self.cut_frac
        self.finished = False
        print('修改预热策略，这里只对bert参数进行预热，其他参数使用正常策略')
        super(TriangularScheduler, self).__init__(optimizer)

    def get_lr(self):
        iteration_step = self.last_epoch % self.T
        if iteration_step <= self.cut:
            p = iteration_step / self.cut
        else:
            p = 1 - (iteration_step - self.cut) / (self.cut * (1 / self.cut_frac - 1))
        gamma = (1 + p * (self.ratio - 1)) / self.ratio
        # print(self.base_lrs)

        # return [base_lr * gamma
        #         for base_lr in self.base_lrs]

        final_lrs = [base_lr * gamma for base_lr in self.base_lrs[:-1]] + [self.base_lrs[-1]]
        return final_lrs



if __name__ == '__main__':
    # model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)),
    #          torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    lr = 0.1
    model = [
        {'params': torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)), 'lr': lr},
        {'params': torch.nn.Parameter(torch.randn(2, 2, requires_grad=True)), 'lr': lr * 10}
    ]
    optim = AdamW(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    scheduler_warmup = TriangularScheduler(optim, cut_frac=0.1, T=100, ratio=32)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()

    for epoch in range(1, 20):
        for idx in range(1, 10):
            scheduler_warmup.step()
            # scheduler_steplr.step(epoch)
            # print(optim.param_groups[0]['lr'])
            # print(optim.param_groups)

            optim.step()  # backward pass (update network)
