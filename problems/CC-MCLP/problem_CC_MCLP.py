from PIL.PngImagePlugin import is_cid
from torch.utils.data import Dataset
import torch
import os
import pickle
import math
import  numpy as np
from problems.MCLP.state_MCLP import StateMCLP



class MCLP(object):
    NAME = 'MCLP'
    @staticmethod
    def get_total_num(dataset, pi):

        # 1) 权重（若提供 dataset['params'] 覆盖）
        params = dataset.get('params', {}) if isinstance(dataset, dict) else {}
        w1 = float(params.get('w1', 0.4))
        w2 = float(params.get('w2', 0.25))
        w3 = float(params.get('w3', 0.1))
        w4 = float(params.get('w4', 0.25))

        # 2) 解包张量
        users = dataset['users']  # [batch, n_user, 2]
        facilities = dataset['facilities']  # [batch, n_facilities, 2]
        demand = dataset['demand']  # [batch, n_user, 1]
        radius = dataset['r'][0]  # 标量

        batch_size, n_user, _ = users.size()
        _, n_facilities, _ = facilities.size()
        _, p = pi.size()

        # 可选项：成本/集聚/竞争（形状优先 [batch, n_facilities]，也兼容 [n_facilities]）
        def to_batched_facility_tensor(key_names):
            for key in key_names:
                if key in dataset:
                    t = dataset[key]
                    if t.dim() == 1:
                        t = t.unsqueeze(0).expand(batch_size, -1)
                    return t
            return torch.zeros((batch_size, n_facilities), dtype=facilities.dtype, device=facilities.device)

        rent = to_batched_facility_tensor(['rent', 'rents', 'cost', 'costs'])  # 设施成本
        B = to_batched_facility_tensor(['B', 'agg', 'agglomeration'])  # 集聚收益
        C = to_batched_facility_tensor(['C', 'comp', 'competition'])  # 竞争成本

        # 3) 覆盖判定：仅在被选设施上判断（隐式满足 Y_ij <= X_j）
        # dist_all: [batch, n_facilities, n_user]
        dist_all = (facilities[:, :, None, :2] - users[:, None, :, :]).norm(p=2, dim=-1)
        pi_expanded = pi.unsqueeze(-1).expand(-1, -1, n_user)  # [batch, p, n_user]
        selected_dist = dist_all.gather(1, pi_expanded)  # [batch, p, n_user]

        # 唯一服务：y_i = 1 若存在任一被选设施覆盖（<= R）
        covered_any = (selected_dist <= radius).any(dim=1).float()  # [batch, n_user]
        w_i = demand.squeeze(-1)  # [batch, n_user]
        coverage_term = (w_i * covered_any).sum(dim=1)  # [batch]

        # 4) X 项的汇总：通过从 [batch, n_facilities] 在 pi 上 gather 然后求和
        selected_rent = rent.gather(1, pi).sum(dim=1)  # [batch]
        selected_B = B.gather(1, pi).sum(dim=1)  # [batch]
        selected_C = C.gather(1, pi).sum(dim=1)  # [batch]

        # 5) 组合目标
        objective = (
                w1 * coverage_term +
                w3 * selected_B -
                w4 * selected_C -
                w2 * selected_rent
        )

        return objective
    @staticmethod
    def make_dataset(*args, **kwargs):
        return MCLPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMCLP.initialize(*args, **kwargs)


class MCLPDataset(Dataset):
    def __init__(self, filename=None, n_users=50, n_facilities=20, num_samples=5000, offset=0, p=8, r=0.2, distribution=None):
        super(MCLPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [row for row in (data[offset:offset + num_samples])]
                p = self.data[0]['p']
                r = self.data[0]['r']
        else:
            # Sample points randomly in [0, 1] square
            self.data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                              facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                              demand=torch.FloatTensor(n_users, 1).uniform_(1, 10),
                              p=p,
                              r=r)
                         for i in range(num_samples)]

        self.size = len(self.data)
        self.p = p
        self.r = r

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]