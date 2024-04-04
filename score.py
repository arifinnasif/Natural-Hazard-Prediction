import torch
import numpy as np
import datetime
import os
import config as cfg

class Cal_params_epoch(object):
    def __init__(self):
        # n1->TP  n2->FP  n3->FN  n4->TN
        self.n1 = 0
        self.n2 = 0
        self.n3 = 0
        self.n4 = 0
        self.n1sum = 0
        self.n2sum = 0
        self.n3sum = 0
        self.n4sum = 0
        self.n1_neigh = 0
        self.n2_neigh = 0
        self.n3_neigh = 0
        self.n4_neigh = 0
        self.n1sum_neigh = 0
        self.n2sum_neigh = 0
        self.n3sum_neigh = 0
        self.n4sum_neigh = 0
        self.eps = 1e-10

    def _transform_sum(self, y_true, y_pred, do_sigmoid=True):
        y_true = y_true.permute(1, 0, 2, 3, 4).cpu().contiguous()
        # y_true = y_true.permute(1, 0, 2, 3).cpu().contiguous()
        y_pred = y_pred.permute(1, 0, 2, 3, 4).cpu().contiguous()
        if do_sigmoid:
            y_pred = torch.round(torch.sigmoid(y_pred))
        else:
            y_pred = torch.round(y_pred)
        frames = y_true.shape[0]
        sum_true = torch.zeros(y_true[0].shape)
        sum_pred = torch.zeros(y_pred[0].shape)
        for i in range(frames):
            sum_true += y_true[i]
            sum_pred += y_pred[i]
        sum_true = torch.flatten(sum_true)
        sum_pred = torch.flatten(sum_pred)
        return sum_true, sum_pred

    def _transform(self, y_true, y_pred, do_sigmoid=True):
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        y_true = torch.flatten(y_true)
        if do_sigmoid:
            y_pred = torch.flatten(torch.sigmoid(y_pred))
        else:
            y_pred = torch.flatten(y_pred)
        y_pred = torch.round(y_pred)
        return y_true, y_pred

    def _POD_(self, n1, n3):
        return torch.true_divide(n1, n1 + n3 + self.eps)

    def _FAR_(self, n1, n2):
        return torch.true_divide(n2, n1 + n2 + self.eps)

    def _TS_(self, n1, n2, n3):
        return torch.true_divide(n1, n1 + n2 + n3 + self.eps)

    def _ETS_(self, n1, n2, n3, r):
        return torch.true_divide(n1 - r, n1 + n2 + n3 - r + self.eps)
    
    def _MICRO_F1_(self, n1, n2, n3):
        precision = torch.true_divide(n1, n1 + n2 + self.eps)
        recall = torch.true_divide(n1, n1 + n3 + self.eps)
        return torch.true_divide(2 * precision * recall, precision + recall + self.eps)
    
    def _MACRO_F1_(self, n1, n2, n3, n4):
        precision_true = torch.true_divide(n1, n1 + n2 + self.eps)
        recall_true = torch.true_divide(n1, n1 + n3 + self.eps)
        f1_true = torch.true_divide(2 * precision_true * recall_true, precision_true + recall_true + self.eps)
        precision_false = torch.true_divide(n4, n4 + n3 + self.eps)
        recall_false = torch.true_divide(n4, n4 + n2 + self.eps)
        f1_false = torch.true_divide(2 * precision_false * recall_false, precision_false + recall_false + self.eps)
        return torch.true_divide(f1_true + f1_false, 2)

    def cal_batch(self, y_true, y_pred, do_sigmoid=True):
        y_true, y_pred = self._transform(y_true, y_pred, do_sigmoid)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        r = torch.true_divide((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4)
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        micro_f1 = self._MICRO_F1_(n1,n2,n3)
        macro_f1 = self._MACRO_F1_(n1,n2,n3,n4)
        self.n1 += n1
        self.n2 += n2
        self.n3 += n3
        self.n4 += n4
        return pod, far, ts, ets, micro_f1, macro_f1

    def cal_batch_sum(self, y_true, y_pred, do_sigmoid=True):
        y_true, y_pred = self._transform_sum(y_true, y_pred, do_sigmoid)
        n1 = torch.sum((y_pred > 0) & (y_true > 0))
        n2 = torch.sum((y_pred > 0) & (y_true < 1))
        n3 = torch.sum((y_pred < 1) & (y_true > 0))
        n4 = torch.sum((y_pred < 1) & (y_true < 1))
        r = torch.true_divide((n1 + n2) * (n1 + n3), n1 + n2 + n3 + n4)
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        micro_f1 = self._MICRO_F1_(n1, n2, n3)
        macro_f1 = self._MACRO_F1_(n1, n2, n3, n4)
        self.n1sum += n1
        self.n2sum += n2
        self.n3sum += n3
        self.n4sum += n4
        return pod, far, ts, ets, micro_f1, macro_f1

    def cal_epoch(self):
        r = torch.true_divide((self.n1 + self.n2) * (self.n1 + self.n3), self.n1 + self.n2 + self.n3 + self.n4)
        pod = self._POD_(self.n1, self.n3)
        far = self._FAR_(self.n1, self.n2)
        ts = self._TS_(self.n1, self.n2, self.n3)
        ets = self._ETS_(self.n1, self.n2, self.n3, r)
        micro_f1 = self._MICRO_F1_(self.n1, self.n2, self.n3)
        macro_f1 = self._MACRO_F1_(self.n1, self.n2, self.n3, self.n4)
        return pod, far, ts, ets, micro_f1, macro_f1

    def cal_epoch_sum(self):
        r = torch.true_divide((self.n1sum + self.n2sum) * (self.n1sum + self.n3sum), self.n1sum + self.n2sum + self.n3sum + self.n4sum)
        pod = self._POD_(self.n1sum, self.n3sum)
        far = self._FAR_(self.n1sum, self.n2sum)
        ts = self._TS_(self.n1sum, self.n2sum, self.n3sum)
        ets = self._ETS_(self.n1sum, self.n2sum, self.n3sum, r)
        micro_f1 = self._MICRO_F1_(self.n1sum, self.n2sum, self.n3sum)
        macro_f1 = self._MACRO_F1_(self.n1sum, self.n2sum, self.n3sum, self.n4sum)
        return pod, far, ts, ets, micro_f1, macro_f1
    
    def cal_batch_neigh(self, y_true, y_pred, r, side, do_sigmoid=True):  # side:  side-length of one grid, km
        y_true, y_pred = self._transform(y_true, y_pred, do_sigmoid)
        y_pred = y_pred.reshape(-1,6,159, 159)
        y_true = y_true.reshape(-1,6,159, 159)
        # 0: unmarked, 1: hit, 2: miss, 3: false alarm, 4: correct negative
        y_sign = np.zeros((y_true.shape[0],6,159, 159))
        for batch in range(y_true.shape[0]):
            for t in range(6):
                for i in range(159):
                    for j in range(159):
                        if y_sign[batch][t][i][j] != 0:
                            continue
                        elif y_true[batch][t][i][j] == 0 and y_pred[batch][t][i][j] == 0:
                            y_sign[batch][t][i][j] = 4
                        elif y_true[batch][t][i][j] > 0 and y_pred[batch][t][i][j] > 0:
                            y_sign[batch][t][i][j] = 1
                        elif y_true[batch][t][i][j] > 0 and y_pred[batch][t][i][j] == 0:
                            flag = 0
                            for di in range(-(r//side),(r//side)+1):
                                if flag == 1: break
                                for dj in range(-(r//side),(r//side)+1):
                                    ii = i + di
                                    jj = j + dj
                                    if ii < 0 or ii >= 159 or jj < 0 or jj >= 159:
                                        continue
                                    if ((di*side)**2 + (dj*side)**2) > r**2:
                                        continue
                                    if y_pred[batch][t][ii][jj] > 0:
                                        y_sign[batch][t][i][j] = 1
                                        flag = 1
                                        break
                            if flag == 0:
                                y_sign[batch][t][i][j] = 2
                        elif y_pred[batch][t][i][j] > 0 and y_true[batch][t][i][j] == 0:
                            flag = 0
                            for di in range(-(r//side), (r//side)+1):
                                if flag == 1: break
                                for dj in range(-(r//side), (r//side)+1):
                                    ii = i + di
                                    jj = j + dj
                                    if ii < 0 or ii >= 159 or jj < 0 or jj >= 159:
                                        continue
                                    if ((di*side)**2 + (dj*side)**2) > r**2:
                                        continue
                                    if y_true[batch][t][ii][jj] > 0:
                                        y_sign[batch][t][i][j] = 1
                                        flag = 1
                                        break
                            if flag == 0:
                                y_sign[batch][t][i][j] = 3

        n1 = np.sum(y_sign == 1)
        n2 = np.sum(y_sign == 3)
        n3 = np.sum(y_sign == 2)
        n4 = np.sum(y_sign == 4)
        r = torch.true_divide((n1 + n2), n1 + n2 + n3 + n4) * (n1 + n3)
        
        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        micro_f1 = self._MICRO_F1_(n1,n2,n3)
        macro_f1 = self._MACRO_F1_(n1,n2,n3,n4)
        self.n1_neigh += n1
        self.n2_neigh += n2
        self.n3_neigh += n3
        self.n4_neigh += n4
        return pod, far, ts, ets, micro_f1, macro_f1

    def cal_batch_sum_neigh(self, y_true, y_pred, r, side, do_sigmoid=True):  # side:  side-length of one grid, km
        y_true, y_pred = self._transform_sum(y_true, y_pred, do_sigmoid)
        y_pred = y_pred.reshape(-1,159, 159)
        y_true = y_true.reshape(-1,159, 159)
        # 0: unmarked, 1: hit, 2: miss, 3: false alarm, 4: correct negative
        y_sign = np.zeros((y_pred.shape[0],159, 159))
        for batch in range(y_true.shape[0]):
            for i in range(159):
                for j in range(159):
                    if y_sign[batch][i][j] != 0:
                        continue
                    elif y_true[batch][i][j] == 0 and y_pred[batch][i][j] == 0:
                        y_sign[batch][i][j] = 4
                    elif y_true[batch][i][j] > 0 and y_pred[batch][i][j] > 0:
                        y_sign[batch][i][j] = 1
                    elif y_true[batch][i][j] > 0 and y_pred[batch][i][j] == 0:
                        flag = 0
                        for di in range(-1,1+1):
                            if flag == 1: break
                            for dj in range(-1,1+1):
                                ii = i + di
                                jj = j + dj
                                if ii < 0 or ii >= 159 or jj < 0 or jj >= 159:
                                    continue
                                # if ((di*side)**2 + (dj*side)**2) > r**2:
                                #     continue
                                if y_pred[batch][ii][jj] > 0:
                                    y_sign[batch][i][j] = 1
                                    flag = 1
                                    break
                        if flag == 0:
                            y_sign[batch][i][j] = 2
                    elif y_pred[batch][i][j] > 0 and y_true[batch][i][j] == 0:
                        flag = 0
                        for di in range(-(r//side), (r//side)+1):
                            if flag == 1: break
                            for dj in range(-(r//side), (r//side)+1):
                                ii = i + di
                                jj = j + dj
                                if ii < 0 or ii >= 159 or jj < 0 or jj >= 159:
                                    continue
                                if ((di*side)**2 + (dj*side)**2) > r**2:
                                    continue
                                if y_true[batch][ii][jj] > 0:
                                    y_sign[batch][i][j] = 1
                                    flag = 1
                                    break
                        if flag == 0:
                            y_sign[batch][i][j] = 3

        n1 = np.sum(y_sign == 1)
        n2 = np.sum(y_sign == 3)
        n3 = np.sum(y_sign == 2)
        n4 = np.sum(y_sign == 4)
        r = torch.true_divide((n1 + n2), n1 + n2 + n3 + n4) * (n1 + n3)

        pod = self._POD_(n1, n3)
        far = self._FAR_(n1, n2)
        ts = self._TS_(n1, n2, n3)
        ets = self._ETS_(n1, n2, n3, r)
        micro_f1 = self._MICRO_F1_(n1,n2,n3)
        macro_f1 = self._MACRO_F1_(n1,n2,n3,n4)
        self.n1sum_neigh += n1
        self.n2sum_neigh += n2
        self.n3sum_neigh += n3
        self.n4sum_neigh += n4
        return pod, far, ts, ets, micro_f1, macro_f1
    
    def cal_epoch_neigh(self):
        r = torch.true_divide((self.n1_neigh + self.n2_neigh), self.n1_neigh + self.n2_neigh + self.n3_neigh + self.n4_neigh)  * (self.n1_neigh + self.n3_neigh)
        pod = self._POD_(self.n1_neigh, self.n3_neigh)
        far = self._FAR_(self.n1_neigh, self.n2_neigh)
        ts = self._TS_(self.n1_neigh, self.n2_neigh, self.n3_neigh)
        ets = self._ETS_(self.n1_neigh, self.n2_neigh, self.n3_neigh, r)
        micro_f1 = self._MICRO_F1_(self.n1_neigh, self.n2_neigh, self.n3_neigh)
        macro_f1 = self._MACRO_F1_(self.n1_neigh, self.n2_neigh, self.n3_neigh, self.n4_neigh)
        return pod, far, ts, ets, micro_f1, macro_f1

    def cal_epoch_sum_neigh(self):
        r = torch.true_divide((self.n1sum_neigh + self.n2sum_neigh), self.n1sum_neigh + self.n2sum_neigh + self.n3sum_neigh + self.n4sum_neigh)  * (self.n1sum_neigh + self.n3sum_neigh)
        pod = self._POD_(self.n1sum_neigh, self.n3sum_neigh)
        far = self._FAR_(self.n1sum_neigh, self.n2sum_neigh)
        ts = self._TS_(self.n1sum_neigh, self.n2sum_neigh, self.n3sum_neigh)
        ets = self._ETS_(self.n1sum_neigh, self.n2sum_neigh, self.n3sum_neigh, r)
        micro_f1 = self._MICRO_F1_(self.n1sum_neigh, self.n2sum_neigh, self.n3sum_neigh)
        macro_f1 = self._MACRO_F1_(self.n1sum_neigh, self.n2sum_neigh, self.n3sum_neigh, self.n4sum_neigh)
        return pod, far, ts, ets, micro_f1, macro_f1


class Model_eval(object):
    def __init__(self, is_save_model, maxPOD = -0.5, maxPOD_epoch = -1, minFAR = 1.1, minFAR_epoch = -1, maxETS = -0.5, maxETS_epoch = -1):
        self.is_save_model = is_save_model
        self.maxPOD = maxPOD
        self.maxPOD_epoch = maxPOD_epoch
        self.minFAR = minFAR
        self.minFAR_epoch = minFAR_epoch
        self.maxETS = maxETS
        self.maxETS_epoch = maxETS_epoch
        # if self.is_save_model:
        #     with open(os.path.join('record.txt'), 'a') as f:
        #         f.write(str(datetime.datetime.now()) + '\r\n')
                # f.write(str(config_dict) + '\r\n')

    def __del__(self):
        info = '`model name: {}`\nmaxPOD: {} maxPOD_epoch: {}\nminFAR: {} minFAR_epoch: {}\nmaxETS: {} maxETS_epoch: {}\n'\
            .format(cfg.train_model_class.__name__, self.maxPOD, self.maxPOD_epoch, self.minFAR, self.minFAR_epoch, self.maxETS, self.maxETS_epoch)
        print(info)
        if self.is_save_model and self.maxPOD_epoch != -1 and self.minFAR_epoch != -1 and self.maxETS_epoch != -1:
            with open(os.path.join( 'record.txt'), 'a') as f:
                f.write(info + '\r\n')

    def eval(self, dataloader, model, epoch, do_sigmoid=True):
        model.eval()
        with torch.no_grad():

            val_calparams_epoch = Cal_params_epoch()
            for i, (X, y, y_aux, idx) in enumerate(dataloader):
                X = X.float().to(cfg.device)
                y = y.float().to(cfg.device)
                # print("hi"+str(i))
                predicted_frames = cfg.ablation(model, X)
                # print(predicted_frames.shape)

                # output
                pod, far, ts, ets, micro_f1, macro_f1 = val_calparams_epoch.cal_batch(y, predicted_frames, do_sigmoid)
                sumpod, sumfar, sumts, sumets, sum_micro_f1, sum_macro_f1 = val_calparams_epoch.cal_batch_sum(y, predicted_frames, do_sigmoid)
                info = 'epoch:{} ({}/{})'.format(epoch, i + 1, len(dataloader))
                print(info)
                del X
                del y
                # del _
                del predicted_frames
            sumpod, sumfar, sumts, sumets, sum_micro_f1, sum_macro_f1 = val_calparams_epoch.cal_epoch_sum()
            info = '`{}` VAL EPOCH INFO: epoch:{} \nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n sum_micro_F1:{:.5f}  sum_macro_F1:{:.5f}\n save model:{}\n'. \
                format(model.__class__.__name__, epoch, sumpod, sumfar, sumts, sumets, sum_micro_f1, sum_macro_f1, self.is_save_model)
            print(info)
            with open(os.path.join( 'record.txt'), 'a') as f:
                f.write(info + '\r\n')
            if self.is_save_model:
                if sumpod > self.maxPOD:
                    self.maxPOD = sumpod
                    self.maxPOD_epoch = epoch
                    self.save_model(model, 'model_maxPOD', epoch)
                if 1e-6 < sumfar < self.minFAR:
                    self.minFAR = sumfar
                    self.minFAR_epoch = epoch
                    self.save_model(model, 'model_minFAR', epoch)
                if sumets > self.maxETS:
                    self.maxETS = sumets
                    self.maxETS_epoch = epoch
                    self.save_model(model, 'model_maxETS', epoch)
                self.save_model(model, 'new_model', epoch)
            model.train()
            return sumets

    def save_model(self, model, name, epoch):
        torch.save(model.state_dict(), os.path.join( '{}_{}.pkl'.format(model.__class__.__name__, name)))
        info = 'save model file: {} successfully! (epoch={})'.format(name, epoch)
        print(info)
        # with open(os.path.join(self.config_dict['RecordFileDir'], 'record.txt'), 'a') as f:
        #     f.write(info + '\r\n')

