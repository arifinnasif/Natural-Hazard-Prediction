from generator import *
from score import *
from model import *
import argparse
import config as cfg

# set random seed
torch.manual_seed(42)



def train(model, epoch=0, maxPOD = -0.5, maxPOD_epoch = -1, minFAR = 1.1, minFAR_epoch = -1, maxETS = -0.5, maxETS_epoch = -1):
    try:
        train_loader = get_train_loader()
        val_loader = get_val_loader()
        test_loader = get_test_loader()

        # model = Mjolnir_02(6, 8).float().to(torch.device("cuda"))

        # loss function
        
        criterion1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))
        criterion2 = nn.MSELoss()
        #criterion = nn.BCELoss()
    

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        # eval
        if epoch == 0:
            model_eval_valdata = Model_eval( is_save_model=True)
            model_eval_testdata = Model_eval( is_save_model=False)
        else:
            model_eval_valdata = Model_eval( is_save_model=True, maxPOD=maxPOD, maxPOD_epoch=maxPOD_epoch, minFAR=minFAR, minFAR_epoch=minFAR_epoch, maxETS=maxETS, maxETS_epoch=maxETS_epoch)
            model_eval_testdata = Model_eval( is_save_model=False, maxPOD=maxPOD, maxPOD_epoch=maxPOD_epoch, minFAR=minFAR, minFAR_epoch=minFAR_epoch, maxETS=maxETS, maxETS_epoch=maxETS_epoch)


        if epoch == 0:
            print('Beginning train!')
        else:
            print('Resuming train!')
        


        while epoch < cfg.epochs:
            for i, (X, y, y_aux, idx) in enumerate(train_loader):
                X = X.float().to(cfg.device)
                y = y.float().to(cfg.device)
                y_aux = y_aux.float().to(cfg.device)

                #predicted_frames = model(X).to(torch.device("cuda"))
                predicted_frames, radar_frames = model(X)


                # backward
                optimizer.zero_grad()
            
                loss = cfg.criterion1_weight*criterion1(torch.flatten(predicted_frames), torch.flatten(y)) \
                        + cfg.criterion2_weight*criterion2(torch.flatten(radar_frames), torch.flatten(y_aux))
                loss.backward()

                # update weights
                optimizer.step()

                # output
                print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i+1, len(train_loader), loss.item()))
                del X
                del y
                del y_aux
                del predicted_frames

            
            model_eval_valdata.eval(val_loader, model, epoch)
            model_eval_testdata.eval(test_loader, model, epoch)
            # print(val_sumets, test_sumets)

            epoch += 1

    except Exception as e:
        print(e)
        del model


parser = argparse.ArgumentParser(
                    prog='Mjolnir',
                    description='What the program does',
                    epilog='Text at the bottom of help')

# --resume 
# if --resume flag is given print resume or else do nothing
parser.add_argument('--resume', action='store_true', help='resume training')

args = parser.parse_args()
if args.resume:
    model_path = input("Enter model path: ")
    model = cfg.train_model_class(cfg.prev_hours, cfg.input_channels)
    model.load_state_dict(torch.load(model_path))
    model.float()
    model.to(cfg.device)

    last_epoch = int(input("Enter last epoch that was recorded: "))

    maxPOD = float(input("Enter maxPOD: "))
    maxPOD_epoch = int(input("Enter maxPOD_epoch: "))
    minFAR = float(input("Enter minFAR: "))
    minFAR_epoch = int(input("Enter minFAR_epoch: "))
    maxETS = float(input("Enter maxETS: "))
    maxETS_epoch = int(input("Enter maxETS_epoch: "))
    
    train(model, epoch=last_epoch+1, maxPOD=maxPOD, maxPOD_epoch=maxPOD_epoch, minFAR=minFAR, minFAR_epoch=minFAR_epoch, maxETS=maxETS, maxETS_epoch=maxETS_epoch)
else:
    model = cfg.train_model_class(cfg.prev_hours, cfg.input_channels).float().to(cfg.device)
    train(model)
