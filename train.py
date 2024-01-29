from generator import *
from score import *
from model import *
import argparse


def train(model, epoch=0, maxPOD = -0.5, maxPOD_epoch = -1, minFAR = 1.1, minFAR_epoch = -1, maxETS = -0.5, maxETS_epoch = -1):
    try:
        full_dataset = CustomDataset()

        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(dataset=train_dataset, batch_size=18, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=val_dataset, batch_size=18, shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=18, shuffle=False, num_workers=0)

        # model = Mjolnir_02(6, 8).float().to(torch.device("cuda"))

        # loss function
        
        criterion1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))
        criterion2 = nn.MSELoss()
        #criterion = nn.BCELoss()
    

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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
        


        while epoch < 200:
            for i, (X, y, y_aux, idx) in enumerate(train_loader):
                X = X.float().to(torch.device("cuda"))
                y = y.float().to(torch.device("cuda"))
                y_aux = y_aux.float().to(torch.device("cuda"))

                #predicted_frames = model(X).to(torch.device("cuda"))
                predicted_frames, radar_frames = model(X)


                # backward
                optimizer.zero_grad()
            
                loss = 80*criterion1(torch.flatten(predicted_frames), torch.flatten(y)) + criterion2(torch.flatten(radar_frames), torch.flatten(y_aux))
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
    model = Mjolnir_02(6, 8)
    model.load_state_dict(torch.load(model_path))
    model.float()
    model.to(torch.device("cuda"))

    last_epoch = int(input("Enter last epoch that was recorded: "))

    maxPOD = float(input("Enter maxPOD: "))
    maxPOD_epoch = int(input("Enter maxPOD_epoch: "))
    minFAR = float(input("Enter minFAR: "))
    minFAR_epoch = int(input("Enter minFAR_epoch: "))
    maxETS = float(input("Enter maxETS: "))
    maxETS_epoch = int(input("Enter maxETS_epoch: "))
    
    train(model, epoch=last_epoch+1, maxPOD=maxPOD, maxPOD_epoch=maxPOD_epoch, minFAR=minFAR, minFAR_epoch=minFAR_epoch, maxETS=maxETS, maxETS_epoch=maxETS_epoch)
else:
    model = Mjolnir_02(6, 8).float().to(torch.device("cuda"))
    train(model)
