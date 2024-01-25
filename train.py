from generator import *
from score import *
from model import *


def train():
    try:
        full_dataset = CustomDataset()

        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.2 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(dataset=train_dataset, batch_size=44, shuffle=True, num_workers=0)
        val_loader = DataLoader(dataset=val_dataset, batch_size=44, shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=44, shuffle=False, num_workers=0)

        model = StepDeep().float().to(torch.device("cuda"))

        # loss function
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))
        #criterion = nn.BCELoss()
    

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # eval
        model_eval_valdata = Model_eval( is_save_model=True)
        model_eval_testdata = Model_eval( is_save_model=False)

        print('Beginning train!')


        for epoch in range(100):
            for i, (X, y, idx) in enumerate(train_loader):
                X = X.float().to(torch.device("cuda"))
                y = y.float().to(torch.device("cuda"))

                #predicted_frames = model(X).to(torch.device("cuda"))
                predicted_frames = model(X)


                # backward
                optimizer.zero_grad()
            
                loss = criterion(torch.flatten(predicted_frames), torch.flatten(y))
                loss.backward()

                # update weights
                optimizer.step()

                # output
                print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i+1, len(train_loader), loss.item()))
                del X
                del y
                del predicted_frames


            val_sumets = model_eval_valdata.eval(val_loader, model, epoch)
            test_sumets = model_eval_testdata.eval(test_loader, model, epoch)
            # print(val_sumets, test_sumets)

    except Exception as e:
        print(e)
        del model


train()