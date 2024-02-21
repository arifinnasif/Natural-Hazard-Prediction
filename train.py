import time
from generator import *
from score import *
from model import *
import argparse
import config as cfg




def train(model, epoch=0, maxPOD = -0.5, maxPOD_epoch = -1, minFAR = 1.1, minFAR_epoch = -1, maxETS = -0.5, maxETS_epoch = -1):
    try:
        with open('record.txt', 'a') as f:
            f.write('Training started at: ' + str(datetime.datetime.now()) + '\n\n')
            # write config
            f.write(str(cfg.__dict__) + '\n\n')
        train_loader = get_train_loader()
        val_loader = get_val_loader()
        test_loader = get_test_loader()

        # model = Mjolnir_02(6, 8).float().to(torch.device("cuda"))

        # loss function
        
        criterion1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))
        criterion2 = nn.MSELoss()
        criterion3 = NewLoss()
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


        temporal_sigma = 1
        spatial_sigma = 20
        


        while epoch < cfg.epochs:
            start_clock = time.time()
            temporal_sigma = temporal_sigma * 0.99
            spatial_sigma = spatial_sigma * 0.99
            for i, (X, y, y_aux, idx) in enumerate(train_loader):

                transformed_y = make_blur(y, temporal_sigma, spatial_sigma)
                X = X.float().to(cfg.device)
                y = y.float().to(cfg.device)
                # y_aux = y_aux.float().to(cfg.device)
                transformed_y = transformed_y.float().to(cfg.device)

                #predicted_frames = model(X).to(torch.device("cuda"))
                predicted_frames = model(X)


                # backward
                optimizer.zero_grad()
            
                loss = cfg.criterion1_weight*criterion1(torch.flatten(predicted_frames), torch.flatten(y)) \
                        + cfg.criterion3_weight*criterion3(torch.flatten(predicted_frames), torch.flatten(y), torch.flatten(transformed_y))
                loss.backward()

                # update weights
                optimizer.step()

                # output
                print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i+1, len(train_loader), loss.item()))
                del X
                del y
                # del y_aux
                del transformed_y
                del predicted_frames

            
            model_eval_valdata.eval(val_loader, model, epoch)
            model_eval_testdata.eval(test_loader, model, epoch)
            # print(val_sumets, test_sumets)

            epoch += 1
            stop_clock = time.time()

            print('Time taken for epoch: ', stop_clock - start_clock, 'seconds')
            estimated_finish_time = (cfg.epochs - epoch) * (stop_clock - start_clock) + datetime.datetime.now().timestamp()
            print('Estimated finish time: ', datetime.datetime.fromtimestamp(estimated_finish_time))

            with open('record.txt', 'a') as f:
                f.write('Time taken for epoch: ' + str(stop_clock - start_clock) + 'seconds\n')
                f.write('Estimated finish time: ' + str(datetime.datetime.fromtimestamp(estimated_finish_time)) + '\n\n')

    except Exception as e:
        print(e)
        del model
    
    finally:
        del model
        del train_loader
        del val_loader
        del test_loader
        del model_eval_valdata
        del model_eval_testdata
        torch.cuda.empty_cache()

        with open('record.txt', 'a') as f:
            f.write('Training ended at: ' + str(datetime.datetime.now()) + '\n\n')

def make_blur(y, temporal_sigma, spatial_sigma):
    transformed_y = gaussian_filter(y[:, :, 0, :, :].cpu().numpy(), sigma=[0, temporal_sigma, spatial_sigma, spatial_sigma])
    max_values = np.max(transformed_y, axis=(2,3))
    zero_mask = max_values == 0

    # Replace zeros with 1 to avoid division by zero
    max_values[zero_mask] = 1

    # Expand dimensions of max_values for broadcasting
    max_values = max_values[:, :, np.newaxis, np.newaxis]

    # Perform division
    transformed_y = np.divide(transformed_y, max_values, where=~zero_mask[:, :, np.newaxis, np.newaxis])

    #transformed_y_2 = transformed_y_2 / np.max(transformed_y_2, axis=(2, 3), keepdims=True)
    transformed_y = transformed_y.reshape(y.shape[0], 6, 1, 159, 159)
    transformed_y = torch.from_numpy(transformed_y)
    return transformed_y

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
