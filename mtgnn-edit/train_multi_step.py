import torch
import numpy as np
import os
import argparse
import time
from util import *
from trainer import Trainer
from net import gtnet
from tqdm import tqdm

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cpu',help='')
#parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--data', type=str, default='./data/pendulum.txt')
#parser.add_argument('--data', type=str, default='./data/us_weather.parquet')

parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
parser.add_argument('--cl', type=str_to_bool, default=False,help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=12,help='number of nodes/variables')
#parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')


parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension') # In dim handles the number of input features, 2 for graph stuff apparently
#parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension') # for pendulums etc?
#parser.add_argument('--seq_in_len',type=int,default=336,help='input sequence length')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')
#parser.add_argument('--seq_out_len',type=int,default=96,help='output sequence length')

parser.add_argument('--layers',type=int,default=2,help='number of layers')
parser.add_argument('--batch_size',type=int,default=256,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
#parser.add_argument('--learning_rate',type=float,default=1e-7,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')


parser.add_argument('--epochs',type=int,default=2,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--save',type=str,default='./save/weights/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--subgraph_size',type=int,default=12,help='k')

parser.add_argument('--runs',type=int,default=1,help='number of runs')

parser.add_argument('--only_validate', action='store_true', default=False,help='whether to skip training and just load the matching best model and run evaluation')
parser.add_argument('--continue_train', action='store_true', default=False,help='Continue training from best epoch')


args = parser.parse_args()
torch.set_num_threads(3)


def main(runid):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    device = torch.device(args.device)

    if 'powerplant' in args.data or 'weather' in args.data or 'ETTm1' in args.data:
        print('load df and dataloader')
        dataloader = load_dataframe(get_data(args.data), args.batch_size, args.seq_in_len, args.seq_out_len)
        print('loaded')
    elif 'pendulum' in args.data or 'ETTm1' in args.data:
        #data_preprocess(args.data, args)
        path = data_preprocess(args.data, args)
        #args.data = '.'.join(args.data.split('.')[:-1])
        #args.data = '.'.join(args.data.split('.')[:-1])
        args.data = path
        dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    else:
        dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)


    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    #load data
    scaler = dataloader['scaler']

    predefined_A = None
    # predefined_A = load_adj(args.adj_data)
    # predefined_A = torch.tensor(predefined_A)-torch.eye(args.num_nodes)
    # predefined_A = predefined_A.to(device)

    # if args.load_static_feature:
    #     static_feat = load_node_feature('data/sensor_graph/location.csv')
    # else:
    #     static_feat = None

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)
    
    print('found', torch.cuda.device_count(), 'GPU devices, using data parallel')
    model = torch.nn.DataParallel(model)

    def plot_adj(store, epoch):
        adj = model.module.gc(torch.arange(args.num_nodes).to(device)).detach().cpu()
        #adj = model.fixed_adj.detach().cpu()
        import matplotlib.pyplot as plt
        plt.imshow(adj)

        if adj.shape[0] == 12:
            # Custom tick labels for three double pendulums
            tick_labels = ['P1_x1', 'P1_y1', 'P1_x2', 'P1_y2', 'P2_x1', 'P2_y1', 'P2_x2', 'P2_y2', 'P3_x1', 'P3_y1', 'P3_x2', 'P3_y2']
            plt.xticks(range(len(tick_labels)), tick_labels, rotation=90)
            plt.yticks(range(len(tick_labels)), tick_labels)
        if store:
            path = f'{args.save}/adjacency'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f'{path}/epoch_{epoch}.png')
        else:
            plt.show()
        # alpha = args.tanhalpha
        # nodevec1 = torch.tanh(alpha*model.gc.lin1(nodevec1))
        # nodevec2 = torch.tanh(alpha*model.gc.lin2(nodevec2))
        # a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        # import matplotlib.pyplot as plt; plt.imshow(model.fixed_adj.detach()); plt.show()

    from glob import glob
    def find_highest_epoch_number(save_path, expid):
        highest_epoch = -1
        pattern = f"{save_path}exp{expid}_*_epoch_*.pth"
        for filename in glob(pattern):
            parts = filename.split('_')
            epoch_num = int(parts[-1].split('.')[0])
            highest_epoch = max(highest_epoch, epoch_num)
        if highest_epoch == -1:
            raise ValueError(f"No saved models found in {save_path}")

        
        return highest_epoch


    print(args)
    print('The receptive field size is', model.module.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    if args.continue_train:
        # When continuing disable curriculum
        engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, cl=False)
    else:
        engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device, args.cl)

    from copy import deepcopy
    before = deepcopy(engine.model.module.gc.lin1.weight)
    if args.only_validate:
        print("ONLY VALIDATING, NO TRAINIG")
        best_epoch = find_highest_epoch_number(args.save, args.expid)
        print("Found best epoch", best_epoch)
        filepath = args.save + "exp" + str(args.expid) + "_" + str(runid) +f"_epoch_{best_epoch}.pth"
        print('loading', filepath)
        missing, unexpected = engine.model.module.load_state_dict(torch.load(filepath))
        print('missing keys', missing)
        print('unexpected keys', unexpected)
        plot_adj(store=True, epoch='best')

    start_epoch = 1
    if args.continue_train:
        print("CONTINUING TRAINING FROM PREVIOUS CHECKPOINT")
        best_epoch = find_highest_epoch_number(args.save, args.expid)
        print("Found best epoch", best_epoch)
        start_epoch = best_epoch

        filepath = args.save + "exp" + str(args.expid) + "_" + str(runid) +f"_epoch_{best_epoch}.pth"
        print('loading', filepath)
        engine.model.module.load_state_dict(torch.load(filepath))
        print('continuing from there')

    after = engine.model.module.gc.lin1.weight
    if args.continue_train or args.only_validate:
        difference = ((before - after)**2).mean()
        print('difference in weights of initialized and pretrained', difference)
        assert difference > 0, "Model weights should have changed after loading"
    
    del before
    del after

    if not args.only_validate:
        print("start training...",flush=True)
        his_loss =[]
        val_time = []
        train_time = []
        minl = 1e5
        best_epoch = start_epoch
        for i in range(start_epoch,args.epochs+1):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).float().to(device) # [64, 12, 207, 2]), [batch, input seq, num_nodes, input_dim]
                trainx= trainx.transpose(1, 3)# ([64, 2, 207, 12]) [batch, input_dim, num_nodes, input seq]
                trainy = torch.Tensor(y).float().to(device) 
                trainy = trainy.transpose(1, 3)
                if iter%args.step_size2==0:
                    perm = np.random.permutation(range(args.num_nodes))
                num_sub = int(args.num_nodes/args.num_split)
                for j in range(args.num_split):
                    if j != args.num_split-1:
                        id = perm[j * num_sub:(j + 1) * num_sub]
                    else:
                        id = perm[j * num_sub:]
                    id = torch.tensor(id).to(device)
                    tx = trainx[:, :, id, :]
                    ty = trainy[:, :, id, :]
                    metrics = engine.train(tx, ty[:,0,:,:],id)
                    train_loss.append(metrics[0])
                    train_mape.append(metrics[1])
                    train_rmse.append(metrics[2])
                if iter % args.print_every == 0 :
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
            t2 = time.time()
            train_time.append(t2-t1)
            #validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            valid_mse = []

            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).float().to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).float().to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:,0,:,:])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
                valid_mse.append(metrics[3])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i,(s2-s1)))
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            mvalid_mse = np.mean(valid_mse)
            his_loss.append(mvalid_loss)
            plot_adj(store=True, epoch=i)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, valid MSE {:.4f} Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_mse, (t2 - t1)),flush=True)

            if mvalid_loss<minl:
                #torch.save(engine.model.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth")
                torch.save(engine.model.module.state_dict(), args.save + "exp" + str(args.expid) + "_" + str(runid) +f"_epoch_{i}.pth")
                best_epoch = i
                minl = mvalid_loss


        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        plot_adj(store=True, epoch='final')

        bestid = np.argmin(his_loss)
        engine.model.module.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +f"_epoch_{best_epoch}.pth"))

        print("Training finished")
        print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    else: # // only validating, skipping all training
        print("skipped training, validating...")
        pass # only
    # realy = torch.Tensor(dataloader['y_val']).to(device) # [batch, input_dim, num_nodes, input seq]
    # realy = realy.transpose(1,3)[:,0,:,:]
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    valid_mse = []
    with torch.no_grad():

        print('Validating...')
        #for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        shape = 0
        for iter, (x, y) in tqdm(enumerate(dataloader['val_loader'].get_iterator()), total=len(dataloader['val_loader'])):
            #if shape == 0:
            #    shape = x.shape[0]
            #elif shape != x.shape[0]:
            #    print('Batch size changed, breaking')
            #    break
            
            testx = torch.Tensor(x).float().to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).float().to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_mse.append(metrics[3])

            # outputs.append(preds.squeeze().detach().cpu())
            # targets.append(testy[:, 0, :, :].squeeze().detach().cpu())
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mse = np.mean(valid_mse)
        log = 'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f} Valid MSE: {:.4f}'
        print(log.format(mvalid_loss, mvalid_mape, mvalid_rmse, mvalid_mse), flush=True)
        del valid_loss
        del valid_mape
        del valid_rmse

        print('Testing...')
        shape = 0
        test_loss = []
        test_mape = []
        test_rmse = []
        test_mse = []
        #for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        for iter, (x, y) in tqdm(enumerate(dataloader['test_loader'].get_iterator()), total=len(dataloader['test_loader'])):
            # if shape == 0:
            #     shape = x.shape[0]
            # elif shape != x.shape[0]:
            #     print('Batch size changed, breaking')
            #     break
            testx = torch.Tensor(x).float().to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).float().to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            test_loss.append(metrics[0])
            test_mape.append(metrics[1])
            test_rmse.append(metrics[2])
            test_mse.append(metrics[3])

        mtest_loss = np.mean(test_loss)
        mtest_mape = np.mean(test_mape)
        mtest_rmse = np.mean(test_rmse)
        mtest_mse = np.mean(test_mse)
        log = 'Test Loss: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f} Test MSE: {:.4f}'
        print(log.format(mtest_loss, mtest_mape, mtest_rmse, mtest_mse), flush=True)

    return vmae, vmape, vrmse, mae, mape, rmse, full_mae, full_mape, full_rmse

if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    full_mae = []
    full_mape = []
    full_rmse = []
    for i in range(args.runs):
         main(i)