import os
import os.path as osp
import yaml
import torch
import time
import argparse
from attrdict import AttrDict
from tqdm import tqdm
import numpy as np
import pickle
from utils.misc import load_module
from bayeso import acquisition, covariance
from utils.log import get_logger, RunningAverage
from utils.paths import results_path, evalsets_path
from utils.utils_bo import get_expected_improvement, twod_to_task, unormalize, Dataset

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', default='train')
    parser.add_argument('--expid', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # Data
    parser.add_argument('--max_num_points', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='ranger')

    # Model
    parser.add_argument('--model', type=str, default="tnpa")

    # Train
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=1000000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--num_ctx', type=int, default=None)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=3000)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # OOD settings
    parser.add_argument('--eval_kernel', type=str, default='rbf')
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()
    if args.expid is not None:
        args.root = osp.join(results_path, args.dataset, args.model, args.expid)
    else:
        args.root = osp.join(results_path, args.dataset, args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/{args.dataset}/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp", "tnpa", "tnpd", "tnpnd"]:
        model = model_cls(**config)
    model.cuda()

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)

    return args.root

def eval_(model, data, max_x = 1, min_x = 0):

    total_trajectories_y = []
    for X_test, y_test in data:
        y_test = y_test.reshape([len(y_test),1])
        X_shape = X_test.shape
        dataset = Dataset(X_test, y_test)[:]
        max_y = -0.1
        max_xi = 0
        max_i = 0
        best_trajectory = []
        #print(X_test)
        #print(y_test)
        test_set = twod_to_task(dataset, max_num_points=256, num_ctx=None, max_val_x=max_x, max_val_y=max_x, min_val_x=min_x, min_val_y=min_x, test=True)
        stds_total = []
        for xi_v in np.arange(-0.1,0.1, 0.01):
            context_x = []
            context_y = []
            trajectory = []
            
            for i in range(10):
                if i == 0:
                    r_val = np.random.randint(0,len(test_set.x[0]),[1])

                    context_x = test_set.x[r_val].reshape([1,1,X_shape[1]])
                    context_y = test_set.y[r_val].reshape([1,1,1])
                distribution = model.predict(context_x,context_y,test_set.x.unsqueeze(0))
                means = distribution.mean.detach().cpu().numpy()[0]
                stds = distribution.scale.detach().cpu().numpy()[0]

                #means = unormalize(means, max_x, min_x)
                stds_total.append(np.mean(stds))
                distribution = model.predict(context_x,context_y,context_x)
                means_new = distribution.mean.detach().cpu().numpy()[0]
                #current_best = unormalize(np.max(means_new), max_x, min_x)
                current_best = np.max(means_new)
                point = get_expected_improvement(current_best, means, stds, xi = xi_v)
                cord_to_sample = np.argmax(point)
                #print(context_y.detach().cpu().numpy().shape)
                point_x = dataset[0][cord_to_sample]
                point_y = dataset[1][cord_to_sample]
                ##NEW ADQUISTION
                acq_vals = -1.0 * acquisition.ei(np.ravel(means), np.ravel(stds), np.squeeze(context_y.detach().cpu().numpy(),0))
                ind_ = np.argmin(acq_vals)
                point_x = dataset[0][ind_]
                point_y = dataset[1][ind_]
                trajectory.append(point_y[0])
                if point_y >= max_y:
                    if point_y == max_y:    
                        if i < max_i:
                            max_y = point_y
                            max_xi = xi_v
                            max_i = i
                            best_trajectory = trajectory
                    else:
                        max_y = point_y
                        max_xi = xi_v
                        max_i = i
                        best_trajectory = trajectory
                #print("context_x[0]: ", context_x[0].get_device())
                context_x = torch.vstack((context_x[0],torch.Tensor(point_x).cuda())).unsqueeze(0)#.reshape([1,len(context_x[0])+1,X_shape[1]])
                context_y = torch.vstack((context_y[0],torch.Tensor(point_y).cuda())).unsqueeze(0)#.reshape([1,len(context_y[0])+1,1])
                #print("Point:", cord_to_sample, 'Value: ', dataset[1][cord_to_sample], "Cordinates: ", dataset[0][cord_to_sample])
        total_trajectories_y.append(np.array(best_trajectory))
        print("Last STDS: ", max(stds))
        print("Last STDS: ", min(stds))
        print("MAX Y: ", max_y, "MAX XI: ", max_xi, "MAX I: ", max_i, "MEAN STD: ", np.mean(stds_total))
        print("TRAJECTORY: ", best_trajectory)
    final_array = np.vstack(total_trajectories_y)
    # #print("FINAL ARRAY: ", final_array)
    mean_y_final = np.mean(final_array, axis=0)
    print("FINAL AVERAGE TRAJECTORY: ", mean_y_final)

def eval_botorch_(model, data, max_x = 1, min_x = 0):

    total_trajectories_y = []
    for X_test, y_test in data:
        print(y_test.shape)
        y_test = y_test.reshape([len(y_test),1])
        X_shape = X_test.shape
        print("X_shape: ", X_shape)
        dataset = Dataset(X_test, y_test)[:]
        max_y = 0
        max_xi = 0
        max_i = 0
        best_trajectory = []
        #print(X_test)
        #print(y_test)
        test_set = twod_to_task(dataset, max_num_points=None, num_ctx=None, max_val_x=max_x, max_val_y=max_x, min_val_x=min_x, min_val_y=min_x, test=True)
        stds_total = []
        for xi_v in np.arange(-0.1,0.1, 0.01):
            context_x = []
            context_y = []
            trajectory = []
            
            for i in range(10):
                if i == 0:
                    r_val = np.random.randint(0,len(test_set.x[0]),[1])

                    context_x = test_set.x[r_val].reshape([1,1,X_shape[1]])
                    context_y = test_set.y[r_val].reshape([1,1,1])
                distribution = model.predict(context_x,context_y,test_set.x.reshape([1,X_shape[0],X_shape[1]]))
                means = distribution.mean.detach().cpu().numpy()[0]
                stds = distribution.scale.detach().cpu().numpy()[0]

                #means = unormalize(means, max_x, min_x)
                stds_total.append(np.mean(stds))
                distribution = model.predict(context_x,context_y,context_x)
                means_new = distribution.mean.detach().cpu().numpy()[0]
                #current_best = unormalize(np.max(means_new), max_x, min_x)
                current_best = np.max(means_new)
                point = get_expected_improvement(current_best, means, stds, xi = xi_v)
                cord_to_sample = np.argmax(point)
                #print(context_y.detach().cpu().numpy().shape)
                point_x = dataset[0][cord_to_sample]
                point_y = dataset[1][cord_to_sample]
                ##NEW ADQUISTION
                acq_vals = -1.0 * acquisition.ei(np.ravel(means), np.ravel(stds), np.squeeze(context_y.detach().cpu().numpy(),0))
                ind_ = np.argmin(acq_vals)
                point_x = dataset[0][ind_]
                point_y = dataset[1][ind_]
                trajectory.append(point_y[0])
                if point_y >= max_y:
                    if point_y == max_y:    
                        if i < max_i:
                            max_y = point_y
                            max_xi = xi_v
                            max_i = i
                            best_trajectory = trajectory
                    else:
                        max_y = point_y
                        max_xi = xi_v
                        max_i = i
                        best_trajectory = trajectory
                #print("context_x[0]: ", context_x[0].get_device())
                context_x = torch.vstack((context_x[0],torch.Tensor(point_x).cuda())).reshape([1,len(context_x[0])+1,X_shape[1]])
                context_y = torch.vstack((context_y[0],torch.Tensor(point_y).cuda())).reshape([1,len(context_y[0])+1,1])
                #print("Point:", cord_to_sample, 'Value: ', dataset[1][cord_to_sample], "Cordinates: ", dataset[0][cord_to_sample])
        total_trajectories_y.append(np.array(best_trajectory))
        print("Last STDS: ", max(stds))
        print("Last STDS: ", min(stds))
        print("MAX Y: ", max_y, "MAX XI: ", max_xi, "MAX I: ", max_i, "MEAN STD: ", np.mean(stds_total))
        print("TRAJECTORY: ", best_trajectory)
    final_array = np.vstack(total_trajectories_y)
    #print("FINAL ARRAY: ", final_array)
    mean_y_final = np.mean(final_array, axis=0)
    print("FINAL AVERAGE TRAJECTORY: ", mean_y_final)

def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    # path, filename = get_eval_path(args)
    # if not osp.isfile(osp.join(path, filename)):
    #     print('generating evaluation sets...')
    #     gen_evalset(args)

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)

    #### FUSION ADDITIONS
    X_train = np.load(f'./datasets/{args.dataset}/X_train.npz')
    y_train = np.load(f'./datasets/{args.dataset}/y_train.npz')

    X_train = np.array([X_train[key].astype('float32')  for key in X_train.keys()])
    y_train = np.array([y_train[key].astype('float32')  for key in y_train.keys()])
    train_ds = Dataset(X_train, y_train)

    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=0)
    #### FUSION ADDITIONS

    #sampler = GPSampler(RBFKernel(), seed=args.train_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(args.root,
                f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(f"Experiment: {args.model}-{args.expid}")
        logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    for step in range(start_step, args.num_epochs+1):
        model.train()
        for batch in train_loader:
            batch = twod_to_task(batch, max_num_points=args.max_num_points,num_ctx=args.num_ctx, max_val_x=1, max_val_y=1, min_val_x=0, min_val_y=0)
            
            # batch = sampler.sample(
            #     batch_size=args.train_batch_size,
            #     max_num_points=args.max_num_points,
            #     device='cuda')
            optimizer.zero_grad()
            if args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, num_samples=args.train_num_samples)
            else:
                outs = model(batch)

            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)
        if step % args.print_freq == 0:
            line = f'{args.model}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                print("EVALUATION")
                line = eval(args, model)
                logger.info(line + '\n')

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, f'{step}_ckpt.tar'))

    args.mode = 'eval'
    eval(args, model)


def get_eval_path(args):
    path = osp.join(evalsets_path, 'gp')
    filename = f'{args.eval_kernel}-seed{args.eval_seed}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    return path, filename


def gen_evalset(args):

    if args.eval_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.eval_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.eval_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.eval_kernel}')
    print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)
    batches = []
    for _ in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(sampler.sample(
            batch_size=args.eval_batch_size,
            max_num_points=args.max_num_points,
            device='cuda'))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))


def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f'eval_{args.eval_kernel}'
            if args.t_noise is not None:
                eval_logfile += f'_tn_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path, filename = get_eval_path(args)

    ##FUSION ADDITIONS
    X_test = np.load(f'./datasets/{args.dataset}/X_test.npz')
    y_test = np.load(f'./datasets/{args.dataset}/y_test.npz')

    X_test = np.array([X_test[key].astype('float32') for key in X_test.keys()])
    y_test = np.array([y_test[key].astype('float32') for key in y_test.keys()])
    print("y_test[0].shape: ", y_test[0].shape)
    test_ds = Dataset(X_test, y_test)
    model.eval()

    eval_(model, test_ds)
    test_loader = torch.utils.data.DataLoader(test_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=0)
    ##FUSION ADDITIONS

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = twod_to_task(batch, max_num_points=args.max_num_points,num_ctx=args.num_ctx, max_val_x=1, max_val_y=1, min_val_x=0, min_val_y=0)
            for key, val in batch.items():
                batch[key] = val.cuda()
            
            if args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, num_samples=args.eval_num_samples)
            else:
                outs = model(batch, test=True)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)
    #sys.exit()
    return line


if __name__ == '__main__':
    main()