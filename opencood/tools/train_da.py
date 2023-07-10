import argparse
import os
import statistics

import torch
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.domain_adaptions.da_module import DomainAdaptationModule


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--model', default='',
                        help='for fine-tuned training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    #######################################################
    # DA_Component: dataloader for source and target domain
    #######################################################

    source_opencood_train_dataset = build_dataset(hypes,
                                                  visualize=False,
                                                  train=True,
                                                  isSim=True)
    # Modify the root to the target domain
    hypes['root_dir'] = hypes['root_dir_target']
    target_opencood_train_dataset = build_dataset(hypes,
                                                  visualize=False,
                                                  train=True,
                                                  isSim=False)

    source_train_loader = DataLoader(source_opencood_train_dataset,
                                     batch_size=hypes['train_params'][
                                         'batch_size'],
                                     num_workers=8,
                                     collate_fn=source_opencood_train_dataset.collate_batch_train,
                                     shuffle=True,
                                     pin_memory=False,
                                     drop_last=True)
    target_train_loader = DataLoader(target_opencood_train_dataset,
                                     batch_size=hypes['train_params'][
                                         'batch_size'],
                                     num_workers=8,
                                     collate_fn=source_opencood_train_dataset.collate_batch_train,
                                     shuffle=True,
                                     pin_memory=False,
                                     drop_last=True)
    # domain adaption module
    DA_module = DomainAdaptationModule(hypes['model']['args'])
    #####################################################################
    # DA_Component: dataloader for real world validation
    #####################################################################
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False,
                                              isSim=False)
    if opencood_validate_dataset is not None:
        print("opencood_validate_dataset is loaded")
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=source_opencood_train_dataset.collate_batch_train,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes, da=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        DA_module.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(source_train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        print('Loaded model from {}'.format(saved_path))

    else:
        if opt.model:
            saved_path = train_utils.setup_train(hypes)
            model_path = opt.model
            init_epoch = 0
            pretrained_state = torch.load(
                os.path.join(model_path, 'latest.pth'))
            model_dict = model.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if (
                    k in model_dict and v.shape == model_dict[k].shape)}
            model_dict.update(pretrained_state)
            model.load_state_dict(model_dict)
            print('Loaded pretrained model from {}'.format(model_path))

        else:
            init_epoch = 0
            # if we train the model from scratch, we need to create a folder
            # to save the model,
            saved_path = train_utils.setup_train(hypes)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    ####################################################################
    # DA_Component: training for source and target domain
    ####################################################################
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(source_train_loader), leave=True)
        target_train_loader_iter = iter(target_train_loader)

        for iteration, source_batch_data in enumerate(source_train_loader):

            try:
                target_batch_data = next(target_train_loader_iter)
            except StopIteration:
                target_train_loader_iter = iter(target_train_loader)
                target_batch_data = next(target_train_loader_iter)

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            source_batch_data = train_utils.to_device(source_batch_data,
                                                      device)
            target_batch_data = train_utils.to_device(target_batch_data,
                                                      device)

            total_batch_data = [source_batch_data['ego'],
                                target_batch_data['ego']]

            if not opt.half:
                ouput_dict = model(total_batch_data)
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       source_batch_data['ego']['label_dict'])

            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(total_batch_data)
                    final_loss = criterion(ouput_dict,
                                           source_batch_data['ego'][
                                               'label_dict'])

            ####DA Loss#####
            da_loss = DA_module(ouput_dict)
            losses = sum(loss for loss in da_loss.values())

            Total_loss = losses + final_loss

            criterion.logging_da(epoch, iteration, len(source_train_loader),
                                 writer, da_loss, pbar=pbar2)

            pbar2.update(1)
            # back-propagation
            if not opt.half:
                Total_loss.backward()
                optimizer.step()
            else:
                scaler.scale(Total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + iteration)

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))

            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
