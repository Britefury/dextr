import click

@click.command()
@click.argument('out_path', type=str)
@click.option('--dataset', type=click.Choice(['pascal_voc']), default='pascal_voc')
@click.option('--arch', type=click.Choice(['deeplab3', 'denseunet161', 'resunet50', 'resunet101']), default='deeplab3')
@click.option('--load_model', type=click.Path(readable=True))
@click.option('--learning_rate', type=float, default=0.1)
@click.option('--pretrained_lr_factor', type=float, default=0.1)
@click.option('--lr_sched', type=click.Choice(['none', 'cosine', 'poly']), default='poly')
@click.option('--lr_poly_power', type=float, default=0.9)
@click.option('--opt_type', type=click.Choice(['sgd', 'adam']), default='sgd')
@click.option('--sgd_weight_decay', type=float, default=1e-4)
@click.option('--target_size', type=int, default=512)
@click.option('--padding', type=int, default=10)
@click.option('--extreme_range', type=int, default=5)
@click.option('--noise_range', type=float, default=1.0)
@click.option('--blob_sigma', type=float, default=10.0)
@click.option('--aug_hflip', is_flag=True, default=False)
@click.option('--aug_rot_range', type=float, default=20.0)
@click.option('--batch_size', type=int, default=5)
@click.option('--num_epochs', type=int, default=100)
@click.option('--iters_per_epoch', type=int, default=1000)
@click.option('--device', type=str, default='cuda:0')
@click.option('--num_workers', type=int, default=8)
def train_dextr(out_path, dataset, arch, learning_rate, load_model, pretrained_lr_factor, lr_sched, lr_poly_power,
                opt_type, sgd_weight_decay,
                target_size, padding, extreme_range, noise_range, blob_sigma, aug_hflip, aug_rot_range,
                batch_size, num_epochs, iters_per_epoch, device, num_workers):
    settings = locals().copy()

    import math
    import itertools
    import time
    import tqdm
    import torch, torch.nn as nn, torch.nn.functional as F
    import torch.utils.data
    from torchvision.transforms import Compose
    from dextr import lr_schedules, repeat_sampler
    from dextr.data_pipeline import dextr_transforms
    from dextr.data_pipeline import pascal_voc_dataset
    from dextr.architectures import deeplab, denseunet, resunet

    torch_device = torch.device(device)

    train_transform = Compose([
        dextr_transforms.DextrTrainingTransform(
            target_size_yx=(target_size, target_size),
            padding=padding,
            extreme_range=extreme_range,
            noise_range=noise_range,
            blob_sigma=blob_sigma,
            hflip=aug_hflip,
            rot_range=math.radians(aug_rot_range),
            fix_box_to_extreme=True
        ),
        dextr_transforms.DextrToTensor(),
        dextr_transforms.DextrNormalize(),
    ])

    eval_transform = Compose([
        # dextr_transforms.DextrEvaluationTransform(
        #     target_size_yx=(target_size, target_size),
        #     padding=padding,
        #     extreme_range=extreme_range,
        #     noise_range=0.0,
        #     blob_sigma=blob_sigma,
        # ),
        dextr_transforms.DextrTrainingTransform(
            target_size_yx=(target_size, target_size),
            padding=padding,
            extreme_range=extreme_range,
            noise_range=noise_range,
            blob_sigma=blob_sigma,
            hflip=aug_hflip,
            rot_range=math.radians(aug_rot_range),
            fix_box_to_extreme=True
        ),
        dextr_transforms.DextrToTensor(),
        dextr_transforms.DextrNormalize(),
    ])

    if dataset == 'pascal_voc':
        train_ds = pascal_voc_dataset.PascalVOCDataset('train', train_transform)
        val_ds = pascal_voc_dataset.PascalVOCDataset('val', eval_transform)
    else:
        print('Unknown dataset {}'.format(dataset))
        return

    # Build network
    if arch == 'deeplab3':
        net_dict = deeplab.dextr_deeplab3(1)
    elif arch == 'denseunet161':
        net_dict = denseunet.dextr_denseunet161(1)
    elif arch == 'resunet50':
        net_dict = resunet.dextr_resunet50(1)
    elif arch == 'resunet101':
        net_dict = resunet.dextr_resunet101(1)
    else:
        print('Unknown network architecture {}'.format(arch))
        return

    net = net_dict['model'].to(torch_device)

    # Load model if path provided
    if load_model is not None:
        snapshot = torch.load(load_model)
        net.load_state_dict(snapshot['model_state'])
        target_size = snapshot['target_size']
        padding = snapshot['padding']
        blob_sigma = snapshot['blob_sigma']
        new_lr_factor = pretrained_lr_factor
    else:
        new_lr_factor = 1.0

    # Create optimizer
    if opt_type == 'sgd':
        optimizer = torch.optim.SGD([
            dict(params=net_dict['pretrained_parameters'], lr=learning_rate * pretrained_lr_factor),
            dict(params=net_dict['new_parameters'], lr=learning_rate * new_lr_factor)],
            momentum=0.9, nesterov=True, weight_decay=sgd_weight_decay, lr=learning_rate)
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam([
            dict(params=net_dict['pretrained_parameters'], lr=learning_rate * pretrained_lr_factor),
            dict(params=net_dict['new_parameters'], lr=learning_rate * new_lr_factor)])
    else:
        print('Unknown optimizer type {}'.format(opt_type))
        return

    # LR schedule
    if iters_per_epoch == -1:
        iters_per_epoch = len(train_ds) // batch_size
    total_iters = iters_per_epoch * num_epochs

    lr_iter_scheduler = lr_schedules.make_lr_scheduler(
        optimizer=optimizer, total_iters=total_iters, schedule_type=lr_sched,
        poly_power=lr_poly_power
    )


    train_sampler = repeat_sampler.RepeatSampler(torch.utils.data.RandomSampler(train_ds))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size, sampler=train_sampler,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size, num_workers=num_workers)


    # Report setttings
    print('Settings:')
    print(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))]))

    # Report dataset size
    print('Dataset:')
    print('len(train_ds)={}'.format(len(train_ds)))
    print('len(val_ds)={}'.format(len(val_ds)))


    train_iter = iter(train_loader)

    print('Training...')
    iter_i = 0
    for epoch in range(num_epochs):
        t1 = time.time()

        train_loss_accum = 0.0

        for batch in tqdm.tqdm(itertools.islice(train_iter, iters_per_epoch), total=iters_per_epoch):
            if lr_iter_scheduler is not None:
                lr_iter_scheduler.step(iter_i)

            input = batch['input'].to(torch_device)
            target_mask = batch['target_mask'].to(torch_device)

            optimizer.zero_grad()

            # Compute class balance
            target_mask_flat = target_mask.view(len(target_mask), -1)
            target_balance = target_mask_flat.mean(dim=1)
            target_balance = target_balance[:, None, None, None]
            bg_weight = target_balance
            fg_weight = 1.0 - target_balance
            weight = bg_weight + (fg_weight - bg_weight) * target_mask

            pred_out = net(input)
            pred_logits = pred_out['out']

            loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, weight=weight)

            loss.backward()

            optimizer.step()

            train_loss_accum += float(loss)

            iter_i += 1

        train_loss_accum /= iters_per_epoch


        val_loss_accum = 0.0
        val_iou_accum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, total=len(val_loader)):
                input = batch['input'].to(torch_device)
                target_mask_cpu = batch['target_mask']
                target_mask = target_mask_cpu.to(torch_device)

                # Compute class balance
                target_mask_flat = target_mask.view(len(target_mask), -1)
                target_balance = target_mask_flat.mean(dim=1)
                target_balance = target_balance[:, None, None, None]
                bg_weight = target_balance
                fg_weight = 1.0 - target_balance
                weight = bg_weight + (fg_weight - bg_weight) * target_mask

                pred_out = net(input)
                pred_logits = pred_out['out']

                loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, weight=weight)
                val_loss_accum += float(loss)

                pred_logits = pred_logits.detach().cpu().numpy()

                pred_fg = pred_logits > 0.0
                true_fg = target_mask_cpu.detach().cpu().numpy() > 0.5

                iou = (pred_fg & true_fg).sum() / max((pred_fg | true_fg).sum(), 1.0)
                val_iou_accum += iou
                n_val_batches += 1

        val_loss_accum /= n_val_batches
        val_iou_accum /= n_val_batches


        t2 = time.time()

        print('Epoch {} took {:.3f}s: loss={:.6f}, VAL loss={:.6f}, mIoU={:.3%}'.format(
            epoch + 1, t2 - t1, train_loss_accum, val_loss_accum, val_iou_accum))


    if out_path.strip() != 'none':
        model_state = {key: value.to('cpu') for key, value in net.state_dict().items()}
        snapshot = dict(
            model_state=model_state,
            target_size=target_size,
            padding=padding,
            blob_sigma=blob_sigma,
        )
        torch.save(snapshot, out_path)


if __name__ == '__main__':
    train_dextr()