import click

@click.command()
@click.argument('job_name', type=str)
@click.option('--dataset', type=click.Choice(['pascal_voc']), default='pascal_voc')
@click.option('--arch', type=click.Choice(['deeplab3', 'denseunet161', 'resunet50', 'resunet101']), default='resunet50')
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
@click.option('--noise_std', type=float, default=1.0)
@click.option('--blob_sigma', type=float, default=10.0)
@click.option('--aug_hflip', is_flag=True, default=False)
@click.option('--aug_rot_range', type=float, default=20.0)
@click.option('--batch_size', type=int, default=6)
@click.option('--num_epochs', type=int, default=100)
@click.option('--iters_per_epoch', type=int, default=1000)
@click.option('--val_every_n_epochs', type=int, default=25)
@click.option('--device', type=str, default='cuda:0')
@click.option('--num_workers', type=int, default=8)
def train_dextr(job_name, dataset, arch, learning_rate, load_model, pretrained_lr_factor, lr_sched, lr_poly_power,
                opt_type, sgd_weight_decay,
                target_size, padding, extreme_range, noise_std, blob_sigma, aug_hflip, aug_rot_range,
                batch_size, num_epochs, iters_per_epoch, val_every_n_epochs, device, num_workers):
    settings = locals().copy()

    import torch.utils.data
    from dextr import model
    from dextr.data_pipeline import pascal_voc_dataset
    from dextr.architectures import deeplab, denseunet, resunet
    import job_output

    output = job_output.JobOutput(job_name, False)
    output.connect_streams()

    # Report setttings
    print('Settings:')
    print(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))]))


    torch_device = torch.device(device)

    if dataset == 'pascal_voc':
        train_ds_fn = lambda transform=None: pascal_voc_dataset.PascalVOCDataset('train', transform)
        val_ds_fn = lambda transform=None: pascal_voc_dataset.PascalVOCDataset('val', transform)
        val_truth_ds_fn = lambda: pascal_voc_dataset.PascalVOCDataset('val', None, load_input=False)
    else:
        print('Unknown dataset {}'.format(dataset))
        return

    # Build network
    if arch == 'deeplab3':
        net = deeplab.dextr_deeplab3(1)
    elif arch == 'denseunet161':
        net = denseunet.dextr_denseunet161(1)
    elif arch == 'resunet50':
        net = resunet.dextr_resunet50(1)
    elif arch == 'resunet101':
        net = resunet.dextr_resunet101(1)
    else:
        print('Unknown network architecture {}'.format(arch))
        return

    # Load model if path provided
    if load_model is not None:
        print('Loading snapshot from {}...'.format(load_model))
        dextr_model = torch.load(load_model, map_location=torch_device)
        new_lr_factor = pretrained_lr_factor
    else:
        dextr_model = model.DextrModel(net, (target_size, target_size), padding, blob_sigma)
        new_lr_factor = 1.0


    def on_epoch_finished(epoch, model):
        output.write_checkpoint(dextr_model)

    model.training_loop(dextr_model, train_ds_fn, val_ds_fn, val_truth_ds_fn, extreme_range,
                        noise_std, learning_rate, pretrained_lr_factor, new_lr_factor,
                        lr_sched, lr_poly_power, opt_type, sgd_weight_decay, aug_hflip, aug_rot_range,
                        batch_size, iters_per_epoch, num_epochs, val_every_n_epochs,
                        torch_device, num_workers, True, on_epoch_finished)


if __name__ == '__main__':
    train_dextr()