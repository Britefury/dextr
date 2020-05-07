import click

@click.command()
def compute_std_params():
    import math
    import numpy as np
    import tqdm
    from dextr.data_pipeline import pascal_voc_dataset, dextr_transforms

    transform = dextr_transforms.DextrTrainingTransform(
        target_size_yx=(512, 512),
        padding=10,
        extreme_range=5,
        noise_std=0.0,
        blob_sigma=10.0,
        hflip=False,
        rot_range=math.radians(0.0),
        fix_box_to_extreme=True
    )

    ds = pascal_voc_dataset.PascalVOCDataset('train', transform)

    # Adapted from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    def aggregate(n_a, mu_a, var_a, n_b, mu_b, var_b):
        delta = mu_b - mu_a
        m_a = var_a * (n_a - 1)
        m_b = var_b * (n_b - 1)
        m2 = m_a + m_b + delta ** 2 * n_a * n_b / (n_a + n_b)
        n_x = n_a + n_b
        mu_x = mu_a + delta * (n_b / n_x)
        var_x = m2 / (n_a + n_b - 1)
        return n_x, mu_x, var_x

    n = 0
    mu = 0.0
    var = 0.0
    for i in tqdm.tqdm(range(len(ds))):
        x = ds[i]['gauss_blobs']
        x_n = np.prod(x.shape)
        x_mu = x.mean()
        x_var = x.var()

        if n == 0:
            n = x_n
            mu = x_mu
            var = x_var
        else:
            n, mu, var = aggregate(n, mu, var, x_n, x_mu, x_var)

    sig = np.sqrt(var)

    print('Mean: {}, std-dev: {}'.format(mu, sig))


if __name__ == '__main__':
    compute_std_params()