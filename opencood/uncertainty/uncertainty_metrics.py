import torch


def binary_entropy(prob, eps=1e-8):

    p = prob.clamp(
        min=eps,
        max=1.0 - eps
    )

    return -(
        p * torch.log(p)
        +
        (1.0 - p)
        * torch.log(1.0 - p)
    )


def predictive_entropy(mean_prob):

    return binary_entropy(mean_prob)


def mutual_information(prob_samples):
    """
    prob_samples:
        [T, ...]

    MI = H[E[p]] - E[H[p]]
    """

    mean_prob = prob_samples.mean(dim=0)

    entropy_mean = binary_entropy(
        mean_prob
    )

    mean_entropy = binary_entropy(
        prob_samples
    ).mean(dim=0)

    return (
        entropy_mean
        - mean_entropy
    ).clamp_min(0.0)


def classification_variance(prob_samples):

    return prob_samples.var(
        dim=0,
        unbiased=False
    )


def regression_variance(reg_samples):

    return reg_samples.var(
        dim=0,
        unbiased=False
    )