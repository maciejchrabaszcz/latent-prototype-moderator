import torch
from sklearn.cluster import KMeans

def perform_covariance_shrinkage(
    covariance_matrix: torch.Tensor,
    alpha_1: float = 1,
    alpha_2: float = 1,
):
    raise NotImplementedError


def get_bayes_precision_estimate(covariance_matrix: torch.Tensor, n: int, d: int):
    inv_cov = d * torch.inverse(
        (n - 1) * covariance_matrix
        + torch.trace(covariance_matrix) * torch.eye(d).to(covariance_matrix.device)
    )
    return inv_cov


def calculate_correlation_matrix(covariance_matrix: torch.Tensor):
    corr = covariance_matrix / torch.sqrt(
        torch.diag(covariance_matrix).unsqueeze(0)
        @ torch.diag(covariance_matrix).unsqueeze(1)
    )
    return corr


def calculate_means_and_cov(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    calculate_per_class_cov: bool = False,
    scale_covariances: bool = False,
):
    """
    Calculate the mean of the hidden states for each label and the covariance matrix.
    The covariance matrix is calculated using the centered hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states tensor of shape (batch_size, hidden_size).
        labels (torch.Tensor): The labels tensor of shape (batch_size,).
        calculate_per_class_cov (bool): Whether to calculate the covariance for each class.
        scale_covariances (bool): Whether to scale the covariance matrices to correlation matrices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the means and covariances for each label.
    """
    unique_labels = len(torch.unique(labels))
    means = []

    N = hidden_states.size()[0]
    if calculate_per_class_cov:
        covs = []
        covs_counts = []
    else:
        covs = None
    for label in range(unique_labels):
        mask = labels == label
        label_hidden_states = hidden_states[mask]

        mean = label_hidden_states.mean(dim=0)
        if calculate_per_class_cov:
            cur_n = label_hidden_states.size(0)
            centered_hidden_states = label_hidden_states - mean.unsqueeze(0)
            cov = centered_hidden_states.T @ centered_hidden_states / (cur_n - 1)
            if scale_covariances:
                # calculate correlation matrix
                cov = calculate_correlation_matrix(cov)
            covs_counts.append(cur_n)
            covs.append(cov)
        means.append(mean)
    centered_hidden_states = torch.concat(
        [
            hidden_states[labels == label] - mean.unsqueeze(0)
            for label, mean in zip(range(unique_labels), means)
        ]
    )
    cov = centered_hidden_states.T @ centered_hidden_states / (N - 1)

    means = torch.stack(means)

    if calculate_per_class_cov:
        covs = torch.stack(covs)
        return means, cov, covs, covs_counts
    return means, cov


def calculate_clusters_and_covs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    calculate_per_class_cov: bool = False,
    scale_covariances: bool = False,
    n_clusters_per_class: int = 2,
    **kmeans_kwargs,
):
    """
    Calculate the mean of the hidden states for each label and the covariance matrix.
    The covariance matrix is calculated using the centered hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states tensor of shape (batch_size, hidden_size).
        labels (torch.Tensor): The labels tensor of shape (batch_size,).
        calculate_per_class_cov (bool): Whether to calculate the covariance for each class.
        scale_covariances (bool): Whether to scale the covariance matrices to correlation matrices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the means and covariances for each label.
    """
    unique_labels = len(torch.unique(labels))
    all_means = []

    N = hidden_states.size()[0]
    if calculate_per_class_cov:
        covs = []
        covs_counts = []
    else:
        covs = None
        covs_counts = None

    all_clustering_labels = []
    for label in range(unique_labels):
        mask = labels == label
        label_hidden_states = hidden_states[mask]
        kmeans = KMeans(n_clusters=n_clusters_per_class, **kmeans_kwargs)
        clustering_labels = torch.tensor(
            kmeans.fit_predict(label_hidden_states.cpu().numpy())
            + (n_clusters_per_class * label)
        ).to(hidden_states.device)
        all_clustering_labels.append(clustering_labels)
        means = torch.tensor(kmeans.cluster_centers_).to(hidden_states.device)
        if calculate_per_class_cov:
            for i in range(n_clusters_per_class):
                cur_hidden_states = label_hidden_states[
                    clustering_labels == i + (label * n_clusters_per_class)
                ]
                cur_n = cur_hidden_states.size(0)
                centered_hidden_states = cur_hidden_states - means[i].unsqueeze(0)
                cov = centered_hidden_states.T @ centered_hidden_states / (cur_n - 1)
                if scale_covariances:
                    # calculate correlation matrix
                    cov = calculate_correlation_matrix(cov)
                covs_counts.append(cur_n)
                covs.append(cov)
        all_means.append(means)
    all_means = torch.concatenate(all_means)
    all_clustering_labels = torch.cat(all_clustering_labels)
    centered_hidden_states = torch.concat(
        [
            hidden_states[all_clustering_labels == label] - mean.unsqueeze(0)
            for label, mean in zip(
                range(unique_labels * n_clusters_per_class), all_means
            )
        ]
    )
    cov = centered_hidden_states.T @ centered_hidden_states / (N - 1)

    if calculate_per_class_cov:
        covs = torch.stack(covs)
    return all_means, cov, covs, covs_counts


def calculate_clusters_and_precision_matrices(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    calculate_per_class_cov: bool = False,
    scale_covariances: bool = False,
    n_clusters_per_class: int = 2,
    **kmeans_kwargs,
):
    """
    Calculate the mean of the hidden states for each label and the covariance matrix.
    The covariance matrix is calculated using the centered hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states tensor of shape (batch_size, hidden_size).
        labels (torch.Tensor): The labels tensor of shape (batch_size,).
        calculate_per_class_cov (bool): Whether to calculate the covariance for each class.
        scale_covariances (bool): Whether to scale the covariance matrices to correlation matrices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the means and covariances for each label.
    """
    all_means, cov, covs, covs_counts = calculate_clusters_and_covs(
        hidden_states=hidden_states,
        labels=labels,
        calculate_per_class_cov=calculate_per_class_cov,
        scale_covariances=scale_covariances,
        n_clusters_per_class=n_clusters_per_class,
        **kmeans_kwargs,
    )
    precision_matrix = get_bayes_precision_estimate(
        cov, hidden_states.size(0), hidden_states.size(1)
    )
    if calculate_per_class_cov:
        precision_matrices = []
        for i in range(len(covs)):
            cur_n = covs_counts[i]
            cur_cov = covs[i]
            cur_precision_matrix = get_bayes_precision_estimate(
                cur_cov, cur_n, hidden_states.size(1)
            )
            precision_matrices.append(cur_precision_matrix)
        precision_matrices = torch.stack(precision_matrices)
    else:
        precision_matrices = None
    return all_means, precision_matrix, precision_matrices


def calculate_means_and_inv_cov(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    calculate_per_class_cov: bool = False,
    scale_covariances: bool = False,
):
    """
    Calculate the mean of the hidden states for each label and the inverse covariance matrix.
    The covariance matrix is calculated using the centered hidden states.

    Args:
        hidden_states (torch.Tensor): The hidden states tensor of shape (batch_size, hidden_size).
        labels (torch.Tensor): The labels tensor of shape (batch_size,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the means and covariances for each label.
    """
    outputs = calculate_means_and_cov(
        hidden_states,
        labels,
        calculate_per_class_cov=calculate_per_class_cov,
        scale_covariances=scale_covariances,
    )
    means = outputs[0]
    cov = outputs[1]

    inv_cov = get_bayes_precision_estimate(
        cov, hidden_states.size(0), hidden_states.size(1)
    )

    if calculate_per_class_cov:
        covs = outputs[2]
        covs_counts = outputs[3]
        inv_covs = []
        for i in range(len(covs)):
            cur_n = covs_counts[i]
            cur_cov = covs[i]
            cur_inv_cov = get_bayes_precision_estimate(
                cur_cov, cur_n, hidden_states.size(1)
            )
            inv_covs.append(cur_inv_cov)
        inv_covs = torch.stack(inv_covs)
        return means, inv_cov, inv_covs
    return means, inv_cov


def get_gda_params(means: torch.Tensor, inv_cov: torch.Tensor):
    """
    Calculate the GDA parameters.

    Args:
        means (torch.Tensor): The means tensor of shape (num_classes, hidden_size).
        inv_cov (torch.Tensor): The inverse covariance matrix of shape (hidden_size, hidden_size).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the GDA parameters.
    """
    priori = torch.log(torch.tensor(1.0 / means.size(0))).to(means.device)
    W = means @ inv_cov
    b = priori - 0.5 * torch.einsum("nd, dc, nc -> n", means, inv_cov, means)

    return {"W": W, "b": b}


def get_gda_pred(hidden_states, W, b, return_probs=True):
    output = hidden_states @ W.T + b
    if return_probs:
        output = torch.softmax(output, dim=1)
    else:
        output = torch.argmax(output, dim=1)
    return output


def get_mahalanobis_distance(hidden_states, means, invs):
    seperate_invs = invs.ndim == 3
    distances = []
    for i in range(len(means)):
        centered_hidden_states = hidden_states - means[i].unsqueeze(0)
        if seperate_invs:
            cur_inv = invs[i]
        else:
            cur_inv = invs
        maha_distance = torch.einsum(
            "ik,kl,il->i",
            centered_hidden_states,
            cur_inv,
            centered_hidden_states,
        )
        distances.append(maha_distance)
    maha_distance = torch.stack(distances, dim=1)
    return maha_distance


def get_mahalanobis_pred(hidden_states, means, inv_covs, return_probs=True):
    output = -get_mahalanobis_distance(hidden_states, means, inv_covs)
    if return_probs:
        output = torch.softmax(output, dim=1)
    else:
        output = torch.argmax(output, dim=1)
    return output


def get_nmc_pred(hidden_states, means, return_probs=True):
    output = -torch.cdist(hidden_states, means, p=2)
    if return_probs:
        output = torch.softmax(output, dim=1)
    else:
        output = torch.argmax(output, dim=1)
    return output
