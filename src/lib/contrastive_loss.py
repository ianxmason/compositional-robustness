import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLayerLoss:
    """
    Implements the contrastive loss function. Built starting from https://github.com/sthalles/SimCLR.
    """
    def __init__(self, single_corr_bs, temperature=0.15, dev=torch.device('cuda')):
        self.single_corr_bs = single_corr_bs
        self.temperature = temperature
        self.dev = dev

    def __call__(self, features, weight, n_views=2):
        if weight == 0:
            return torch.tensor(0.0).to(self.dev)
        if n_views <= 1:
            return torch.tensor(0.0).to(self.dev)

        # Flatten spatial dimensions if 4D
        features = features.reshape(features.shape[0], -1)
        assert len(features) % self.single_corr_bs == 0
        features = F.normalize(features, dim=1)

        labels = torch.cat([torch.arange(self.single_corr_bs) for _ in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.dev)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.dev)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        contrastive_criterion = nn.CrossEntropyLoss()
        """
        This is the basic contrastive case with n_views = 2
            logits = torch.cat([positives, negatives], dim=1)
            logits = logits / self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.dev)
            loss = contrastive_criterion(logits, labels)

        We want a more general case with arbitrary n_views (== number of corruptions in the batch)
        As a starting point for why this works: https://github.com/sthalles/SimCLR/issues/16 (also issue 33)
        """
        for i in range(n_views - 1):
            logits = torch.cat([positives[:, i:i + 1], negatives], dim=1)
            logits = logits / self.temperature
            if i == 0:
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.dev)
                loss = contrastive_criterion(logits, labels)
            else:
                loss += contrastive_criterion(logits, labels)

        return weight * loss
