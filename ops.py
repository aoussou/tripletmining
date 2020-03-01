#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:17:26 2020

@author: john
"""
import torch

def get_triplet_mask(labels):

    indices_equal = torch.eye(labels.shape[0]).cuda()
    indices_not_equal = 1-indices_equal
    
    i_not_equal_j = torch.unsqueeze(indices_not_equal,2).bool()
    i_not_equal_k = torch.unsqueeze(indices_not_equal,1).bool()
    j_not_equal_k = torch.unsqueeze(indices_not_equal,0).bool()
    
    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k 
    
    label_equal = torch.eq(torch.unsqueeze(labels,0),torch.unsqueeze(labels,1))
    
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    
    valid_labels = i_equal_j & (~i_equal_k)
    
#    print(distinct_indices)
#    print('*'*10)
#    print(valid_labels)
#    
#    STOP
    
    mask = (distinct_indices & valid_labels)
    mask = mask.type(torch.cuda.FloatTensor)
    return mask

def batch_all_triplet_loss(labels, embeddings, margin):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
#    pairwise_dist = torch.sqrt(torch.cdist(embeddings,embeddings.t(), p=2))   
#    pairwise_dist = pairwise_dist*(1.0 - torch.eye(pairwise_dist.shape[0]).cuda())
    
    pairwise_dist = pairwise_distances(embeddings)
 
    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
#    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
#    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
#    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = get_triplet_mask(labels)
    triplet_loss = mask*triplet_loss
    
    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.max(triplet_loss, torch.tensor(0.0).cuda())

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.gt(triplet_loss, 1e-16)
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets.float() + 1e-16)
    # very important to do .float(), otherwise the result will be an integer, sometimes exactly 0
    
    return triplet_loss, fraction_positive_triplets

def pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.mm(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.max(distances, torch.tensor(0.0).cuda())

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0)
        mask = mask.type(torch.cuda.FloatTensor)
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances