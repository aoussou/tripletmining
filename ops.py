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
    pairwise_dist = torch.cdist(embeddings,embeddings, p=2)

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
    triplet_loss = torch.max(triplet_loss, torch.tensor(0).cuda())

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.gt(triplet_loss, 1e-16)
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets