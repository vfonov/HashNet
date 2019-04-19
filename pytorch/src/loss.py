import torch
#from torch.autograd import Variable

def pairwise_loss(outputs1, outputs2, label1, label2, 
                  sigmoid_param=1.0, l_threshold=15.0, class_num=1.0):
    similarity = (torch.mm(label1.float(), label2.float().t()) > 0.0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)
    mask_dot = dot_product >  l_threshold
    mask_exp = dot_product <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = (torch.log(1+exp_product) - similarity * dot_product)

    loss =  torch.sum(torch.masked_select(exp_loss, mask_ep)) + \
            torch.sum(torch.masked_select(dot_loss, mask_dp)) * class_num + \
            torch.sum(torch.masked_select(exp_loss, mask_en)) + \
            torch.sum(torch.masked_select(dot_loss, mask_dn))

    return loss / (torch.sum(mask_positive.float()) * class_num + torch.sum(mask_negative.float()))
