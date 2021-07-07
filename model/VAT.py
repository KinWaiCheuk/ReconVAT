import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from nnAudio import Spectrogram
from .constants import *
from model.utils import Normalization

class stepwise_VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, XI, epsilon, n_power):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon

    def forward(self, model, x):  
        with torch.no_grad():
            y_ref, _ = model(x) # This will be used as a label, therefore no need grad()
            
        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d)
            y_pred, _ = model(x + r)
            dist =F.binary_cross_entropy(y_pred, y_ref)
            dist.backward() # Calculate gradient wrt d
            d = d.grad.detach()
            model.zero_grad() # prevent gradient change in the model    

        # generating virtual labels and calculate VAT    
        r_adv = self.epsilon * _l2_normalize(d)
#         logit_p = logit.detach()
        y_pred, _ = model(x + r_adv)
        vat_loss = F.binary_cross_entropy(y_pred, y_ref)              
            
        return vat_loss, r_adv  # already averaged
    
def _l2_normalize(d):
    d = d/torch.norm(d, dim=2, keepdim=True)
    return d