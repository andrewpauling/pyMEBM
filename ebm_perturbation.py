#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:49:34 2019

@author: andrewpauling
"""

import numpy as np
import matplotlib.pyplot as plt
from attrdict import AttrDict

from pyMEBM.ebm import EBM


class EBMPert(EBM):
    """
    Class containing a perturbation energy balance model
    """
    def __init__(self,
                 ocn_heat_uptake=False,
                 feedbacks='flat',                 
                 **kwargs):
        
        self.ocn_heat_uptake = ocn_heat_uptake
        self.feedbacks = feedbacks
        
        super(EBMPert, self).__init__(**kwargs)
    
    