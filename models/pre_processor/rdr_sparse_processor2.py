'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

import torch
import torch.nn as nn
import time

# V2
class RadarSparseProcessor_L(nn.Module):
    def __init__(self, cfg):
        super(RadarSparseProcessor_L, self).__init__()
        self.cfg = cfg
        self.roi = cfg.DATASET.RDR_SP_CUBE.ROI

        x_min, x_max = self.roi['x']
        y_min, y_max = self.roi['y']
        z_min, z_max = self.roi['z']
        self.min_roi = [x_min, y_min, z_min]

        self.grid_size = cfg.DATASET.RDR_SP_CUBE.GRID_SIZE
        self.input_dim = cfg.MODEL.PRE_PROCESSOR.INPUT_DIM

        if self.cfg.DATASET.RDR_SP_CUBE.METHOD == 'quantile':
            self.type_data = 0
        else:
            print('* Exception error (Pre-processor): check RDR_SP_CUBE.METHOD')

    def forward(self, dict_item):
        if self.type_data == 0:
            sp_cube = dict_item['ldr_pc_64'].cuda()

            sp_indices = dict_item['pts_batch_indices_ldr_pc_64'].cuda()

            # Cut Doppler if self.input_dim = 4
            sp_cube = sp_cube[:,:self.input_dim]

            # Get z, y, x coord
            x_min, y_min, z_min = self.min_roi
            grid_size = self.grid_size
            
            x_coord, y_coord, z_coord = sp_cube[:, 0:1], sp_cube[:, 1:2], sp_cube[:, 2:3]

            z_ind = torch.ceil((z_coord-z_min) / grid_size).long()
            y_ind = torch.ceil((y_coord-y_min) / grid_size).long()
            x_ind = torch.ceil((x_coord-x_min) / grid_size).long() # -40.2 -> 0 for y
                
            sp_indices = torch.cat((sp_indices.unsqueeze(-1), z_ind, y_ind, x_ind), dim = -1)

            dict_item['sp_features_l'] = sp_cube
            dict_item['sp_indices_l'] = sp_indices
            
        return dict_item
