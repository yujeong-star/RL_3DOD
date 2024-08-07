import torch.nn as nn

from models import pre_processor, backbone_3d, head, roi_head, img_cls

class RL3DF_gate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL
        
        self.list_module_names = [
            'pre_processor', 'pre_processor2', 'img_cls', 'backbone_3d', 'head', 'roi_head', 
        ]
        self.list_modules = []
        self.build_rl_detector()

    def build_rl_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_img_cls(self):
        if self.cfg_model.get('IMG_CLS', None) is None:
            return None
        
        module = img_cls.__all__[self.cfg_model.IMG_CLS.NAME]()
        return module 

    def build_pre_processor(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
        return module 
    
    def build_pre_processor2(self):
        if self.cfg_model.get('PRE_PROCESSOR2', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR2.NAME](self.cfg)
        return module 

    def build_backbone_3d(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module

    def forward(self, x):
        for module in self.list_modules:
            x = module(x)
        return x