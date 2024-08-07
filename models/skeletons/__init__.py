from .rl_3df_gate import RL3DF_gate

def build_skeleton(cfg):
    return __all__[cfg.MODEL.SKELETON](cfg)

__all__ = {
    'RL3DF_gate': RL3DF_gate,
}
