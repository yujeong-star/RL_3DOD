import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pipelines.pipeline_detection_v1_0 import PipelineDetection_v1_0

EXP_NAME = ''
MODEL_EPOCH = 0

if __name__ == '__main__':
    PATH_CONFIG = f'./logs/{EXP_NAME}/config.yml'
    PATH_MODEL = f'./logs/{EXP_NAME}/models/model_{MODEL_EPOCH}.pt'

    pline = PipelineDetection_v1_0(PATH_CONFIG, mode='test')
    import shutil
    shutil.copy2(os.path.realpath(__file__), os.path.join(pline.path_log, 'executed_code.txt'))
    pline.load_dict_model(PATH_MODEL)
    
    pline.validate_kitti_conditional(epoch=MODEL_EPOCH, list_conf_thr=[0.3], is_subset=False, is_print_memory=False)
    
