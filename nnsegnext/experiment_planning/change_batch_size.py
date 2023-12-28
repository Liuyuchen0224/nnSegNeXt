from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/home/xuesheng3/liuyuchen/Data/nnSegnext_raw_data_base/nnSegnext_preprocessed/Task004_BrainTissueIBSR/nnSegnextPlansv2.1_plans_3D.pkl'
    output_file = '/home/xuesheng3/liuyuchen/Data/nnSegnext_raw_data_base/nnSegnext_preprocessed/Task004_BrainTissueIBSR/nnSegnextPlansv2.1_plans_3D.pkl'
    input_file1 = '/home/liuyc/PaperProject/Results/attu/3d_fullres/Task001_BrainTissueHCP/nnSegnextTrainerV2_attu__nnSegnextPlansv2.1/fold_0/model_final_checkpoint.model.pkl'


    a = load_pickle(input_file)
    # b = load_pickle(input_file2)
    # c=a
    # a['init']=['/home/liuyc/PaperProject/Datasets/nnSegnext_raw_data_base/nnSegnext_preprocessed/Task001_BrainTissueHCP/nnSegnextPlansv2.1_plans_3D.pkl',0,'/home/liuyc/PaperProject/Results/attu/3d_fullres/Task001_BrainTissueHCP/nnSegnextTrainerV2_attu__nnSegnextPlansv2.1',
    # '/home/liuyc/PaperProject/Datasets/nnSegnext_raw_data_base/nnSegnext_preprocessed/Task001_BrainTissueHCP',False,0,True,False,True]

    # a['name']='nnSegnextTrainerV2_attu'
    # a['class']=   "<class 'nnsegnext.training.network_training.nnSegnextTrainerV2_attu.nnSegnextTrainerV2_attu'>"
    # a['plans']['data_identifier']='nnSegnextData_plans_v2.1'
    
    a['plans_per_stage'][0]['patch_size'] = [128,128,160]
    # # # # #a['plans_per_stage'][0]['batch_size'] = int(np.floor(6 / 9 * a['plans_per_stage'][0]['batch_size']))"""  """
    # # # # a['plans_per_stage'][0]['batch_size'] = 4
    
    save_pickle(a, output_file)
    b = load_pickle(input_file)
    print(a['plans_per_stage'][0]['patch_size'])