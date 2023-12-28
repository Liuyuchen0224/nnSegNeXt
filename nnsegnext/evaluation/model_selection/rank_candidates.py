#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnsegnext.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "nnSegnextPlans"

    overwrite_plans = {
        'nnSegnextTrainerV2_2': ["nnSegnextPlans", "nnSegnextPlansisoPatchesInVoxels"], # r
        'nnSegnextTrainerV2': ["nnSegnextPlansnonCT", "nnSegnextPlansCT2", "nnSegnextPlansallConv3x3",
                            "nnSegnextPlansfixedisoPatchesInVoxels", "nnSegnextPlanstargetSpacingForAnisoAxis",
                            "nnSegnextPlanspoolBasedOnSpacing", "nnSegnextPlansfixedisoPatchesInmm", "nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_warmup': ["nnSegnextPlans", "nnSegnextPlansv2.1", "nnSegnextPlansv2.1_big", "nnSegnextPlansv2.1_verybig"],
        'nnSegnextTrainerV2_cycleAtEnd': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_cycleAtEnd2': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_reduceMomentumDuringTraining': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_graduallyTransitionFromCEToDice': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_independentScalePerAxis': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_Mish': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_Ranger_lr3en4': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_fp32': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_GN': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_momentum098': ["nnSegnextPlans", "nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_momentum09': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_DP': ["nnSegnextPlansv2.1_verybig"],
        'nnSegnextTrainerV2_DDP': ["nnSegnextPlansv2.1_verybig"],
        'nnSegnextTrainerV2_FRN': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_resample33': ["nnSegnextPlansv2.3"],
        'nnSegnextTrainerV2_O2': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_ResencUNet': ["nnSegnextPlans_FabiansResUNet_v2.1"],
        'nnSegnextTrainerV2_DA2': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_allConv3x3': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_ForceBD': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_ForceSD': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_LReLU_slope_2en1': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_lReLU_convReLUIN': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_ReLU': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_ReLU_biasInSegOutput': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_ReLU_convReLUIN': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_lReLU_biasInSegOutput': ["nnSegnextPlansv2.1"],
        #'nnSegnextTrainerV2_Loss_MCC': ["nnSegnextPlansv2.1"],
        #'nnSegnextTrainerV2_Loss_MCCnoBG': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_Loss_DicewithBG': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_Loss_Dice_LR1en3': ["nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_Loss_Dice': ["nnSegnextPlans", "nnSegnextPlansv2.1"],
        'nnSegnextTrainerV2_Loss_DicewithBG_LR1en3': ["nnSegnextPlansv2.1"],
        # 'nnSegnextTrainerV2_fp32': ["nnSegnextPlansv2.1"],
        # 'nnSegnextTrainerV2_fp32': ["nnSegnextPlansv2.1"],
        # 'nnSegnextTrainerV2_fp32': ["nnSegnextPlansv2.1"],
        # 'nnSegnextTrainerV2_fp32': ["nnSegnextPlansv2.1"],
        # 'nnSegnextTrainerV2_fp32': ["nnSegnextPlansv2.1"],

    }

    trainers = ['nnSegnextTrainer'] + ['nnSegnextTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'nnSegnextTrainerNewCandidate24_2',
        'nnSegnextTrainerNewCandidate24_3',
        'nnSegnextTrainerNewCandidate26_2',
        'nnSegnextTrainerNewCandidate27_2',
        'nnSegnextTrainerNewCandidate23_always3DDA',
        'nnSegnextTrainerNewCandidate23_corrInit',
        'nnSegnextTrainerNewCandidate23_noOversampling',
        'nnSegnextTrainerNewCandidate23_softDS',
        'nnSegnextTrainerNewCandidate23_softDS2',
        'nnSegnextTrainerNewCandidate23_softDS3',
        'nnSegnextTrainerNewCandidate23_softDS4',
        'nnSegnextTrainerNewCandidate23_2_fp16',
        'nnSegnextTrainerNewCandidate23_2',
        'nnSegnextTrainerVer2',
        'nnSegnextTrainerV2_2',
        'nnSegnextTrainerV2_3',
        'nnSegnextTrainerV2_3_CE_GDL',
        'nnSegnextTrainerV2_3_dcTopk10',
        'nnSegnextTrainerV2_3_dcTopk20',
        'nnSegnextTrainerV2_3_fp16',
        'nnSegnextTrainerV2_3_softDS4',
        'nnSegnextTrainerV2_3_softDS4_clean',
        'nnSegnextTrainerV2_3_softDS4_clean_improvedDA',
        'nnSegnextTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'nnSegnextTrainerV2_3_softDS4_radam',
        'nnSegnextTrainerV2_3_softDS4_radam_lowerLR',

        'nnSegnextTrainerV2_2_schedule',
        'nnSegnextTrainerV2_2_schedule2',
        'nnSegnextTrainerV2_2_clean',
        'nnSegnextTrainerV2_2_clean_improvedDA_newElDef',

        'nnSegnextTrainerV2_2_fixes', # running
        'nnSegnextTrainerV2_BN', # running
        'nnSegnextTrainerV2_noDeepSupervision', # running
        'nnSegnextTrainerV2_softDeepSupervision', # running
        'nnSegnextTrainerV2_noDataAugmentation', # running
        'nnSegnextTrainerV2_Loss_CE', # running
        'nnSegnextTrainerV2_Loss_CEGDL',
        'nnSegnextTrainerV2_Loss_Dice',
        'nnSegnextTrainerV2_Loss_DiceTopK10',
        'nnSegnextTrainerV2_Loss_TopK10',
        'nnSegnextTrainerV2_Adam', # running
        'nnSegnextTrainerV2_Adam_nnSegnextTrainerlr', # running
        'nnSegnextTrainerV2_SGD_ReduceOnPlateau', # running
        'nnSegnextTrainerV2_SGD_lr1en1', # running
        'nnSegnextTrainerV2_SGD_lr1en3', # running
        'nnSegnextTrainerV2_fixedNonlin', # running
        'nnSegnextTrainerV2_GeLU', # running
        'nnSegnextTrainerV2_3ConvPerStage',
        'nnSegnextTrainerV2_NoNormalization',
        'nnSegnextTrainerV2_Adam_ReduceOnPlateau',
        'nnSegnextTrainerV2_fp16',
        'nnSegnextTrainerV2', # see overwrite_plans
        'nnSegnextTrainerV2_noMirroring',
        'nnSegnextTrainerV2_momentum09',
        'nnSegnextTrainerV2_momentum095',
        'nnSegnextTrainerV2_momentum098',
        'nnSegnextTrainerV2_warmup',
        'nnSegnextTrainerV2_Loss_Dice_LR1en3',
        'nnSegnextTrainerV2_NoNormalization_lr1en3',
        'nnSegnextTrainerV2_Loss_Dice_squared',
        'nnSegnextTrainerV2_newElDef',
        'nnSegnextTrainerV2_fp32',
        'nnSegnextTrainerV2_cycleAtEnd',
        'nnSegnextTrainerV2_reduceMomentumDuringTraining',
        'nnSegnextTrainerV2_graduallyTransitionFromCEToDice',
        'nnSegnextTrainerV2_insaneDA',
        'nnSegnextTrainerV2_independentScalePerAxis',
        'nnSegnextTrainerV2_Mish',
        'nnSegnextTrainerV2_Ranger_lr3en4',
        'nnSegnextTrainerV2_cycleAtEnd2',
        'nnSegnextTrainerV2_GN',
        'nnSegnextTrainerV2_DP',
        'nnSegnextTrainerV2_FRN',
        'nnSegnextTrainerV2_resample33',
        'nnSegnextTrainerV2_O2',
        'nnSegnextTrainerV2_ResencUNet',
        'nnSegnextTrainerV2_DA2',
        'nnSegnextTrainerV2_allConv3x3',
        'nnSegnextTrainerV2_ForceBD',
        'nnSegnextTrainerV2_ForceSD',
        'nnSegnextTrainerV2_ReLU',
        'nnSegnextTrainerV2_LReLU_slope_2en1',
        'nnSegnextTrainerV2_lReLU_convReLUIN',
        'nnSegnextTrainerV2_ReLU_biasInSegOutput',
        'nnSegnextTrainerV2_ReLU_convReLUIN',
        'nnSegnextTrainerV2_lReLU_biasInSegOutput',
        'nnSegnextTrainerV2_Loss_DicewithBG_LR1en3',
        #'nnSegnextTrainerV2_Loss_MCCnoBG',
        'nnSegnextTrainerV2_Loss_DicewithBG',
        # 'nnSegnextTrainerV2_Loss_Dice_LR1en3',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
        # 'nnSegnextTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
