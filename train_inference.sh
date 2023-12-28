#!/bin/bash


while getopts 'c:n:t:r:p' OPT; do
    case $OPT in
        c) cuda=$OPTARG;;
        n) name=$OPTARG;;
		t) task=$OPTARG;;
        r) train="true";;
        p) predict="true";;
        
    esac
done
echo $name	


if ${train}
then
	
	cd /home/yuchen/nnSegnext/nnsegnext/
	CUDA_VISIBLE_DEVICES=${cuda} nnSegnext_train 3d_fullres nnSegnextTrainerV2_${name} ${task} 0
fi

if ${predict}
then


	cd /home/yuchen/nnSegnext/DATASET/nnSegnext_raw/nnSegnext_raw_data/Task001_HCP/
	CUDA_VISIBLE_DEVICES=${cuda} nnSegnext_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr nnSegnextTrainerV2_${name}
	python inference.py ${name}
fi



