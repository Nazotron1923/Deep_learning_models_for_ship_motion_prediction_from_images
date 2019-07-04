#!/bin/bash
# change your path here(path to BASE_DIR, i.e. Pre's parent folder,without '/' at the end)
BASE_DIR=/home/interns/zhi
IMG_DIR=3dmodel/mixData
TEST_DIR=3dmodel/mixData
CODE_DIR=/Pre/
IDX=1
RESULT_FILE=/result.txt
RES_DIR=results/
declare -a model_list=("cnn" "CNN_LSTM")
#declare -a model_list=("cnn")
cd $BASE_DIR
start_tm=`date +%s%N`
# T is time gap
for T in {10..40..3} ; do
# for T in {20..30..1} ; do
    cd $BASE_DIR$CODE_DIR
    python3 3dmodel/transformData.py -i $IMG_DIR"/paras_origin.json" -o $IMG_DIR"/labels.json" -t ${T};
    python3 3dmodel/transformData.py -i $TEST_DIR"/paras_origin.json" -o $TEST_DIR"/labels.json" -t ${T};
    cp $BASE_DIR$CODE_DIR$IMG_DIR"/labels.json" $RES_DIR."labels_"${T}".json"
    cd $BASE_DIR
    for model in "${model_list[@]}"; do
        echo "loop No:"$IDX
        echo "current model:"${model}"; time gap: "$T
        # change to a suitable epochs number(50 is ok, 1 for test)
        python3 -m Pre.train -tf $BASE_DIR$CODE_DIR$IMG_DIR --model_type ${model} -t ${T} --num_epochs 100 -bs 8
        python3 -m Pre.test -f $BASE_DIR$CODE_DIR$TEST_DIR -m ${model} -w $BASE_DIR$CODE_DIR$RES_DIR${model}"_model_"${T}"_tmp.pth" -t $T -bs 8
        python3 $BASE_DIR$CODE_DIR"pltDiff.py" -m ${model} -t $T -o $BASE_DIR$CODE_DIR$TEST_DIR"/paras_origin.json" -p $BASE_DIR$CODE_DIR$RES_DIR"predictions_"${model}"_"${T}".json"
    let IDX+=1
    echo -e "\n"
    done
done
cd $BASE_DIR$CODE_DIR
python pltModelTimegap.py -f $BASE_DIR$CODE_DIR$RESULT_FILE
end_tm=`date +%s%N`
use_tm=`echo $end_tm $start_tm | awk '{print($1-$2)/3600000000000}'`
echo "time used:"$use_tm" h"
echo "All finished !"
