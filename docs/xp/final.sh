fusioncheckpoints=(
  "fusionfusion-custom16rll-multi-2021-05-03 11-30-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=39-val_loss=0.0038"
  "fusionfusion-custom16rll-multi-2021-05-03 22-03-freiburgthermal-c3-sord-1,2,3-a1-logl2-lw-rgb,ir-epoch=41-val_loss=0.0016"
  "fusionfusion-custom16rll-multi-2021-05-07 11-08-freiburgthermal-c3-sord-1,2,3-a1-logl2-rgb,ir-epoch=25-val_loss=0.0036"
)

xp=thermo
arg=$1

for checkpoint in "${fusioncheckpoints[@]}"
do
  for dataset in freiburgthermal
  do
    mkdir -p results/$dataset/$xp/txt
    txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
    echo "$checkpoint"
    isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
    if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
      echo "--> running eval"
      python3 fusion-test.py --bs 1 --fusion custom --modalities rgb,ir --save --save_xp mishmash --decoders multi --fusion_activ softmax --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp $xp --loss_weight > "$txtoutput" 2>&1
    fi
  done
  for dataset in multispectralseg thermalvoc
  do
    mkdir -p results/$dataset/$xp/txt
    txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
    echo "$checkpoint"
    isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
    if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
      echo "--> running eval"
      python3 fusion-test.py  --bs 1 --fusion custom --modalities rgb,ir --save --save_xp mishmash --decoders multi --fusion_activ softmax --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp $xp --loss_weight --test_set full > "$txtoutput" 2>&1
    fi
    echo "--> summary"
    tail -14 "$txtoutput"
  done
done

singlecheckpoints=(
  "2021-05-06 13-48-freiburgthermal-c13-sord-1,2,3-a1-logl2-rgb-epoch=23-val_loss=0.0037"
)

for checkpoint in "${singlecheckpoints[@]}"
do
  for dataset in freiburgthermal
  do
    if [[ $arg == "overlay" ]]; then
      echo "--> generating overlay"
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${singlecheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir --gt
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${singlecheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[1]}_affordances" --rgb --ir
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[1]}_affordances" --rgb --ir --gt
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir --gt
    fi
    mkdir -p results/$dataset/$xp/txt
    txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
    echo "$checkpoint"
    isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
    if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
      echo "--> running eval"
      python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp $xp --modalities rgb --loss_weight > "$txtoutput" 2>&1
      if [[ $arg == "overlay" ]]; then
        echo "--> generating overlay"
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${singlecheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir --gt
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${singlecheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[1]}_affordances" --rgb --ir
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[1]}_affordances" --rgb --ir --gt
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir --gt
      fi
    fi
  done
  for dataset in multispectralseg thermalvoc
  do
    if [[ $arg == "overlay" ]]; then
      echo "--> generating overlay"
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${singlecheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[1]}_affordances" --rgb --ir
      python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir
    fi
    mkdir -p results/$dataset/$xp/txt
    txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
    echo "$dataset $checkpoint"
    isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
    if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
      echo "--> running eval"
      python3 lightning.py --bs 1 --dataset $dataset --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --save --save_xp $xp --modalities rgb --loss_weight > "$txtoutput" 2>&1
      if [[ $arg == "overlay" ]]; then
        echo "--> generating overlay"
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${singlecheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[1]}_affordances" --rgb --ir
        python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${fusioncheckpoints[0]}_affordances" --model2 "${fusioncheckpoints[2]}_affordances" --rgb --ir
      fi
    fi
    echo "--> summary"
    tail -14 "$txtoutput"
  done
done
