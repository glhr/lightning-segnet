## Freiburg

xp=coolfusion
dataset=freiburg
arg=$1


checkpoints_ssma=(
"fusionfusion-ssma16-single-2021-04-26 06-40-freiburg-c3-kl-rgb,depth-epoch=75-val_loss=0.1338"
"fusionfusion-ssma16-multi-2021-04-26 07-35-freiburg-c3-kl-rgb,depth-epoch=102-val_loss=0.1500"
"fusionfusion-ssma16-late-2021-04-30 18-58-freiburg-c3-kl-rgb,depth-epoch=119-val_loss=0.1430"

"fusionfusion-ssma16-single-2021-04-26 09-56-freiburg-c3-kl-rgb,ir-epoch=79-val_loss=0.1066"
"fusionfusion-ssma16-multi-2021-04-26 08-44-freiburg-c3-kl-rgb,ir-epoch=80-val_loss=0.1199"
"fusionfusion-ssma16-late-2021-04-30 20-05-freiburg-c3-kl-rgb,ir-epoch=119-val_loss=0.1183"

"fusionfusion-ssma16-single-2021-04-25 22-30-freiburg-c3-kl-rgb,depth,ir-epoch=129-val_loss=0.1145"
"fusionfusion-ssma16-multi-2021-04-25 20-49-freiburg-c3-kl-rgb,depth,ir-epoch=125-val_loss=0.1195"
"fusionfusion-ssma16-late-2021-04-30 16-29-freiburg-c3-kl-rgb,depth,ir-epoch=64-val_loss=0.1264"
)


eval_ssma() {
  python3 fusion-test.py  --bs 1 --fusion $unit --dataset $dataset --modalities $modalities --save --bs 1 --save_xp $xp --decoders $decoders --test_checkpoint "lightning_logs/${checkpoint}.ckpt" > "$txtoutput" 2>&1
}

eval_custom() {
  python3 fusion-test.py  --bs 1 --fusion custom --dataset $dataset --modalities $modalities --save --bs 1 --save_xp $xp --decoders $decoders --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --fusion_activ softmax --pretrained_last_layer > "$txtoutput" 2>&1
}

eval_customrll() {
  python3 fusion-test.py  --bs 1 --fusion custom --dataset $dataset --modalities $modalities --save --bs 1 --save_xp $xp --decoders $decoders --test_checkpoint "lightning_logs/${checkpoint}.ckpt" --fusion_activ softmax > "$txtoutput" 2>&1
}


run() {
  mkdir -p results/$dataset/$xp/txt
  txtoutput="results/${dataset}/${xp}/txt/${checkpoint}.txt"
  echo "$checkpoint"
  isInFile=$(cat "$txtoutput" | grep -c "DATALOADER:0 TEST RESULTS")
  if [ ! -f "$txtoutput" ] || [ $isInFile -eq 0 ] ; then
    echo "--> running eval"
    case "$unit" in
      'ssma')
          eval_ssma
          ;;
      'custom')
          eval_custom
          ;;
      'customrll')
          eval_customrll
          ;;
      esac
    python3 fusion-test.py  --bs 1 --fusion $unit --dataset $dataset --modalities $modalities --save --bs 1 --save_xp $xp --decoders $decoders --test_checkpoint "lightning_logs/${checkpoint}.ckpt" > "$txtoutput" 2>&1
  fi
  echo "--> summary"
  tail -14 "$txtoutput"
  if [[ $arg == "overlay" ]]; then
    echo "--> generating overlay"
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoints[0]}_affordances" --model2 "${checkpoints[1]}_affordances" --gt
  fi
}

unit=ssma
modalities="rgb,depth"

decoders=single
checkpoint="fusionfusion-ssma16-single-2021-04-26 06-40-freiburg-c3-kl-rgb,depth-epoch=75-val_loss=0.1338"
run
decoders=multi
checkpoint="fusionfusion-ssma16-multi-2021-04-26 07-35-freiburg-c3-kl-rgb,depth-epoch=102-val_loss=0.1500"
run
decoders=late
checkpoint="fusionfusion-ssma16-late-2021-04-30 18-58-freiburg-c3-kl-rgb,depth-epoch=119-val_loss=0.1430"
run

modalities="rgb,ir"
decoders=single
checkpoint="fusionfusion-ssma16-single-2021-04-26 09-56-freiburg-c3-kl-rgb,ir-epoch=79-val_loss=0.1066"
run
decoders=multi
checkpoint="fusionfusion-ssma16-multi-2021-04-26 08-44-freiburg-c3-kl-rgb,ir-epoch=80-val_loss=0.1199"
run
decoders=late
checkpoint="fusionfusion-ssma16-late-2021-04-30 20-05-freiburg-c3-kl-rgb,ir-epoch=119-val_loss=0.1183"
run

modalities="rgb,depth,ir"
decoders=single
checkpoint="fusionfusion-ssma16-single-2021-04-25 22-30-freiburg-c3-kl-rgb,depth,ir-epoch=129-val_loss=0.1145"
run
decoders=multi
checkpoint="fusionfusion-ssma16-multi-2021-04-25 20-49-freiburg-c3-kl-rgb,depth,ir-epoch=125-val_loss=0.1195"
run
decoders=late
checkpoint="fusionfusion-ssma16-late-2021-04-30 16-29-freiburg-c3-kl-rgb,depth,ir-epoch=64-val_loss=0.1264"
run

unit=custom

modalities="rgb,depth"
decoders=single
checkpoint="fusionfusion-custom16-single-2021-04-20 18-31-freiburg-c3-kl-rgb,depth-epoch=43-val_loss=0.1429"
run
decoders=multi
checkpoint="fusionfusion-custom16-multi-2021-04-20 19-24-freiburg-c3-kl-rgb,depth-epoch=141-val_loss=0.1369"
run
decoders=late
checkpoint="fusionfusion-custom16-late-2021-04-29 18-41-freiburg-c3-kl-rgb,depth-epoch=39-val_loss=0.1488"
run

modalities="rgb,ir"
decoders=single
checkpoint="fusionfusion-custom16-single-2021-04-22 16-04-freiburg-c3-kl-rgb,ir-epoch=120-val_loss=0.1227"
run
decoders=multi
checkpoint="fusionfusion-custom16-multi-2021-04-22 14-55-freiburg-c3-kl-rgb,ir-epoch=93-val_loss=0.1186"
run
decoders=late
checkpoint="fusionfusion-custom16-late-2021-04-29 19-48-freiburg-c3-kl-rgb,ir-epoch=104-val_loss=0.1244"
run

modalities="rgb,depth,ir"
decoders=single
checkpoint="fusionfusion-custom16-single-2021-04-24 18-29-freiburg-c3-kl-rgb,depth,ir-epoch=103-val_loss=0.1281"
run
decoders=multi
checkpoint="fusionfusion-custom16-multi-2021-04-24 13-20-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1147"
run
decoders=late
checkpoint="fusionfusion-custom16-late-2021-04-29 17-09-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1169"
run

checkpoints_custom=(
"fusionfusion-custom16-single-2021-04-20 18-31-freiburg-c3-kl-rgb,depth-epoch=43-val_loss=0.1429"
"fusionfusion-custom16-multi-2021-04-20 19-24-freiburg-c3-kl-rgb,depth-epoch=141-val_loss=0.1369"
"fusionfusion-custom16-late-2021-04-29 18-41-freiburg-c3-kl-rgb,depth-epoch=39-val_loss=0.1488"

"fusionfusion-custom16-single-2021-04-22 16-04-freiburg-c3-kl-rgb,ir-epoch=120-val_loss=0.1227"
"fusionfusion-custom16-multi-2021-04-22 14-55-freiburg-c3-kl-rgb,ir-epoch=93-val_loss=0.1186"
"fusionfusion-custom16-late-2021-04-29 19-48-freiburg-c3-kl-rgb,ir-epoch=104-val_loss=0.1244"

"fusionfusion-custom16-single-2021-04-24 18-29-freiburg-c3-kl-rgb,depth,ir-epoch=103-val_loss=0.1281"
"fusionfusion-custom16-multi-2021-04-24 13-20-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1147"
"fusionfusion-custom16-late-2021-04-29 17-09-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1169"
)

unit=customrll

modalities="rgb,depth"
decoders=multi
checkpoint="fusionfusion-custom16rll-multi-2021-04-23 12-22-freiburg-c3-kl-rgb,depth-epoch=102-val_loss=0.1522"
run
decoders=late
checkpoint="fusionfusion-custom16rll-late-2021-04-30 10-43-freiburg-c3-kl-rgb,depth-epoch=104-val_loss=0.1485"
run

modalities="rgb,ir"
decoders=multi
checkpoint="fusionfusion-custom16rll-multi-2021-04-23 09-27-freiburg-c3-kl-rgb,ir-epoch=148-val_loss=0.1258"
run
decoders=late
checkpoint="fusionfusion-custom16rll-late-2021-04-30 12-04-freiburg-c3-kl-rgb,ir-epoch=104-val_loss=0.1262"
run

modalities="rgb,depth,ir"
decoders=multi
checkpoint="fusionfusion-custom16rll-multi-2021-04-24 16-46-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1102"
run
decoders=late
checkpoint="fusionfusion-custom16rll-late-2021-04-30 14-50-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1150"
run

checkpoints_customRLL=(
"fusionfusion-custom16rll-multi-2021-04-23 12-22-freiburg-c3-kl-rgb,depth-epoch=102-val_loss=0.1522"
"fusionfusion-custom16rll-late-2021-04-30 10-43-freiburg-c3-kl-rgb,depth-epoch=104-val_loss=0.1485"

"fusionfusion-custom16rll-multi-2021-04-23 09-27-freiburg-c3-kl-rgb,ir-epoch=148-val_loss=0.1258"
"fusionfusion-custom16rll-late-2021-04-30 12-04-freiburg-c3-kl-rgb,ir-epoch=104-val_loss=0.1262"

"fusionfusion-custom16rll-multi-2021-04-24 16-46-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1102"
"fusionfusion-custom16rll-late-2021-04-30 14-50-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1150"
)

## Cityscapes

checkpoints_ssma=(
"fusionfusion-ssma16-multi-2021-05-01 15-44-cityscapes-c3-kl-rgb,depthraw-epoch=24-val_loss=0.0876"
"fusionfusion-ssma16-single-2021-05-01 09-20-cityscapes-c3-kl-rgb,depthraw-epoch=17-val_loss=0.0901"
"fusionfusion-ssma16-late-2021-05-01 02-36-cityscapes-c3-kl-rgb,depthraw-epoch=21-val_loss=0.0879"

"fusionfusion-ssma16-multi-2021-05-01 18-18-cityscapes-c3-kl-rgb,depth-epoch=23-val_loss=0.0904"
"fusionfusion-ssma16-single-2021-04-26 10-54-cityscapes-c3-kl-rgb,depth-epoch=24-val_loss=0.0884"
"fusionfusion-ssma16-late-2021-05-01 00-55-cityscapes-c3-kl-rgb,depth-epoch=22-val_loss=0.0872"
)

checkpoints_custom=(
"fusionfusion-custom16-multi-2021-04-21 00-22-cityscapes-c3-kl-rgb,depthraw-epoch=23-val_loss=0.0873"
"fusionfusion-custom16-single-2021-04-21 07-16-cityscapes-c3-kl-rgb,depthraw-epoch=2-val_loss=0.0920"
"fusionfusion-custom16-late-2021-04-29 22-42-cityscapes-c3-kl-rgb,depthraw-epoch=13-val_loss=0.0883"

"fusionfusion-custom16-multi-2021-04-21 14-31-cityscapes-c3-kl-rgb,depth-epoch=5-val_loss=0.0876"
"fusionfusion-custom16-single-2021-04-21 16-17-cityscapes-c3-kl-rgb,depth-epoch=17-val_loss=0.0901"
"fusionfusion-custom16-late-2021-04-30 07-53-cityscapes-c3-kl-rgb,depth-epoch=5-val_loss=0.0878"
)

checkpoints_customRLL=(
"fusionfusion-custom16rll-multi-2021-04-23 10-38-cityscapes-c3-kl-rgb,depthraw-epoch=13-val_loss=0.0888"
"fusionfusion-custom16rll-late-2021-04-30 23-13-cityscapes-c3-kl-rgb,depthraw-epoch=13-val_loss=0.0891"

"fusionfusion-custom16rll-multi-2021-04-24 05-51-cityscapes-c3-kl-rgb,depth-epoch=7-val_loss=0.0880"
"fusionfusion-custom16rll-late-2021-04-30 21-32-cityscapes-c3-kl-rgb,depth-epoch=20-val_loss=0.0892"
)
