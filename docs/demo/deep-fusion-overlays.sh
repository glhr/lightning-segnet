## Freiburg

xp=coolfusion
dataset=freiburg
arg=$1

if [[ $arg == "nogpu" ]]; then
  gpu="--gpus 0"
else
  gpu=""
fi

run_single() {
    echo "--> generating overlay"
    mods=$1
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoints[0]}_affordances" ${mods}
}



run_comp() {
    echo "--> generating overlay"
    mods=$1
    python3 overlay_imgs.py --dataset $dataset --xp $xp --model "${checkpoints[0]}_affordances" --model2 "${checkpoints[1]}_affordances" ${mods}
}


# checkpoints=(
# "2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474"
# "fusionfusion-custom16-multi-2021-04-20 19-24-freiburg-c3-kl-rgb,depth-epoch=141-val_loss=0.1369"
# "fusionfusion-custom16-multi-2021-04-22 14-55-freiburg-c3-kl-rgb,ir-epoch=93-val_loss=0.1186"
# "fusionfusion-custom16-multi-2021-04-24 13-20-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1147"
# )
# run_comp "--prefix modcomp-dualcustom"
# run_single "--gt --rgb --depth --ir --nopred"

# checkpoints=(
# "2021-04-17 18-57-freiburg-c6-kl-rgb,depth-epoch=87-val_loss=0.1472"
# "fusionfusion-custom16-single-2021-04-20 18-31-freiburg-c3-kl-rgb,depth-epoch=43-val_loss=0.1429"
# "fusionfusion-custom16-late-2021-04-29 18-41-freiburg-c3-kl-rgb,depth-epoch=39-val_loss=0.1488"
# "fusionfusion-custom16-multi-2021-04-20 19-24-freiburg-c3-kl-rgb,depth-epoch=141-val_loss=0.1369"
# )
# run_comp "--prefix rgb,d-archcomp-custom"
# run_single "--gt --rgb --depth --ir --nopred"
#
# checkpoints=(
# "2021-04-17 18-57-freiburg-c6-kl-rgb,depth-epoch=87-val_loss=0.1472"
# "fusionfusion-ssma16-single-2021-04-26 06-40-freiburg-c3-kl-rgb,depth-epoch=75-val_loss=0.1338"
# "fusionfusion-ssma16-late-2021-04-30 18-58-freiburg-c3-kl-rgb,depth-epoch=119-val_loss=0.1430"
# "fusionfusion-ssma16-multi-2021-04-26 07-35-freiburg-c3-kl-rgb,depth-epoch=102-val_loss=0.1500"
# )
# run_comp "--prefix rgb,d-archcomp-ssma"
#
# checkpoints=(
# "2021-04-17 20-25-freiburg-c6-kl-rgb,depth,ir-epoch=81-val_loss=0.1352"
# "fusionfusion-custom16-single-2021-04-24 18-29-freiburg-c3-kl-rgb,depth,ir-epoch=103-val_loss=0.1281"
# "fusionfusion-custom16-late-2021-04-29 17-09-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1169"
# "fusionfusion-custom16-multi-2021-04-24 13-20-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1147"
# )
# run_comp "--prefix rgb,d,ir-archcomp-custom"

# checkpoints=(
# "2021-04-17 20-25-freiburg-c6-kl-rgb,depth,ir-epoch=81-val_loss=0.1352"
# "fusionfusion-custom16-single-2021-04-24 18-29-freiburg-c3-kl-rgb,depth,ir-epoch=103-val_loss=0.1281"
# "fusionfusion-custom16rll-late-2021-04-30 14-50-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1150"
# "fusionfusion-custom16rll-multi-2021-04-24 16-46-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1102"
# )
# run_comp "--prefix rgb,d,ir-archcomp-customrll"

checkpoints=(
"fusionfusion-ssma16-multi-2021-04-25 20-49-freiburg-c3-kl-rgb,depth,ir-epoch=125-val_loss=0.1195"
"fusionfusion-custom16rll-multi-2021-04-24 16-46-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1102"
)
run_comp "--prefix unitcomp-middual --gt"

unit=none
checkpoints_baselines=(
"2021-04-08 13-31-freiburg-c6-kl-rgb-epoch=43-val_loss=0.1474"
"2021-04-17 18-57-freiburg-c6-kl-rgb,depth-epoch=87-val_loss=0.1472"
"2021-04-17 19-40-freiburg-c6-kl-rgb,ir-epoch=149-val_loss=0.1349"
"2021-04-17 20-25-freiburg-c6-kl-rgb,depth,ir-epoch=81-val_loss=0.1352"
)
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
checkpoints_customRLL=(
"fusionfusion-custom16rll-multi-2021-04-23 12-22-freiburg-c3-kl-rgb,depth-epoch=102-val_loss=0.1522"
"fusionfusion-custom16rll-late-2021-04-30 10-43-freiburg-c3-kl-rgb,depth-epoch=104-val_loss=0.1485"

"fusionfusion-custom16rll-multi-2021-04-23 09-27-freiburg-c3-kl-rgb,ir-epoch=148-val_loss=0.1258"
"fusionfusion-custom16rll-late-2021-04-30 12-04-freiburg-c3-kl-rgb,ir-epoch=104-val_loss=0.1262"

"fusionfusion-custom16rll-multi-2021-04-24 16-46-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1102"
"fusionfusion-custom16rll-late-2021-04-30 14-50-freiburg-c3-kl-rgb,depth,ir-epoch=62-val_loss=0.1150"
)

## Cityscapes

dataset=cityscapes

checkpoints=(
"2021-04-18 10-12-cityscapes-c30-kl-rgb,depthraw-epoch=23-val_loss=0.1024"
"fusionfusion-custom16-single-2021-04-21 07-16-cityscapes-c3-kl-rgb,depthraw-epoch=2-val_loss=0.0920"
"fusionfusion-custom16-late-2021-04-29 22-42-cityscapes-c3-kl-rgb,depthraw-epoch=13-val_loss=0.0883"
"fusionfusion-custom16-multi-2021-04-21 00-22-cityscapes-c3-kl-rgb,depthraw-epoch=23-val_loss=0.0873"
)
run_single "--gt --rgb --depthraw --nopred"
run_single "--gt --rgb --depth --depthraw --nopred"
run_comp "--prefix rgb,draw-archcomp-custom"

checkpoints=(
"fusionfusion-custom16-late-2021-04-29 22-42-cityscapes-c3-kl-rgb,depthraw-epoch=13-val_loss=0.0883"
"fusionfusion-custom16-late-2021-04-30 07-53-cityscapes-c3-kl-rgb,depth-epoch=5-val_loss=0.0878"
"fusionfusion-custom16-multi-2021-04-21 00-22-cityscapes-c3-kl-rgb,depthraw-epoch=23-val_loss=0.0873"
"fusionfusion-custom16-multi-2021-04-21 14-31-cityscapes-c3-kl-rgb,depth-epoch=5-val_loss=0.0876"
)
run_comp "--prefix depthcomp-latedual-custom"

checkpoints=(
"fusionfusion-ssma16-multi-2021-05-01 15-44-cityscapes-c3-kl-rgb,depthraw-epoch=24-val_loss=0.0876"
"fusionfusion-custom16rll-multi-2021-04-23 10-38-cityscapes-c3-kl-rgb,depthraw-epoch=13-val_loss=0.0888"
)
run_comp "--prefix unitcomp-middual --gt"

checkpoints_baselines=(
"2021-04-09 03-40-cityscapes-c30-kl-rgb-epoch=18-val_loss=0.0918"
"2021-04-18 10-12-cityscapes-c30-kl-rgb,depthraw-epoch=23-val_loss=0.1024"
"2021-04-18 00-48-cityscapes-c30-kl-rgb,depth-epoch=23-val_loss=0.0999"
)
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
