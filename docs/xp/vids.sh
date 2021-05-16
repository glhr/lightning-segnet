ffmpeg -y -r 15 -f image2 -s 1920x1080 -i results/synthia/combo/overlay/synthiaOmni_F_%06d-combo-Rgb-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p results/synthia/synthia-combo-test.mp4

ffmpeg -y -r 40 -f image2 -s 1920x1080 -i results/kaistped/demo-s$set-$v/overlay/kaistped%05d-demo-s$set-$v-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p results/kaistped/kaistped-s$set-$v.mp4

ffmpeg -y -r 40 -f image2 -s 1920x1080 -i results/freiburthermal/thermo/overlay/kaistped%05d-demo-s$set-$v-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p results/kaistped/kaistped-s$set-$v.mp4

ffmpeg -r 8 -f image2 -pattern_type glob -i "results/freiburgthermal/${xp}/overlayRgbIr-_${checkpoint}_affordances/${dataset}*-RgbIr-pred_overlay.png" -c:v libx264 -qp 0 "results/$dataset/${checkpoint}.mp4"
