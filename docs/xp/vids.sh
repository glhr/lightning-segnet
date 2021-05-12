ffmpeg -y -r 15 -f image2 -s 1920x1080 -i results/synthia/combo/overlay/synthiaOmni_F_%06d-combo-Rgb-pred_overlay.png -vcodec libx264 -crf 25  -pix_fmt yuv420p results/synthia/synthia-combo-test.mp4
