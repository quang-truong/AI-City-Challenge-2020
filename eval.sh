python train_xent_tri.py \
-s AIC20_ReID_CamID \
-t AIC20_ReID_CamID \
-a resnet92_attention \
-j 16 \
--height 224 \
--width 224 \
--evaluate \
--load-weights log/resnet92_attention-aic20-camid/model.pth.tar-67 \
--no-pretrained
