python predict_labels.py \
-s AIC20_ReID_Full \
-t AIC20_ReID_Full \
-a resnet92_attention \
-j 16 \
--height 224 \
--width 224 \
--load-weights log/resnet92_attention-aic20-camid/model.pth.tar-67 \
--no-pretrained