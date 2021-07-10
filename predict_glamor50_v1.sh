python train_xent_tri.py \
-s AIC20_ReID_Full \
-t AIC20_ReID_Full \
-a glamor50_v1 \
-j 16 \
--predict \
--load-weights log/weights/new-glamor50_v1-aic20-eval/model.pth.tar-17 \
--use-avai-gpus \
--no-pretrained