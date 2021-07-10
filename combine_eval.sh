python train_xent_tri.py \
-s AIC20_ReID \
-t AIC20_ReID \
-j 16 \
--combine-eval-dist \
resnext101-AIC20_ReID-distmat-55.pkl \
resnext101-AIC20_ReID-distmat-43.pkl \
resnext101-AIC20_ReID-distmat-39.pkl \
--combine-lambda 1.0 1.0 0.0 \
--use-avai-gpus \
--no-pretrained