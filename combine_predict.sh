python train_xent_tri.py \
-s AIC20_ReID_Full \
-t AIC20_ReID_Full \
-j 16 \
--combine-predict \
new-glamor50_v1-AIC20_ReID_Full-distmat.pkl \
new-glamor50_nv1-AIC20_ReID_Full-distmat.pkl \
--combine-lambda 1.0 1.0 \
--use-avai-gpus \
--save-dir log/ranked_results/Best_Result \
--no-pretrained