python train_xent_tri.py \
-a glamor50_nv1 \
-s AIC20_ReID \
-t AIC20_ReID \
-j 16 \
--lr-scheduler multi_step \
--lr 0.0003 \
--stepsize 5 25 50 \
--gamma 0.3 \
--label-smooth \
--height 128 \
--width 256 \
--margin 0.3 \
--lambda-xent 1 \
--lambda-htri 10 \
--lambda-center 0 \
--train-batch-size 32 \
--train-sampler RandomIdentitySampler \
--num-instances 4 \
--random-erase \
--color-jitter \
--color-aug \
--max-epoch 80 \
--load-weights log/weights/glamor50_nv1-aic20-simu-vehicleID-v1/model.pth.tar-25 \
--save-dir log/new-glamor50_nv1-aic20-eval \
--no-pretrained \
--eval-freq 1 \
--print-freq 200 \
--gpu-devices 1