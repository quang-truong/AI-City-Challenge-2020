python train_xent_tri.py \
-a resnet92_attention \
-s AIC20_ReID_Simu_Color \
-t AIC20_ReID_Simu_Color \
-j 16 \
--lr-scheduler multi_step \
--lr 0.002 \
--stepsize 5 15 \
--optim sgd \
--gamma 0.5 \
--label-smooth \
--height 224 \
--width 224 \
--lambda-xent 1 \
--lambda-htri 0 \
--lambda-center 0 \
--train-batch-size 72 \
--test-batch-size 100 \
--train-sampler RandomSampler \
--num-instances 6 \
--max-epoch 20 \
--color-jitter \
--color-aug \
--random-erase \
--save-dir log/resnet92_attention-aic20-simu-color \
--no-pretrained \
--eval-freq 1 \
--use-avai-gpus