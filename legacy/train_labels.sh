python train_labels.py \
-a resnext101 \
-s AIC20_ReID_Full \
-t AIC20_ReID_Full \
-j 16 \
--lr-scheduler multi_step \
--lr 0.002 \
--stepsize 20 30 \
--optim sgd \
--gamma 0.5 \
--label-smooth \
--height 224 \
--width 224 \
--lambda-xent 1 \
--lambda-htri 0 \
--train-batch-size 72 \
--test-batch-size 100 \
--train-sampler RandomSampler \
--max-epoch 100 \
--color-jitter \
--color-aug \
--random-erase \
--no-pretrained \
--eval-freq 1 \
--load-weights log/resnext101-aic20-color-eval-continued/model.pth.tar-15 \
--extract-features \
--use-avai-gpus