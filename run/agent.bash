name=agent
flag="--attn soft --train listener
      --features rn50x4
      --feature_size 640
      --batchSize 64
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --option_size 8
      --option_step 3
      --entropyCoef 0.01
      --criticLr 1e-4
      --seed 123
      --subout max --dropout 0.2 --optim adam --lr 1e-4 --iters 200000 --maxAction 15"
# Uncomment to fix random seed for reproducibility:
# flag="$flag --seed 123"
# flag="$flag --fusionProj"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 CUDA_LAUNCH_BLOCKING=1 python r2r_src/train.py $flag --name $name
