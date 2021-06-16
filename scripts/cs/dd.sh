python src/compressed_sensing.py --net dd --dataset ffhq-69000 --input-type full-input --num-input-images 1 --batch-size 1 --image-size 256 --measurement-type circulant --noise-std 0.0 --num-measurements 40000 --model-type dd --max-update-iter 20000 --optimizer-type adam --learning-rate 0.04 --momentum 0. --num-random-restarts 1 --checkpoint-iter 1 --cuda 

