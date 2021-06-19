python src/compressed_sensing.py --checkpoint-path ./checkpoints/glow/graph_unoptimized.pb --net glow --dataset celebA --num-input-images 1 --batch-size 1  --measurement-type circulant --noise-std 16.0 --num-measurements 2500 --model-type langevin --print-stats --checkpoint-iter 1 --cuda --mloss-weight -1 --learning-rate 5e-5 --sigma-init 64 --sigma-final 16 --L 10 --T 200 --zprior-weight -1 --annealed

python src/compressed_sensing.py --checkpoint-path ./checkpoints/glow/graph_unoptimized.pb --net glow --dataset celebA --num-input-images 1 --batch-size 1  --measurement-type circulant --noise-std 16.0 --num-measurements 5000 --model-type langevin --print-stats --checkpoint-iter 1 --cuda --mloss-weight -1 --learning-rate 5e-5 --sigma-init 64 --sigma-final 16 --L 10 --T 200 --zprior-weight -1 --annealed

python src/compressed_sensing.py --checkpoint-path ./checkpoints/glow/graph_unoptimized.pb --net glow --dataset celebA --num-input-images 1 --batch-size 1  --measurement-type circulant --noise-std 16.0 --num-measurements 10000 --model-type langevin --print-stats --checkpoint-iter 1 --cuda --mloss-weight -1 --learning-rate 5e-5 --sigma-init 64 --sigma-final 16 --L 10 --T 200 --zprior-weight -1 --annealed

python src/compressed_sensing.py --checkpoint-path ./checkpoints/glow/graph_unoptimized.pb --net glow --dataset celebA --num-input-images 1 --batch-size 1  --measurement-type circulant --noise-std 16.0 --num-measurements 15000 --model-type langevin --print-stats --checkpoint-iter 1 --cuda --mloss-weight -1 --learning-rate 1e-5 --sigma-init 64 --sigma-final 16 --L 10 --T 200 --zprior-weight -1 --annealed

python src/compressed_sensing.py --checkpoint-path ./checkpoints/glow/graph_unoptimized.pb --net glow --dataset celebA --num-input-images 1 --batch-size 1  --measurement-type circulant --noise-std 16.0 --num-measurements 20000 --model-type langevin --print-stats --checkpoint-iter 1 --cuda --mloss-weight -1 --learning-rate 1e-5 --sigma-init 64 --sigma-final 16 --L 10 --T 200 --zprior-weight -1 --annealed

python src/compressed_sensing.py --checkpoint-path ./checkpoints/glow/graph_unoptimized.pb --net glow --dataset celebA --num-input-images 1 --batch-size 1  --measurement-type circulant --noise-std 16.0 --num-measurements 30000 --model-type langevin --print-stats --checkpoint-iter 1 --cuda --mloss-weight -1 --learning-rate 1e-5 --sigma-init 64 --sigma-final 16 --L 10 --T 200 --zprior-weight -1 --annealed

python src/compressed_sensing.py --checkpoint-path ./checkpoints/glow/graph_unoptimized.pb --net glow --dataset celebA --num-input-images 1 --batch-size 1  --measurement-type circulant --noise-std 16.0 --num-measurements 35000 --model-type langevin --print-stats --checkpoint-iter 1 --cuda --mloss-weight -1 --learning-rate 1e-5 --sigma-init 64 --sigma-final 16 --L 10 --T 200 --zprior-weight -1 --annealed


