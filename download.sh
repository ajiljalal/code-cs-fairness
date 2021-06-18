mkdir -p checkpoints/glow
mkdir datasets

wget -N https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0 -O datasets/afhq.zip
curl https://openaipublic.azureedge.net/glow-demo/large3/graph_unoptimized.pb > checkpoints/glow/graph_unoptimized.pb

unzip datasets/afhq.zip -d ./datasets
