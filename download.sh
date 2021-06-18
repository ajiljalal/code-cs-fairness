mkdir -p checkpoints/glow
mkdir datasets

# afhq dataset
wget -N https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0 -O datasets/afhq.zip
# glow checkpoint
curl https://openaipublic.azureedge.net/glow-demo/large3/graph_unoptimized.pb > checkpoints/glow/graph_unoptimized.pb
# test images
gdown https://drive.google.com/uc?id=1dQGyLXYv7OPOFgHJ8Va0vYnqrIqJeXz3
# ncsnv2 checkpoint
gdown https://drive.google.com/uc?id=1jDcIxI0nqv6AOBhHpgbH7OIW63QU77lh
# stylegan checkpointss
gdown https://drive.google.com/uc?id=1I4KtMsW61nRI6TOaadpA5m_6Ygd4QrgR

# extract stuff
unzip datasets/afhq.zip -d ./datasets
tar -zxvf test_images.tar.gz

# accidentally included broken symoblic links in the tar archive, so
# delete folders before shuffling and creating cat/dog validation data
# over different biases
rm -r test_images/cat*
bash shuffle_catdog.sh

# extract ncsnv2 and stylegan2 checkpoints
tar -zxvf ncsnv2_checkpoint.tar.gz
tar -zxvf stylegan2_checkpoints.tar.gz

