mkdir -p test_images/cat10dog90/imgs/
cd test_images/cat10dog90/imgs/
shuf -ezn 500 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 55 ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

cd ../../../
mkdir -p test_images/cat20dog80/imgs/
cd test_images/cat20dog80/imgs/
shuf -ezn 500 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 125 ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

cd ../../../
mkdir -p test_images/cat30dog70/imgs/
cd test_images/cat30dog70/imgs/
shuf -ezn 500 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 215 ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

cd ../../../
mkdir -p test_images/cat40dog60/imgs/
cd test_images/cat40dog60/imgs/
shuf -ezn 500 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 333 ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

cd ../../../
mkdir -p test_images/cat50dog50/imgs/
cd test_images/cat50dog50/imgs/
shuf -ezn 500 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 500 ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

cd ../../../
mkdir -p test_images/cat60dog40/imgs/
cd test_images/cat60dog40/imgs/
shuf -ezn 333 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 500 ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

cd ../../../
mkdir -p test_images/cat70dog30/imgs/
cd test_images/cat70dog30/imgs/
shuf -ezn 215 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 500  ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

cd ../../../
mkdir -p test_images/cat80dog20/imgs/
cd test_images/cat80dog20/imgs/
shuf -ezn 125 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 500 ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

cd ../../../
mkdir -p test_images/cat90dog10/imgs/
cd test_images/cat90dog10/imgs/
shuf -ezn 55 ../../../datasets/afhq/val/dog/* | xargs -0 ln -st .
shuf -ezn 500 ../../../datasets/afhq/val/cat/* | xargs -0 ln -st .
list=$(ls * | shuf); set i=0; for name in $list; do new=$(printf "%06d.jpg" "$i"); mv $name $new; let i=i+1; done

