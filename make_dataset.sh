mkdir data
wget https://zenodo.org/api/records/8265464/files-archive
mkdir data/raw_data
unzip files-archive -d data/raw_data
mkdir -p data/fetal
unzip -o "data/raw_data/Diverse Fetal Head Images.zip" -d data/fetal
unzip -o "data/raw_data/Trans-cerebellum.zip" -d data/fetal
unzip -o "data/raw_data/Trans-thalamic.zip" -d data/fetal
unzip -o "data/raw_data/Trans-ventricular.zip" -d data/fetal
mkdir data/segmentation
mkdir data/segmentation/images
mkdir data/segmentation/labels
cp  "data/fetal/Diverse Fetal Head Images/Orginal_train_images_to_959_661/"*.png data/segmentation/images
cp  "data/fetal/Trans-cerebellum/Trans-cerebellum/"*.png data/segmentation/images
cp  "data/fetal/Trans-thalamic/Trans-thalamic/"*.png data/segmentation/images
cp  "data/fetal/Trans-ventricular/Trans-ventricular/"*.png data/segmentation/images
unzip -o -j "data/fetal/Diverse Fetal Head Images/Test-Dataset-CitySpaces.zip" "gtFine/default/*" -d data/segmentation/labels
unzip -o -j "data/fetal/Trans-cerebellum/Trans-cerebellum-Cityscapes.zip" "gtFine/default/*" -d data/segmentation/labels
unzip -o -j "data/fetal/Trans-thalamic/Trans-thalamic-Cityscapes.zip" "gtFine/default/*" -d data/segmentation/labels
unzip -o -j "data/fetal/Trans-ventricular/Trans-ventricular-CityScapes.zip" "gtFine/default/*" -d data/segmentation/labels
