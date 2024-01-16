rm -rf checkpoints
mkdir checkpoints
cd checkpoints
echo -e "Downloading extractors"
gdown --fuzzy https://drive.google.com/file/d/1o7RTDQcToJjTm9_mNWTyzvZvjTWpZfug/view

unzip t2m.zip

echo -e "Cleaning\n"
rm t2m.zip
cd ..
echo -e "Downloading done!"

echo -e "Downloading glove (in use by the evaluators)"
gdown --fuzzy https://drive.google.com/file/d/1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6/view?usp=sharing
rm -rf glove

unzip glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"

mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smpl/' folder\n"
gdown 1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip

echo -e "Downloading done!"

cd ..