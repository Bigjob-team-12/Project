@ECHO ON
title Start

cd _src/data_collection
python data_collection_zooseyo.py

cd ../data_processing
python image_data_download.py

cd ../data_analysis/dog_image_similarity
python predict_dog_data.py
python crawling_to_preprocessed.py

pause


