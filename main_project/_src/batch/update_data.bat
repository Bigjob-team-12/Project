@ECHO ON
title Start

cd ../data_collection
python data_collection_zooseyo.py

cd ../data_processing
python image_data_download.py

cd ../data_analysis/dog_image_similarity
python predict_dog_data.py
python crawling_to_preprocessed.py

cd ../re_id/code
python reid_gallery.py

cd ../../../batch
python send_email.py

pause