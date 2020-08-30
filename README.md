# 이미지 유사도 기반 실종 반려동물 찾기

- 진행 기간: 2020. 07 ~ 2020. 09

- 프로젝트 상세
  - 인터넷에 산재되어 있는 유기동물 보호 정보를 취합하여 이미지의 <b>시각적 유사도 기반</b> 검색/ 노출.
  - 공고 건 중 찾고자 하는 반려동물과 <b>가장 유사한 동물</b>을 우선적으로 노출, 또는 신규 공고 등록 시 유사도 확인하여 알림.
  - <b>주기적으로</b> 신규 데이터 수집-전처리-적재-임베딩 및 등록 중인 query 이미지와 유사도 비교 후 push까지 <b>자동화</b>. 
  - 데이터:
    > for 모델 학습 - 품종 등 카테고리 labeled 된 배포 데이터셋 출처 다수  
    > for 시스템 실사용 - 농림부 동물보호관리시스템, 유기동물네트워크 게시판 개인 업로드, SNS 등
  - 모델 keywords:
    > Metric Learning (e.g. triplet/ center/ large margin loss)  
    > DeepRanking, Attention, GradCAM  
    > Image Retrieval
  - 기타 고려사항:
    > 눈 색깔, 무늬 형태 등 도메인 특화 피처들의 반영 가능성  
    > 야외 생활로 인한, 유기 기간에 따른 외형적 변화
    
#### System
```Python
$ main_project
│
├─ _db
│   └─ data
│       ├─ crawling_data
│       │       └─ [개]  #  Directory for crawled images. When the image processing is complete, go [preprocessed_data] directory
│       ├─ model_data
│       │       ├─ gallery
│       │       │     └─ gallery_list # Temp directory for gallery images(option) 
│       │       ├─ query
│       │       │     └─ query_list  # Temp directory for query image
│       │       └─ working
│       │             ├─ dog_breeds.txt  # Image statistics used in training 
│       │             ├─ result.csv   #   
│       │             ├─ to_reid.csv   # List of images that have passed Pearson correlation coefficient, region, and date filters
│       │             ├─ to_web.csv   # List of images sent to the web
│       │             └─ tmp.h5   # weights for classification model 
│       └─ preprocessed_data
│               └─ dog_class ─ image   # Preprocessed image using YOLO-v4
└─ _src
     ├─ batch
     │     └─ update_data.bat   # Batch file for crawling and Preprocessing 'post' images.    
     ├─ data_analysis
     │     ├─ dog_image_similarity
     │     │          ├─ copy_image.py
     │     │          ├─ crawling_to_preprocessed.py
     │     │          ├─ dog_breed_similarity_comparison.py
     │     │          ├─ extract_similar_image_path.py
     │     │          └─ predict_dog_data.py
     │     └─ re_id
     │           └─ code
     │                ├─ reid_query.py   # Extract query image feature (512 vectors)
     │                ├─ reid_gallery.py  # Extract gallery image feature (512 vectors)
     │                ├─ reid_sort.py   # Calculate image similarity using cosine distance and sorting index   
     │                ├─ train.py   # To train model 
     │                └─ model.py   # Model structure for train.py
     │                
     ├─ data_analysis
     │     └─ data_collection_zooseyo.py
     ├─ data_processing
     │     └─ image_data_download.py # Code for downloading image 
     │     └─ yolo_v4
     │            └─  detect.py   # Detect dogs and cats. Crop target image and save 
     └─ web
         ├─ static
         │     ├─ bootstrap
         │     └─ images
         │           ├─ input_image
         │           └─ uploads
         ├─ templates
         │      ├─ find_my_dog_a.html
         │      ├─ find_my_dog_q.html
         │      └─ index.html   # Main page
         ├─ app.py
         └─ database.sqlite  # DB 
```


#### 프로젝트 진행 과정

- **`200706-200710`**
  - 아이디어 회의
- **`200713-200717`**
  - 아이디어 회의
  - 팀별 발표(변리사 & 학생 피드백)
- **`200720-200724`**
  - 환경 구축
  - 데이터 수집
    - [종합 유기견 보호센터](http://www.zooseyo.or.kr/zooseyo_or_kr.html?) : 유기 동물(공고 & 보호 중)
- **`200727-200731`**
  - 수집된 데이터 EDA
- **`200803-200807`**
  - Data preprocessing code 작성 
    - File renaming & random shuffle, Cross validation, Image Augmentation 
  - pet re-identification
    - [person re identification](https://github.com/waylybaye/Person_reID_baseline_pytorch#dataset--preparation) 응용 
  - 개 품종 분류 및 결과 EDA
- **`200810-200814`**
  - 분류 모델 softmax값을 이용한 cosine 유사도 측정
  - 임의의 이미지에 대해 공고 이미지와 유사도 측정 후 유사한 이미지 N개 출력
- **`200817-200821`**
  - 웹 디자인 구상
  - Object detection & Image crop하는 code 작성 (Using faster R-CNN)   
- **`200824-200831`**
  - flow diagram 작성
  - Demo 용 웹 페이지 생성(flask, DB 연동)
  - 분류기와 re-id 연결
  - Object detection & Image crop(이미지 전처리)에 YOLO-V4 사용
  - 날짜, 지역 필터링 적용
  - 이메일 알림 기능 추가
