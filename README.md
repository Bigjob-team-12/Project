# 이미지 유사도 기반 실종 반려동물 찾기

- 진행 기간: 2020. 07 ~ 2020. 09

####프로젝트 상세
  - 인터넷에 산재되어 있는 유기동물 보호 정보를 취합하여 공고 이미지의 <b>시각적 유사도 기반</b> 추출합니다.
  - <b>Fine-grained Classification</b>을 통해 생김새를 파악합니다. 외견적 특성을 정의할 수 있는 세부 품종 분류를 수행한 후 품종별 확률값을 도출합니다. 
  - 도출된 확률값을 바탕으로 탐색 대상과 <b>Query Image의 Correlation </b>분석을 수행, 유사성이 낮은 개체를 후보군에서 제외합니다.
  - <b>Metric Learning</b>을 통해 털의 색, 조합의 유사성을 확인합니다. 반려동물의 털은 여러 색의 혼합으로 구성된 경우가 많습니다. 
  - 신규 등록 개체 확인은 주기적으로 자동 진행되며, 사용자 요청 시 기존 Query 이미지와 비교를 지속 수행하여 새로이 유사한 개체가 발견되었을 때 보호자에게 <b>알림</b>을 보냅니다.
  - 데이터:
    > for 모델 학습 - 품종 등 카테고리 labeled 된 배포 데이터셋 출처 다수  
    > for 시스템 실사용 - 농림부 동물보호관리시스템, 유기동물네트워크 게시판 개인 업로드, SNS 등
  - 모델 keywords:
    > Metric Learning 
    > DeepRanking, Attention
    > Image Retrieval
  - 기타 고려사항:
    > 눈 색깔, 무늬 형태 등 도메인 특화 피처들의 반영 가능성  
    > 야외 생활로 인한, 유기 기간에 따른 외형적 변화
    
    <b>특정 개체 특성 중심 탐색</b>
    > 품종 등 메타데이터를 바탕으로 한 단순 검색을 넘어 개체의 시각적 특성을 바탕으로 한 유사성 비교를 수행합니다. 명확한 품종 분류가 어려우며 보호동물의 대부분을 차지하는 믹스견 역시 세부적인 외견적 특징을 분석하여 가장 유사한 개체를 찾아냅니다.

    <b>파이프라인 자동화</b>
    > 반려동물을 잃어버린 보호자는 커뮤니티 게시판과 오프라인 전단지를 오가며 탐색을 수행하게 됩니다. 보호소에 신규 등록되는 보호 동물을 전부 확인할 수 있다면 좋겠지만, 하루에도 300건 이상 등록되는 공고를 모두 확인하는 것은 쉽지 않습니다. 본 서비스는 주기적으로 신규 등록 개체를 자동 확인하여 기존 Query 개체와 비교하며, 유사도가 높은 개체가 발견될 경우 알림을 발신하여 보호자의 탐색 부담을 줄여줍니다.

    <b>Fine-Grained Classification</b>
    > 개인지 고양이인지 판단하는 단순 분류가 아닌 포메라니안, 골든 리트리버와 같은 세부적인 품종의 외견 정보를 바탕으로 한 분류를 수행합니다. 각 품종은 얼핏 보기에는 유사하지만 애견협회에서 별도 기준을 세울 만큼 뚜렷한 차이를 가집니다. 또한 여러 품종의 특성을 가진 믹스견을 고려하여, 무슨 품종인지가 아닌 외형적으로 추측되는 각 품종의 확률값을 도출합니다.

    <b>Metric Learning</b>
    > 반려동물의 털은 단일 컬러가 아닌 여러 색의 혼합으로 구성된 경우가 많습니다. 본 서비스는 유사한 색상과 조합을 가진 개체를 찾기 위해 Query 이미지와 검색 대상 이미지의 특성을 추출한 후, Similarity를 계산합니다. 이 때 Similarity 계산을 위해 Metric Learning을 적용하였습니다.

    <b>앙상블(stacking)</b>
    > 생김새나 털의 색 배합, 둘 중 하나만으로 특정 개체를 찾기는 어렵습니다. 이목구비 형태의 유사성과 털의 색, 무늬 등 패턴의 유사성을 모두 확인하여 반영하기 위해 두 모델을 연결하여 Stacking 기법을 적용했습니다.

    
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
│       │             ├─ result.csv   # List of image softmax by breeds
│       │             ├─ to_reid.csv   # List of images that have passed Pearson correlation coefficient, region, and date filters
│       │             ├─ to_web.csv   # List of images sent to the web
│       │             └─ tmp.h5   # weights for classification model 
│       └─ preprocessed_data
│               └─ dog_class ─ image   # Preprocessed image using YOLO-v4
└─ _src
     ├─ batch
     │     ├─ send_email.py     # Check updated DB and send e-mail
     │     └─ update_data.bat   # Batch file for crawling and Preprocessing 'post' images.    
     ├─ data_analysis
     │     ├─ dog_image_similarity
     │     │          ├─ copy_image.py # Copy the file from the input path to the output path
     │     │          ├─ crawling_to_preprocessed.py # After preprocessing, train the model and move the file
     │     │          ├─ dog_breed_similarity_comparison.py # Image similarity comparison by pearson correlation
     │     │          ├─ extract_similar_image_path.py # Filtering images through breed classifier
     │     │          └─ predict_dog_data.py # Prediction by image
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
         │     └─ assets
         │           └─ img # Image and Icon for Web Design
         │     ├─ bootstrap # Bootstrap CSS, JS Files
         │     ├─ css # CSS files for Web Design
         │     ├─ js # JavaScript files for Web Design
         │     └─ images
         │           ├─ input_image
         │           └─ uploads # User Query Images
         ├─ templates
         │      ├─ find_my_dog_a.html  # Result Page
         │      └─ index.html   # Main page
         └─ app.py # Web Application (Calls Models)
```
#### requirement
python==3.8 <br>
pytorch==1.6.0 (window, conda, cuda==10.1) conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
absl-py==0.10.0
easydict==1.9
Flask==1.1.2
Flask-Dropzone==1.5.4
Flask-SQLAlchemy==2.4.4
matplotlib==3.2.2
numba==0.50.1
numpy==1.18.5
opencv-python==4.4.0.42
pandas==1.0.5
pillow==7.2.0
pretrainedmodels==0.7.4
pymysql==0.9.2
scipy==1.4.1
tensorflow==2.3.0 
torchvision==0.7.0 
tqdm==4.47.0
urllib3==1.25.9
werkzeug==1.0.1
yaml==0.2.5
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
  - 윈도우 작업스케줄러를 이용해 배치파일 적용(데이터 수집, 적재, 모델 학습 자동화)
  - flow diagram 작성
  - 웹 페이지 생성(flask, DB 연동)
  - 분류기와 re-id 연결
  - Object detection & Image crop(이미지 전처리)에 YOLO-V4 사용
  - 날짜, 지역 필터링 적용
  - 이메일 알림 기능 추가
