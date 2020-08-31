# 빅데이터 청년인재 고려대 과정 12조

## 프로젝트 명 : 찾아주 Cat! :cat: Dog! :dog:
- 진행 기간: 2020.07 ~ 2020.08

## 1. 개발 배경 :question:
- '19년 기준 유실 및 유기동물 **13만 5천마리** 중 입양(26.4%), **자연사`(24.8%)`**, **안락사`(21.8%)`**, **반환`(12.1%)`**
- 만약 실제로 동물을 잃어버렸을 경우 **전단지 제작 베포, 커뮤니티 신고 게시, 관리시스템 수시 확인**
- **"운 좋게"** 보호소 인계 시, **카운트타운** 시작.
- [농림부 동물보호관리시스템 공고](https://www.animal.go.kr/front/awtis/public/publicList.do?menuNo=1000000055)에 **`7-10일`**, 일 평균 관리시스템 등록 수는 **`370건`**
- 공고에 올라온 이후부터 매일 약 370건 개별 확인 필요
- 기존 상용 서비스 비교

| **종합유기견보호센터** | **포인핸드** |
| :-----------: | :-----------: |
| 유기동물 종합 커뮤니티 | 사설 모바일 서비스 |
| 실종 신고 등록 활성화 | 입양 사업에 초점 |

- 관리시스템 단순 연동 전달 이외 **`반환`에 특화된 기술집약 서비스 전무**

## 2. 서비스 개요 :memo:
- **사진 한장**으로 실종 동물을 찾도록 도와주는 **`반환` 특화 이미지 `매칭` 시스템**
- 주요 기능
  - 관리시스템 데이터 동기화 / 추가 수집 **자동화**
  - 유저 업로드 이미지와 **유사도 기반** 검색 / 목록화
  - 신규 공고 자동 매칭 및 **자동 이메일 발송**
 
## 3. 팀 소개 및 역할 :two_men_holding_hands:
> **손우진**
- 개발 환경 구축, 전반적인 프로젝트 관리, 발표

> **권태양**
- 공고 데이터 수집, 적재 및 모델 학습 자동화, 품종 분류기 구현

> **이상헌**
- 데이터 전처리, 분류 모델 평가, re id 구현, 발표 보조

> **홍승혜**
- 모델 학습 데이터 수집, WEB 구현, 이메일 push 기능 

## 4. 시스템 구성도
<div>
  <img width="700" src="https://github.com/Bigjob-team-12/Project/blob/master/_img/system.png">
</div>

## 5. 프로젝트 요약
- 인터넷에 산재되어 있는 **유기동물 공고 정보**를 `수집` 및 `적재`
- **Fine-grained Classification**을 통해 생김새를 파악
- 외견적 특성을 정의할 수 있는 세부 품종 분류를 수행한 후 **품종별 확률값 도출**
- 도출된 확률값 바탕으로 탐색 대상과 Query Image의 **PCC(Pearson Correlation Coefficient)** 기반 `유사도 계산`, 유사성이 낮은 개체 후보군에서 제외
- **Metric Learning**을 통해 털의 색, 조합의 유사성을 확인.
- 작업 스케줄러를 이용해 **파이프라인 자동화**(신규 공고 데이터 수집 및 적재, 모델 학습)
- 사용자 요청 시 기존 Query 이미지와 비교를 지속 수행, 새로이 유사한 개체가 발견되었을 때 **사용자에게 메일 발송**

#### 주요 기능
- **특정 개체 특성 중심 탐색**
  > 품종 등 메타데이터를 바탕으로 한 단순 검색을 넘어 개체의 시각적 특성을 바탕으로 한 유사성 비교를 수행<br/>
  공고 데이터의 90%를 차지하는 믹스견 역시 세부적인 특징을 분석해 유사한 개체를 찾아냄
- **파이프라인 자동화**
  > 주기적으로 신규 등록 개체 수집 및 분석하여 기존 Query 개체와 비교<br/>
  유사도가 높은 개체 발견될 경우 메일 발송
- **Fine-Grained Classification**
  > 세부적인 품종의 외견 정보를 바탕으로 분류를 수행<br/>
  여러 품종의 특성을 가진 믹스견을 고려해, 외형적으로 추측되는 각 품종의 확률값을 도출
- **Metric Learning**
  > Query 이미지와 유사한 색상(단일 컬러 및 여러 색의 혼합)인 개체를 찾기 위해 특성을 추출한 후, Similarity를 계산<br/>
  Similarity 계산을 위해 Metric Learning을 적용
- **Stacking ensemble**
  > Fine-Grained Classification으로 각 품종 확률값 기준 필터링<br/>
  필터링된 이미지를 Metric Learning을 통해 털의 색, 조합의 유사성 확인<br/>
  이목구비 형태의 유사성과 털의 색, 무늬 등 패턴의 유사성을 모두 확인하기 위해 두 모델을 연결

## 6. System
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
     │     └─ update_data.bat   # Batch file for crawling and Preprocessing 'post' images
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
     │     └─ data_collection_zooseyo.py # 'http://www.zooseyo.or.kr/zooseyo_or_kr.html' site data crawling
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

## 7. requirement
```
python==3.8
pytorch==1.6.0 (window, conda, cuda==10.1) conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
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

## 8. 프로젝트 진행 과정

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
