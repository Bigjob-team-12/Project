#### dog_breed_classification

- dog_breed_classification : 품종 분류 및 모델 저장 / [참고](https://www.kaggle.com/gpiosenka/general-purpose-image-classifier-keras-tf2)

<dog_breed_classification 진행 과정>
1.	TF2_classify 함수에서 파라미터 입력 후 get_data_set 함수 호출 
2.	get_data_set에서 경로를 찾는다. mode에 따라 다른데, train, test, valid가 각각 다른 폴더에 있으면 mode = sep이다. 
    Train 데이터가 있는 폴더의 경로를 찾으면 read_files 함수 호출 
3.	이때 train 폴더에 있는 폴더의 개수가 class 개수이다. 폴더안에 있는 파일을 가져온다. 이미지 파일 가져오면서 resize를 해준다. (224,224) 
4.	다시 get_data_set 함수에서 이미지를 numpy 행렬로 변환
5.	이후 TF2_classify 함수에서 get_steps 함수 호출 get_steps에서는 batchsize에 따른 step계산을 한다. 
6.	이후 make_model 함수 호출, 이때 3가지 선택지가 있는데, ‘L’ 은 정확도가 높지만 요구하는 메모리가 크다. M은 요구하는 메모리는 작지만 속도는 빠르다. 이때 pretrained 된 mobilenet V2를 사용하는데, 이미지넷 기반으로 학습된 모델이다. 추론이 시작되는 부분인 mobile.layers[-6]을 잘라서 새로운 fully-connected layer를 붙여준다. 이후 다 같이 트레이닝 시킨다. 
7.	이후 make_generators 호출한다. 이미지 전처리를 하는 단계.
8.	make_predictions , display_pred 함수는 best 가중치를 가지고 추론 진행 + accuracy 반환 
9.	wrapup안에는 accuracy에 따라서 학습을 조금 더 할지, 말지 정해주는 로직이 있다.


- dog_breed_similarity_comparison : test dataset의 image와 적재된 image와 유사도 비교 후 비슷한 image 보여주기
- model_testing_with_real_data : model test 및 test dataset의 softmax 확률값 저장
- show_similar_images : 하나의 image 입력 시 공고에 올라온 image와 유사도 비교 후 비슷한 image 보여주기
