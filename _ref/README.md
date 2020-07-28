### paper 참고 자료 링크

- [Metric Learning 발전 동향](https://blog.est.ai/2020/02/%EB%A9%94%ED%8A%B8%EB%A6%AD%EB%9F%AC%EB%8B%9D-%EA%B8%B0%EB%B0%98-%EC%95%88%EA%B2%BD-%EA%B2%80%EC%83%89-%EC%84%9C%EB%B9%84%EC%8A%A4-%EA%B0%9C%EB%B0%9C%EA%B8%B02/)
- [DeepLanking1](https://umbum.dev/262)
- [DeepLanking2](https://you359.github.io/meta%20learning/DeepRanking/)
- [Triplet Network](https://m.blog.naver.com/PostView.nhn?blogId=4u_olion&logNo=221478534498&proxyReferer=https:%2F%2Fwww.google.com%2F)



### dataset 참고 자료 링크
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)
  - 120 카테고리, 총 20,580장
  - annotation: Class labels, Bounding boxes
- [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
  - 개, 고양이 총 37종 * 최대 200장. 고양이 2371장, 개 4978장, 총 7349장
  - annotation: (a) species and breed name; (b) a tight bounding box (ROI) around the head of the animal; (c) a pixel level foreground-background segmentation (Trimap).
  - <img src="https://www.robots.ox.ac.uk/~vgg/data/pets/pet_annotations.jpg" width="300">
- [70 Dog Breeds-Image Data Set](https://www.kaggle.com/gpiosenka/70-dog-breedsimage-data-set?)
  - 개인 인터넷 수집. 70종 * 10장 = 700장
  - 종별 폴더 분류. 이외의 annotation 없음.
- [Cat Dataset](https://www.kaggle.com/crawford/cat-dataset)
  - annotation: Left Eye, Right Eye, Mouth, Left Ear-1, Left Ear-2, Left Ear-3, Right Ear-1, Right Ear-2, Right Ear-3
  - 종 정보 없음. 9997장
- [ImageNet](http://image-net.org/synset?wnid=n02121808)
  - 페르시안, 앙고라, 샴고양이, 포메라니안, 사모예드 등 종별 300~2천여 개의 데이터 링크 제시.
    - Tabby, puppy 등 특성 분류도 존재.
  - 이미지는 flickr 등의 링크로 제공. 직접 다운로드 필요.
    - 확인 결과 삭제된 데이터, 2마리 이상의 개체가 포함되어 있는 이미지 다수 존재.
