---
layout: post
title: document classification
---

(c) HakOh 2018

## 1. 개발 목적

1) Web Crawler
- 일반적으로, 데이터를 모으고 정리하는 것은 딥러닝을 활용한 자연어처리 프로젝트에서 정말 중요함.
- 따라서, 이번 프로젝트에 적합한 웹크롤러를 개발하고자 함.

2) Data Preprocessing
- 모은 데이터에 어떤 전처리를 하는지에 모델의 성능이 크게 좌우됨.
- 특히 한국어 모델에 맞는 데이터 전처리는 상당히 어려움.
- 따라서, 이번 프로젝트를 통해 효율적인 데이터 전처리 과정을
개발하고자 함.

3) CNN for Document Classification
- 문서분류는 현재도 활발하게 연구가 이루어지고 있는 분야 중 하나임. - 그 중 CNN 구조는 모든 데이터에 대해 매우 좋은 성능을 보임. [1]
- 따라서, 이번 프로젝트에서 ‘입력층은 Word2vec 단어 embedding을
연결한 문장, 그 후에 multiple filters를 가진 convolutional층, max-pooling 층을 연결하고, 마지막엔 softmax 분류기를 가지는’ 구조를 구현하고자 함.

## 2. 개발 목표

### 1) 목표

(1) Web Crawler
- 총 12개의 카테고리에 대하여 데이터 수집
- (2008 – 2018) 총 1500만건의 공공데이터 활용. (https://www.open.go.kr/)
- 파일이 없는 경우, 엑셀파일이 업로드 된 경우 등의 예외처리.

(2) Data Preprocessing
- Pdf, Hwp 파일을 Txt로 변환.
- 형태소 분석 (Twitter, Komoran)
- 한글 외에 다른 문자 제거, 빈도수 낮은 단어 제거, Padding(7280)
- Wiki Corpus를 덮어 씌워서 어휘량을 늘림.

(3) CNN for Document Classification
- Filter size = 3, 4, 5
- #of Filter = 100
- Maxpooling = 2
- Hidden dimension = 50
- Dropout = 0.5
- Activation function = ‘Relu’
- FC : activation function = ‘Softmax’

(4) Demo
- Shutil module을 이용하여 실제 문서를 11개의 카테고리에 대해 분류.

### 2) 제약사항

(1) 크게 제한된 데이터 수
- 기존의 1500만건 데이터를 모두 이용하려 했으나 2014년 이전 자료는 직접
기관에 청구해야 해서 크롤러를 이용해 얻을 수 없음.
- 총 12개의 카테고리에 대해 분류하려 했으나 데이터 수가 10,000건도 되지 않은 카테고리가 있기 때문에 실제 사용하지 못할 수 있음.
- 문서의 개인 정보 보안상의 문제, 데이터 서버의 안정성 문제 등으로 인해 크롤링이 원활하지 않음.

## 3. 개발 내용

### 1) System Architecture

![architecture](/images/architecture.png)

### 2) Web Crawler

- BeautifulSoap(from bs4) 사용. - Selenium 사용.
- 예외처리
  - 파일이 없거나 다른 형식의 파일(엑셀, 이미지)
  - 비공개, 정보 보안상 문제로 접근이 제한된 문서

### 3) Data Preprocessing

(1) Web Crawler를 통해 받은 Pdf, Hwp 파일을 각각 Pdf2text pilot, HwpConv 프로그램을 이용해 Txt로 변환.

(2) 한글 외에 다른 문자 제거, 빈도수 낮은 문자 제거(5회 이하).

(3) Komoran 형태소 분석기를 이용하여, 형태소를 비롯한 어근,
접두사/접미사, 품사(POS, part-of-speech) 등 다양한 언어적 속성의
구조를 파악.

(4) Padding
- Padding값을 조절을 못하여 처음에는 7280까지 Padding을 했음.
- Padding 평균값이 70-80 인 것에 비하면 너무 크기 때문에 1000으로 조정.
(1000이상인 값은 200여개였기 때문에 버리기로 판단)
- Padding값이 0인 것이 3천여개에 달하였기 때문에 0인 값도 모두 삭제.

(5) Wiki Corpus, Word2vec
- 데이터에 문장이 거의 없기 때문에, line by line 문장을 임의로 생성하여
Word2vec 실행.
- 성능을 올리기 위해 Wiki Corpus 사용.

### 4) CNN for Document Classification
(1) Hyper parameters
- Filter size = 3, 4, 5
- #of Filter = 100
- Maxpooling = 2
- Hidden dimension = 50
- Dropout = 0.5
- Activation function = ‘Relu’
- FC : activation function = ‘Softmax’

(2) CNN model
- Word2Vec Size를 100에서 300으로 변경
- Category 2개 일 때 만 “Binary_Cross Entropy” 사용.
- 그 이상의 카테고리에서는 “Categorical Cross Entropy” 사용.
- Category 2개 일 때 “Activation Function”을 “Sigmoid”를 “Relu”로 조정
- Overfitting 문제를 줄이기 위해 Dropout을 0.7, Regularization 0.01로 수정..

(3) Word2Vec

(4) Softmax function

## 4. 개발 결과

### 1)목표 달성 여부
(1) Web Crawler
- 총 11개의 카테고리에 대하여 데이터 수집
- (2014-2018) 약 221,000건의 공공데이터 활용 (https://www.open.go.kr/)
- 파일이없는경우,엑셀파일이업로드된경우등의예외처리

(2) Data Preprocessing
- Pdf, Hwp 파일을 Txt로 변환.
- 형태소 분석(Twitter, Komoran)
- 한글 외에 다른 문자 제거, 빈도수 낮은 단어 제거, Padding(1000)
- Wiki Corpus를 덮어 씌워서 어휘량을 늘림

(3) CNN for Document Classification
- Filter size = 3, 4, 5
- #of Filter = 100
- Maxpooling = 2
- Hidden dimension = 50
- Dropout = 0.7
- Activation function = ‘Relu’
- FC : activation function = ‘Softmax’

(4) 카테고리 개수 별 성능 (개별 결과의 전처리, 모델과 동일)

![result](/images/result.png)

(5) Demo
- Shutil module을 이용, 실제 문서를 11개 미만의 카테고리에 대해 분류.
- Accuracy에 비해 낮은 분류 성능이 나와 현재 오류 개선 중.


### 2) 발전 가능성
(1) Vocab을 만들 때 품사 태깅 정보도 활용 하면 성능 개선에 훨씬 도움이 될 수 있음. (명사, 동사, 형용사, 부사 등의 content word 이용.)

(2) 데이터가 턱 없이 부족한 상태로 진행을 하였기 때문에 데이터가 더 많다면 확실히 좋은 성능이 나올 것이라 기대함.

(3) 품사 태그 다 떼고 embedding 한 것과 content word만 남기고 Embedding 한 것을 같이 사용.

(4) 보다 나은 한글 말뭉치를 찾아서 Embedding을 해줌. 단, Embedding 할 때 단어장을 만들 때 썼던 형태소 분석기를 써주고 이 또한 필요 없는 조사나 어미 등을 삭제 하여 성능 개성 도모.

(5) Word2vec 대신 Doc2Vec을 활용하면 성능개선이 가능하다. [2]

## 5. Reference

[1] Kim, Y. (2014). Convolutional Neural Networks for Sentence
Classification. Proceedings of the 2014 Conference on Empirical
Methods in Natural Language Processing (EMNLP 2014), 1746–1751.

[2] Dowoo Kim, Myoung-Wan Koo (2017). Doc2Vec과 Word2Vec을 활용한 Convolutional Neural Network 기반 한국어 신문 기사 분류( 정보과학회논문지 제44권 제7호, 2017.7, 742-747 (6 pages))
Kim, Y. (2014). Convolutional Neural Networks for Sentence
Classification. Proceedings of the 2014 Conference on Empirical
Methods in Natural Language Processing (EMNLP 2014), 1746–1751.
   
[3] Kalchbrenner, N., Grefenstette, E., & Blunsom, P. (2014). A Convolutional Neural Network for Modelling Sentences. Acl, 655–665.

[4] Santos, C. N. dos, & Gatti, M. (2014). Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts. In COLING-2014 (pp.
69–78).

[5] Johnson, R., & Zhang, T. (2015). Effective Use of Word Order for Text Categorization with Convolutional Neural Networks. To Appear:
NAACL-2015, (2011).

[6] Johnson, R., & Zhang, T. (2015). Semi-supervised Convolutional Neural Networks for Text Categorization via Region Embedding.

[7] Wang, P., Xu, J., Xu, B., Liu, C., Zhang, H., Wang, F., & Hao, H. (2015). Semantic Clustering and Convolutional Neural Network for Short Text Categorization. Proceedings ACL 2015, 352–357.

[8] Zhang, Y., & Wallace, B. (2015). A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification,

[9] Nguyen, T. H., & Grishman, R. (2015). Relation Extraction: Perspective from Convolutional Neural Networks. Workshop on Vector Modeling for NLP, 39–48.

[10] Sun, Y., Lin, L., Tang, D., Yang, N., Ji, Z., & Wang, X. (2015). Modeling Mention , Context and Entity with Neural Networks for Entity Disambiguation, (Ijcai), 1333–1339.

[11] Zeng, D., Liu, K., Lai, S., Zhou, G., & Zhao, J. (2014). Relation Classification via Convolutional Deep Neural Network. Coling, (2011),
2335–2344.

[12] Gao, J., Pantel, P., Gamon, M., He, X., & Deng, L. (2014). Modeling Interestingness with Deep Neural Networks.

[13] Shen, Y., He, X., Gao, J., Deng, L., & Mesnil, G. (2014). A Latent Semantic Model with Convolutional-Pooling Structure for Information
Retrieval. Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management – CIKM ’14,
101–110.

[14] Weston, J., & Adams, K. (2014). # T AG S PACE : Semantic Embeddings from Hashtags, 1822–1827.

[15] Santos, C., & Zadrozny, B. (2014). Learning Character-level Representations for Part-of-Speech Tagging. Proceedings of the 31st International Conference on Machine Learning, ICML-14(2011),
1818–1826.

[16] Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification, 1–9.

[17] Zhang, X., & LeCun, Y. (2015). Text Understanding from Scratch. arXiv E-Prints, 3, 011102.

[18] Kim, Y., Jernite, Y., Sontag, D., & Rush, A. M. (2015). Character-Aware Neural Language Models.
