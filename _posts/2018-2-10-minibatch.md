---
layout: post
title: minibatch
---

(c) HakOh 2018

## W4 - minibatch, loss function

#### 학습
- 데이터로부터 가중치의 값을 정하는 방법을 학습이라고 한다.

#### end-to-end
- 데이터(입력)에서 목표한 결과를 사람의 개입 없이 얻는 것.

#### 오버피팅
- 한쪽 이야기(특정 데이터셋)만 너무 많이 들어서 편견이 생겨버린 상태.

#### 손실 함수
- 신경망의 '하나의 지표'를 기준으로 최적의 매개변수 값을 탐색합니다. 신경망 학습에서 사용하는 지표를 손실 함수(loss function)이라고 합니다.

#### 평균제곱오차(MSE)
- 추정값과 참값의 차이를 제곱해서 평균 낸 값
- $E = \frac{1}{2}\Sigma_{k}(y_k - t_k)^(2)$

#### 교차엔트로피오차(CEE)
- 추정값의 확률분포와 참값의 확률분포의 차이
- $E = -\Sigma_{k}t_{k}\log{y_k}$
- $t_k$가 원-핫 인코딩이기 때문에 실질적으로는 정답일때의 출력만 계산된다.


### 교차엔트로피오차 t가 원-핫 인코딩 일때
```python
def cross_entropy_error(y,t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))
```
- np.log()함수에 0이 입력되면 마이너스 무한대를 뜻하는 -inf가 되어 더 이상 계산을 못하기 때문에 아주 작은 값인 delta를 더해줘서 절대 0이 되지 않게 한다.

### 교차엔트로피오차 t가 숫자 레이블 일때
```python
def cross_entropy_error(y,t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta)[t])
```

#### 미니배치
- 훈련 데이터 일부 중에 표본 추출한 것

```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape) #(60000, 784)
print(t_train.shape) #(60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

#### 미니배치 vs 배치
- 배치 : 데이터 전체를 여러 개 묶음으로 나누것, 중복불가
- 미니배치 : 데이터 전체를 일부로 추린 것, 중복가능(무작위 랜덤 표본 추출)

전체 데이터가 60,000 장이라고 했을 때,
- 배치 : 배치 사이즈가 100 장이라면, 배치 하나당 100장이고 총 배치 개수는 600개
- 미니배치 : 60,000 장에서 100장만 표본추출한 것
#### 배치용 교차엔트로피 오차 구현

```python
def cross_entropy_error(y, t):
  if y.dim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum(t * np.log(y)) / batch_size
```

```python
def cross_entropy_error(y, t):
  if y.dim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
```

#### 최적화의 기준으로 정확도가 아니라 손실 함수를 사용하는 이유
- 최적의 가중치를 찾을 때 정확도보다 손실 함수를 써야 정밀하게 최적화
왜냐하면 정확도는 손실 함수 값보다 불연속적이기 때문이다.
불연속적이면 미분값이 0이 되는 곳이 많다는 뜻이고, 미분값이 0이 되면 그만큼 정밀하게 최적화할 수 없다는 뜻이기 때문이다.



## Further study

+ **(not yet)** 피클로 저장하는 법 해보기

### solved
+ **(solved)** 부모 디렉토리의 부모 디렉토리에 있는 파이썬 파일 import하는 법  
```python
import os
os.listdir('.')
```
+ **(solved)** 네트워크를 짤 때 뉴런의 개수가 어떤 의도를 가진 것인지 의미 파악하기 == input feature를 뉴런 개수만큼 압축하거나 더 세밀하게 나눈다.
+ **(solved)** X의 row=1, column=2 인데 (2,)으로 표시되는 이유? == 1차원 칼럼벡터나 1차원 로우벡터는 (k,)으로 표시한다.
+ **(solved)** 출력층의 활성화 함수에서 binary classification에서는 sigmoid를 쓰고, multiclass classification에서는 softmax를 사용하는 이유? == 생각해보니 softmax는 sigmoid 보다 exp 연산을 더 많이 하므로 연산이 더 비싸서 binary에서는 sigmoid를 쓰는 것 같다.
+ **(solved)** 학습 단계에서 softmax 함수를 굳이 적용하는 이유가 무엇일까? == 학습시 loss를 계산해서 optimize를 할 때 카테고리 데이터는 score로 loss를 최적화하기보다는 각 카테고리에 해당할 확률을 계산해서 loss를 최적화하는 것이 논리적으로 더 맞다.
+ **(solved)** 추론 단계에서 softmax 함수를 생략해도 되는 이유는? == 학습 단계와 달리 loss를 구해서 optimize할 필요가 없고 단지 score 벡터의 argmax만 구하면 되기 때문이다.
