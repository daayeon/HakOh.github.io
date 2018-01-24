---
layout: post
title: neural network and softmax
---


(c) HakOh 2018

## W2 - 신경망 구성하기


#### 3층 신경망 구현하기
- init_network() 함수는 가중치와 편향을 초기화하고 딕셔너리 변수인 network에 저장합니다.
- forward() 함수는 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현합니다.


```python
#define sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#define identity_function
def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    # 출력층의 함수로써 항등 함수를 썼다.
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print (y)
```
#### softmax 구현

```python
#softmax 구현

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

#softmax

def softmax(a):
    c = np.max(a)  
    exp_a = np.exp(a-c) # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

#### activation function
+ binary classification - 일반적으로 sigmoid 사용
+ multiclass classification - 일반적으로 softmax 사용

#### 소프트맥스 함수의 장점
+ 소프트맥스 함수를 출력층에 적용하면 확률적으로 문제를 대응할 수 있는 장점이 생긴다. 이전 레이어에서 계산된 score를 확률값으로 해석할 수 있도록 도와주기 때문이다.

#### 소프트맥스 함수 구현시 주의점
+ 오버플로 - 표현할 수 있는 범위가 한정되어 너무 큰 값을 표현할 수 없는 문제.
그래서 오버플로를 방지하기 위해서 소프트맥스 함수를 적용할 원소 중 최대값을 구해놓고, 각 원소마다 최대값을 뺀 후(일종의 정규화)에 지수함수를 적용하는 방법을 사용한다.

#### 데이터 전처리와 정규화

데이터를 특정 범위로 변환하는 처리를 정규화(normalization)라고 한다. 신경망의 입력 데이터에 특정 변환을 가하는 것을 전처리(pre-processing)라고 한다. 예를 들어, MNIST 데이터의 픽셀값을 0-255 범위인 것을 0-1범위로 변환하는 것은 입력 이미지에 대한 전처리 작업으로 정규화를 수행한 것이다.
