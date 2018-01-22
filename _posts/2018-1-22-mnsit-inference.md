---
layout: post
title: mnist data classification
---

(c) HakOh 2018

## MNIST 데이터를 분류하는 신경망

**학습된 매개 변수를 사용하여 추론과정을 구현**
1. 훈련 데이터(학습 데이터)를 사용해 가중치 매개변수를 학습
2. 추론 단계에서는 앞서 학습한 매개변수를 사용하여 입력데이터 분류

**MNIST 데이터 분류**

MNIST data를 불러온 다음 train set, test set으로 나눕니다.
```python
# MNIST 데이터를 불러와서 training set과 test set으로 나눕니다.
def get_data():
    (X_train, t_train), (X_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return X_test, t_test
```

학습된 가중치 매개변수를 읽어옵니다.
```python
# 미리 학습시킨 weight를 저장했던 pickle 파일을 불러옵니다.
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network
```

input data를 받아서 함수를 정의합니다.
```python
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
```

위의 함수들을 사용하여 신경망에 의한 추론을 수행하고, 정확도를 평가합니다.
```python
# x_test와 t_test를 리턴합니다.
x, t = get_data()
# 미리 저장해둔 hidden layer1 (50 units) + hidden layer2(100 units)으로 구성된 network를 리턴합니다.
network = init_network()

# 정확도를 나타내는 변수 accuracy_cnt를 정수 타입으로 초기화 합니다.
accuracy_cnt = 0
# x_test를 하나씩 순회합니다.
for i in range(len(x)):
    # 순서대로 x_test를 network에 넣어서 softmax를 통과한 10차원 벡터 y를 리턴합니다.
    y = predict(network, x[i])
    # 10차원 벡터 y 중에서 가장 값이 큰 index를 p에 저장합니다.
    p = np.argmax(y)
    # 해당 x_test의 label과 p가 일치하면 정확도에 1을 더합니다.
    if p == t[i]:
        accuracy_cnt += 1
print ("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```




### Quote

- sys.path.append(os.pardir): 부모디렉토리로부터 파일을 찾을 수 있도록 path 설정
- pickle : 프로그램 실행 중에 특정객체를 파일로 저장하는 기능
