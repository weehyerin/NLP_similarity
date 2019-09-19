# NLP_similarity

- 텍스트가 얼마나 유사한지를 표현하는 방식
- 다른 구조의 문장이지만, 의미가 비슷하면 두 문장의 유사도가 높다
- 데이터를 정량화 하는 것이 중요

> 데이터 정량화하는 방법
-	1. 단순히 텍스트를 벡터화한 후 벡터화된 각 문장 간의 유사도를 측정 
-	2. 같은 단어의 개수를 사용해서 유사도를 판단하는 방법
-	3. 형태소로 나누어 형태소를 비교하는 방법
-	4. 자소 단위로 나누어 단어를 비교하는 방법 등

데이터 : kaggle : https://www.kaggle.com/c/quora-question-pairs/data

다운로드
~~~
kaggle competitions download -c quora-question-pairs
~~~
  - 만약, 403 error가 나면 (https://www.kaggle.com/c/quora-question-pairs/rules) 여기서 accept를 해주면 됨. 
  
#### 데이터 설명
질문과 답변을 할 수 있는 사이트 --> 중복된 질문들을 잘 찾기만 한다면 이미 잘 작성된 답변들을 사용자들이 참고하게 할 수 있음

## 데이터 분석과 전처리
- 그 데이터 분석을 통해 ***데이터 특징 파악***
- 특징을 파악 후, 데이터 전처리 작업 진행

#### 데이터 zip 파일 풀기
~~~
import zipfile

DATA_IN_PATH = './data_in/'

file_list = ['train.csv.zip', 'test.csv.zip', 'sample_submission.csv.zip']

for file in file_list:
    zipRef = zipfile.ZipFile(DATA_IN_PATH + file, 'r')
    zipRef.extractall(DATA_IN_PATH)
    zipRef.close()
~~~

> 데이터 설명
> train : 학습 데이터
> test : 평가 데이터
> sample_submission : 모델을 통해 평가 데이터에 대한 예측 겨로가를 캐글 홈페이지에 제출할 때 양식을 맞추기 위해 보여주는 예시 파일

<img width="587" alt="스크린샷 2019-09-12 오후 10 08 46" src="https://user-images.githubusercontent.com/37536415/64786676-f0580380-d5a9-11e9-8fb5-681e8cef3859.png">

id : rkr god epdlxjdml rhdbgks dlseprtm
qid : 각 질문들의 고유한 인덱스 값
question : 각 질문의 내용
is_duplicate : 0 또는 1(0이면 두 개의 질문이 중복이 아니고, 1이면 두 개의 질문이 중복)

파일 크기 : 
test.csv                      314.0MB
train.csv                     63.0MB
sample_submission.csv         22.0MB

> 평가 데이터가 훈련 데이터보다 크다
> test data가 더 큰 이유는 쿼라의 경우 질문에 대해 데이터의 수가 적다면 각각을 검색을 통해 중복을 찾아내는 편법을 사용할 수 있는데, 이러한 편법을 방지하기 위해서

**현재 데이터는 한 개 row 당 두 개의 질문을 담고 있음**
- 두 개의 질문을 한 번에 분석하기 위해 판다스의 시리즈를 통해 두 개의 질문을 하나로 합친다. 
~~~
train_set = pd.Series(train_data['question1'].tolist() + train_data['question2'].tolist()).astype(str)
~~~

#### 질문 중복 분석

~~~
print('교육 데이터의 총 질문 수: {}'.format(len(np.unique(train_set))))
print('반복해서 나타나는 질문의 수: {}'.format(np.sum(train_set.value_counts() > 1)))
~~~

> 교육 데이터의 총 질문 수: 537361
> 반복해서 나타나는 질문의 수: 111873
> 중복을 제거한 유일한 질문값만 확인하기 위해 질문 수 확인

넘파이를 확인하여 중복을 제거 : unique, value_counts 함수 하숑

총 row의 개수가 40만개이므로, 각 row 당 두 개의 질문이 있으니 == 총 80만개의 질문
이 중 53만개가 unique한 개수, 나머지 27만개는 중복된 것
27만개의 중복된 것들은 11만개로 이루어짐.

![image](https://user-images.githubusercontent.com/37536415/64829343-e7e1e600-d606-11e9-8c7c-ec0c533bf3a1.png)
> 그림 설명 : x : 중복 개수, y : 동일한 중복 횟수를 가진 질문의 개수
> 중복 횟수가 1인것이 가장 많고, 대부분 중복 개수 50개 이하에 분포

중복 최대 개수: 161
중복 최소 개수: 1
중복 평균 개수: 1.50
중복 표준편차: 1.91
중복 중간길이: 1.0
제 1 사분위 중복: 1.0
제 3 사분위 중복: 1.0

![image](https://user-images.githubusercontent.com/37536415/64829422-3abb9d80-d607-11e9-89f1-a07e602d5436.png)


#### 데이터에 어떤 단어가 포함되었는가?

~~~
from wordcloud import WordCloud
cloud = WordCloud(width=800, height=600).generate(" ".join(train_set.astype(str)))
plt.figure(figsize=(15, 10))
plt.imshow(cloud)
plt.axis('off')
~~~

워드클라우드 활용
![image](https://user-images.githubusercontent.com/37536415/64830687-cbe14300-d60c-11e9-8255-ee80df56592d.png)

결과 : best, way, good ... + donuld trump가 많이 등장(선거 기간 중 학습 데이터를 만들었기 때문)

#### duplicate column 확인

![image](https://user-images.githubusercontent.com/37536415/64830729-0c40c100-d60d-11e9-8ca9-6b3e601e9075.png)

> 한쪽으로 치우쳐져 있음. 중복이 아닌 데이터 25만 개에 의존도가 높아지면서 데이터가 한쪽 라벨로 편항될 수 있음. 
> 학습이 원활하게 되기 위해 조정해주면 좋음

#### text 길이 

![image](https://user-images.githubusercontent.com/37536415/64830764-37c3ab80-d60d-11e9-897c-f98c4f9063e6.png)

- 질문 길이 최대 값: 1169
- 질문 길이 평균 값: 59.82
- 질문 길이 표준편차: 31.96
- 질문 길이 중간 값: 51.0
- 질문 길이 제 1 사분위: 39.0
- 질문 길이 제 3 사분위: 72.0

#### 사용 단어 개수 세기
1. 단어 쪼개기
~~~
train_word_counts = train_set.apply(lambda x:len(x.split(' ')))
~~~

![image](https://user-images.githubusercontent.com/37536415/64830831-8bce9000-d60d-11e9-81db-195d16b889f1.png)

- 질문 단어 개수 최대 값: 237
- 질문 단어 개수 평균 값: 11.06
- 질문 단어 개수 표준편차: 5.89
- 질문 단어 개수 중간 값: 10.0
- 질문 단어 개수 제 1 사분위: 7.0
- 질문 단어 개수 제 3 사분위: 13.0
- 질문 단어 개수 99 퍼센트: 31.0

> 물음표가있는 질문: 99.87%
> 수학 태그가있는 질문: 0.12%
> 질문이 가득 찼을 때: 6.31%
> 첫 글자가 대문자 인 질문: 99.81%
> 대문자가있는 질문: 99.95%
> 숫자가있는 질문: 11.83%

전체적으로 질문들이 물음표와 대문자로 된 찻 문자를 가지고 있다. 
- 모든 질문이 보편적으로 가지고 있는 특징 --> **질문의 보편적인 특징은 삭제**
+) 위에 설명) 라벨이 한 쪽으로 치우쳐져 있으므로(중복인지 아닌지) 양 맞춰주기

----------

## 데이터 전처리

### 1. 라벨의 균형을 맞추기
중복이 아닌 데이터의 개수가 많아서 줄이기

1. loc function : 라벨이 1인 경우와 0인 경우를 분리하기
2. 적은 데이터의 개수를 많은 데이터에 대한 비율을 계산
3. 비율만큼 데이터 많은 것에 대해 샘플링
~~~
train_pos_data = train_data.loc[train_data['is_duplicate'] == 1]
train_neg_data = train_data.loc[train_data['is_duplicate'] == 0]

class_difference = len(train_neg_data) - len(train_pos_data)
sample_frac = 1 - (class_difference / len(train_neg_data))

train_neg_data = train_neg_data.sample(frac = sample_frac)
# 비슷한 크기의 데이터를 다시 합치기
train_data = pd.concat([train_neg_data, train_pos_data])
~~~

### 2. 특수문자 삭제, 대문자 -> 소문자 바꾸기

~~~
# 정규표현식으로 전처리하기 위해 re 라이브러리 사용
# FILTERS : 물음표와 마침표를 포함해서 제거하고자 하는 기호의 집합
change_filter = re.compile(FILTERS)

# 질문을 리스트로 만듦
questions1 = [str(s) for s in train_data['question1']]
questions2 = [str(s) for s in train_data['question2']]

# 전처리한 리스트를 담을 그릇
filtered_questions1 = list()
filtered_questions2 = list()

# filter로 정의된 것(특수문자 지우기) 후, .lower()로 모두 소문자로 바꿔서 리스트에 저장
for q in questions1:
     filtered_questions1.append(re.sub(change_filter, "", q).lower())
        
for q in questions2:
     filtered_questions2.append(re.sub(change_filter, "", q).lower())
~~~

## 3. tokenizing, 각 텍스트를 인덱스로 바꾸기, 문장 길이 맞추기

### 3-1. tokenizing
**한 row 당 두개의 질문이 있는데 그 두 개를 합쳐서 tokenizing**
<br>이렇게 하는 이유 : 동일하게 tokenizing을 진행하고, 전체 단어 사전을 만들기 위해서

~~~
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_questions1 + filtered_questions2)
~~~

### 3-2. 각 텍스트를 인덱스로 바꾸기

~~~
questions1_sequence = tokenizer.texts_to_sequences(filtered_questions1)
questions2_sequence = tokenizer.texts_to_sequences(filtered_questions2)
~~~
![image](https://user-images.githubusercontent.com/37536415/65015506-b8abdb80-d95c-11e9-8eb8-b9b4b4179b53.png)
> 단어 사전에 맞추어 텍스트를 인덱스화 됨.

### 3-3. 최대 길이 정해서 조정해주기

~~~
q1_data = pad_sequences(questions1_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
q2_data = pad_sequences(questions2_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
~~~
> 긴 길이의 텍스트는 잘라주고, 짧은 질문은 0으로 채워주기

## 데이터 저장하기

~~~
TRAIN_Q1_DATA = 'train_q1.npy'
TRAIN_Q2_DATA = 'train_q2.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
DATA_CONFIGS = 'data_configs.json'

np.save(open(DATA_IN_PATH + TRAIN_Q1_DATA, 'wb'), q1_data)
np.save(open(DATA_IN_PATH + TRAIN_Q2_DATA , 'wb'), q2_data)
np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA , 'wb'), labels)

json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'))
~~~

~~~
labels = np.array(train_data['is_duplicate'], dtype=int)
~~~

<br><각 데이터 설명><br>
> TRAIN_Q1_DATA : 한 row에 있는 질문 1에 대해 인덱스 화 시킨 것
> TRAIN_Q2_DATA : 한 row에 있는 질문 2에 대해 인덱스 화 시킨 것
> TRAIN_LABEL_DATA : is_duplicate이라는 column을 따로 뽑아낸 것, target data일듯

### 현재는 train data를 정제하였음. test data도 위의 과정처럼 정제하는 단계 필요

-------

## 모델링
### 1. XG 부스트 모델
XG 부스트란? 'eXtream Gradient Boosting
- 앙상블의 한 방법인 부스팅 기법을 사용

앙상블 기법이란? 여러 개의 학습 알고리즘을 사용해서 더 좋은 성능을 얻는 방법
1. 배깅 : 여러 개의 학습 알고리즘, 모델을 통해 각각 결과를 예측하고 모든 결과를 동등하게 보고 취합해서 결과를 얻는 방식
2. 부스팅 : 배깅은 여러 알고리즘의 결과를 다 동일하게 취합한다면, 부스팅은 각 결과를 순차적으로 취합하는데 모델이 학습 후 잘못 예측한 부분에 가중치를 줘서 다시 모델로 가서 학습하는 방식

![image](https://user-images.githubusercontent.com/37536415/65016967-84d2b500-d960-11e9-8fb4-2aac0ae6bd42.png)
- 싱글은 앙상블 기법이 아니라 단순히 하나의 모델만으로 결과를 내는 방법

#### XG 부스트 - 트리 부스팅
- 랜덤 포레스트 모델이란 여러 개의 decision tree를 사용해 결과를 평균 내는 방법 : 배깅
- 트리 부스팅 : 여러 개의 decision tree를 사용하지만 단순히 결과를 평균 내는 것이 아니라 결과를 보고 오답에 대해 가중치 부여, 
<br>가중치가 부여된 오답에 대해서는 관심을 가지고 정답이 될 수 있도록 결과를 만들고 해당 결과에 대한 다른 오답을 찾아 다시 똑같은 작업을 반복적으로 진행

> **XG boost : 트리 부스팅 방식에 경사 하강법을 통해 최적화하는 방법**
> 연산량을 줄이기 위해 의사결정 트리를 구성할 때 병렬 처리를 사용해 빠른 시간에 학습 가능

-------
## XG 부스트 모델 구현


1. 데이터 가져오기
~~~
train_q1_data = np.load(open(DATA_IN_PATH + TRAIN_Q1_DATA_FILE, 'rb'))
train_q2_data = np.load(open(DATA_IN_PATH + TRAIN_Q2_DATA_FILE, 'rb'))
train_labels = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA_FILE, 'rb'))
~~~
위에 저장해 놓은 q1, q2, label을 각각 가져옴

<br>

~~~
train_input = np.stack((train_q1_data, train_q2_data), axis=1) 
~~~

- stack을 활용해여 두 질문을 하나의 쌍으로 만들어줌.([[A 질문], [B 질문]])

<br>

<검증 데이터 만들기 - test 데이터가 아닌, train 데이터로 만들기>
~~~
train_input, eval_input, train_label, eval_label = train_test_split(train_input, train_labels, test_size=0.2, random_state=4242)
~~~
> 20%를 검증 데이터로 사용

2. xgboost 라이브러리 설치 및 사용하기

~~~
pip install xgboost
~~~
> 설치
~~~
train_data = xgb.DMatrix(train_input.sum(axis=1), label=train_label) # 학습 데이터 읽어 오기
eval_data = xgb.DMatrix(eval_input.sum(axis=1), label=eval_label) # 평가 데이터 읽어 오기
~~~
> xgb 라이브러리의 데이터 형식 따르기 위해서 DMatrix 형태로 바꿔주기
train_input.sum(axis=1) : train_input은 두 개의 데이터를 합쳐준 변수값이다. 두 질문을 하나로 합쳐준다.

3. 모델 학습하기 
~~~
params = {} # 인자를 통해 XGB모델에 넣어 주자 
params['objective'] = 'binary:logistic' # 로지스틱 예측을 통해서 
params['eval_metric'] = 'rmse' # root mean square error를 사용  

bst = xgb.train(params, train_data, num_boost_round = 1000, evals = data_list, early_stopping_rounds=10)
~~~
> 모델을 학습하기 전에 파라미터 옵션을 적어주기
> params['objective'] = 'binary:logistic' : 목적함수
> params['eval_metric'] = 'rmse' : 평가 지표
> num_boost_round : 데이터를 반복하는 횟수
> evals : data_list = [(train_data, 'train'), (eval_data, 'valid')] - 모델 검증시 사용할 데이터
> early_stopping_rounds : overfitting을 방지하기 위해서 조기 종료, 만약 10 epoch동안 에러 값이 별로 줄어들지 않으면 학습을 조기에 종료

![image](https://user-images.githubusercontent.com/37536415/65018710-b2b9f880-d964-11e9-9478-d8bc2eddacb6.png)
> 657 부근에서 10개 정도가 거의 error가 바뀌지 않았기 때문에 early stopping이 됨.

### predict
~~~
test_input = np.stack((test_q1_data, test_q2_data), axis=1) 
test_data = xgb.DMatrix(test_input.sum(axis=1))
test_predict = bst.predict(test_data)
~~~
> test data를 사용해서 예측하기 
-----------
### 2. CNN 모델
- 두 개의 텍스트 문장으로 돼 있기 때문에 병렬적인 구조를 가진 모델을 만들어야 함.

CNN 텍스트 유사도 분석 모델이란?

<br>

문장에 대한 의미 벡터를 합성곱 신경망을 통해 추출해서 그 벡터에 대한 유사도 측정


![image](https://user-images.githubusercontent.com/37536415/65018787-dda44c80-d964-11e9-8057-82790a8cc758.png)
> 모델에 입력하고자 하는 데이터 : 2개
> 기준 문장 : 문장에 유사도를 보기 위해서는 기준이 되는 문장이 필요
> 학습이 진행된 후, 문장의 의미를 파악할 수 있기 때문에 "I love deep learning", "deep nlp is awesome"이라는 문장의 유사도는 높음.

1. 기준 문장과 대상 문장에 대해서 인덱싱을 거쳐 문자열 형태의 문장을 인덱스 벡터 형태로 구성
2. 인덱스 벡터는 임베딩 과정으로 임베딩 벡터로 바뀐 행렬로 구성
3. 임베딩 과정을 통해 나온 문장 행렬은 기준 문장과 대상 문장 각각에 해당하는 CNN 블록(합성 곱 층 + 맥스 풀링) 을 거침
4. 임베딩 블록, cnn 블록을 거친 벡터는 문장에 대한 의미 벡터
5. 두 문장에 대한 의미 벡터를 가지고 여러 방식으로 유사도 구할 수 있음.

--> 이번에는 완전연결 층을 거친 후 최종적으로 로지스틱 회귀 방법을 통해 문장 유사도 점수를 측정할 것


-------
## CNN 모델 구현

1. library import
~~~
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

import json
~~~

> 모델 구현 : tensorflow + numpy
> 단어 사전 정보 가져오기 : json
> 데이터를 학습/검증으로 나누기 위해 : sklearn

2. 데이터 가져오기 

~~~
TEST_Q1_DATA_FILE = 'test_q1.npy'
TEST_Q2_DATA_FILE = 'test_q2.npy'
TEST_ID_DATA_FILE = 'test_id.npy'
~~~
> train_q1 : 기준 문장
> train_q1 : 기준 문장과 비교할 대상의 문장
> test_id : 라벨 값과 데이터의 단어 사전과 단어 사전의 크기값을 가지고 있는 데이터

3. 학습 데이터와 검증 데이터 나누기 

~~~
X = np.stack((q1_data, q2_data), axis=1)
y = labels
train_X, eval_X, train_y, eval_y = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)

train_Q1 = train_X[:,0]
train_Q2 = train_X[:,1]
eval_Q1 = eval_X[:,0]
eval_Q2 = eval_X[:,1]
~~~

X = np.stack((q1_data, q2_data), axis=1) : 두개의 데이터를 하나로 합친 뒤 사용, 공평하게 처리하기 위해서

4. 다시 두 질문으로 나누기
~~~
def rearrange(base, hypothesis, label):
    features = {"x1": base, "x2": hypothesis}
    return features, label

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((train_Q1, train_Q2, train_y))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(rearrange)
    dataset = dataset.repeat(EPOCH)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((eval_Q1, eval_Q2, eval_y))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(rearrange)
    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()
~~~

> rearrange : parameter - (기준 질문, 대상 질문, 라벨), 인자값을 통해 두 개의 질문을 하나의 딕셔너리 형태로 만들어 return 
> train_input_fn : 학습을 위한 입력함수
> eval_input_fn : 검증을 위한 입력 함수

------
## CNN 모델 구현
1. CNN 블록
~~~
def basic_conv_sementic_network(inputs, name): # 문장에 대한 의미벡터를 만드는 과정
    conv_layer = tf.keras.layers.Conv1D(CONV_FEATURE_DIM, 
                                        CONV_WINDOW_SIZE, 
                                        activation=tf.nn.relu, 
                                        name=name + 'conv_1d',
                                        padding='same')(inputs)
    
    max_pool_layer = tf.keras.layers.MaxPool1D(MAX_SEQUENCE_LENGTH, 
                                               1)(conv_layer)

    output_layer = tf.keras.layers.Dense(CONV_OUTPUT_DIM, 
                                         activation=tf.nn.relu,
                                         name=name + 'dense')(max_pool_layer)
    output_layer = tf.squeeze(output_layer, 1)
    
    return output_layer
~~~
> 합성곱 계층 + 맥스 풀링 계층
> (합성곱 계층 + 맥스 풀링 계층)을 통과한 후에는 차원을 바꾸기 위해 Dense 층 통과

2. 모델 함수
~~~
def model_fn(features, labels, mode): 
# parameter - 입력값, 라벨, 모델 함수가 사용된 모드

    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
# 각 parameter를 어떤 상태인지 변수에 저장

    embedding = tf.keras.layers.Embedding(VOCAB_SIZE,
                                          WORD_EMBEDDING_DIM)
    base_embedded_matrix = embedding(features['x1'])
    hypothesis_embedded_matrix = embedding(features['x2'])
# 임베딩 객체를 생성한 후 해당 객체를 사용해 기존 문장과 대상 문장을 임베딩 벡터로 만듦
    
    base_embedded_matrix = tf.keras.layers.Dropout(0.2)(base_embedded_matrix)
    hypothesis_embedded_matrix = tf.keras.layers.Dropout(0.2)(hypothesis_embedded_matrix)  
    base_sementic_matrix = basic_conv_sementic_network(base_embedded_matrix, 'base')
    hypothesis_sementic_matrix = basic_conv_sementic_network(hypothesis_embedded_matrix, 'hypothesis')  
# 임베딩된 값을 CNN블록에 적용
# base : 기준 문장, hypothesis : 대상 문장

    merged_matrix = tf.concat([base_sementic_matrix, hypothesis_sementic_matrix], -1)

    similarity_dense_layer = tf.keras.layers.Dense(SIMILARITY_DENSE_FEATURE_DIM,
                                             activation=tf.nn.relu)(merged_matrix)
    
    similarity_dense_layer = tf.keras.layers.Dropout(0.2)(similarity_dense_layer)    
    logit_layer = tf.keras.layers.Dense(1)(similarity_dense_layer)
    logit_layer = tf.squeeze(logit_layer, 1)
    similarity = tf.nn.sigmoid(logit_layer)
# 윗 단계에서 CNN 블록을 통과했으면 문장에 대한 의미 벡터를 만들었을 것
# 그 벡터값을 가지고, 유사도 측정

    if PREDICT:
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  predictions={
                      'is_duplicate':similarity
                  })
    # 유사도 값을 딕셔너리 형태로 전달

    loss = tf.losses.sigmoid_cross_entropy(labels, logit_layer)
    # 손실값 계산 -> logits 값(결과 값)과 라벨을 통해 손실값 계산, 이후 sigmoid 적용

    if EVAL:
        accuracy = tf.metrics.accuracy(labels, tf.round(similarity))
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  eval_metric_ops= {'acc': accuracy},
                  loss=loss)
    # train data가 아닌, 검증 데이터) 정확도 리턴 + loss 값 return
    
    if TRAIN:
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  train_op=train_op,
                  loss=loss)
    # 학습 상태 리턴 : 가중치를 최적화 해야하므로, (AdamOptimizer)손실값 적용
~~~

~~~
est = tf.estimator.Estimator(model_fn, model_dir=model_dir)
est.train(train_input_fn)
~~~
> 입력한 모델 함수를 Estimator로 입력 후 train

~~~
INFO:tensorflow:Loss for final step: 0.46720922.
~~~
> 최종 손실값을 나타내주고, 학습이 종료됨.

~~~
est.evaluate(eval_input_fn) #eval
~~~
> training한 후에 evaluate 하기

-----------

### 3. MaLSTM

MaLSTM이란?
- 순서가 있는 데이터에 적합
- 순환 신경망(RNN)계 모델
- 문장의 sequence 형태로 학습
- **유사도를 구하기 위해 활용하는 대표적인 모델 : MaLSTM - Manhattan Distance + LSTM**

![image](https://user-images.githubusercontent.com/37536415/65206661-21fd2d00-daca-11e9-9104-9faf5f30bc26.png)

> 의미 벡터 : LSTM의 마지막 step인 (LSTM(a), h3) + (LSTM(b), h4) - 문장의 모든 단어에 대한 정보가 반영된 값으로 전체 문장을 대표하는 벡터
> 두 벡터에 대해 맨하탄 거리를 계싼해서 유사도를 측정한 후, 실제 라벨과 비교해 학습

~~~
def Malstm(features, labels, mode):
        
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    
    def basic_bilstm_network(inputs, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            #NUM_LAYERS 수만큼 쌓기(fw : 전방, bw : 후방)
            lstm_fw = [
                tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(HIDDEN),output_keep_prob=DROPOUT_RATIO)
                for layer in range(NUM_LAYERS)
            ]
            lstm_bw = [
                tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(HIDDEN),output_keep_prob=DROPOUT_RATIO)
                for layer in range(NUM_LAYERS)
            ]
            # 'MultiRNNCELL'을 통해 여러 층이 쌓인 LSTM을 묶는다.
            multi_lstm_fw = tf.nn.rnn_cell.MultiRNNCell(lstm_fw)
            multi_lstm_bw = tf.nn.rnn_cell.MultiRNNCell(lstm_bw)
            
            # 이후 "bidirectional_dynamic_rnn"기능을 통해 양방향 lstm 구현
            (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = multi_lstm_fw,
                                                       cell_bw = multi_lstm_bw,
                                                       inputs = inputs,
                                                       dtype = tf.float32)
            outputs = tf.concat([fw_outputs, bw_outputs], 2)
            
            return outputs[:,-1,:]
    embedding = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    
    base_embedded_matrix = embedding(features['base'])
    hypothesis_embedded_matrix = embedding(features['hypothesis'])
    
    base_sementic_matrix = basic_bilstm_network(base_embedded_matrix, 'base')
    hypothesis_sementic_matrix = basic_bilstm_network(hypothesis_embedded_matrix, 'hypothesis')
    
    base_sementic_matrix = tf.keras.layers.Dropout(DROPOUT_RATIO)(base_sementic_matrix)
    hypothesis_sementic_matrix = tf.keras.layers.Dropout(DROPOUT_RATIO)(hypothesis_sementic_matrix)
    
    logit_layer = tf.exp(-tf.reduce_sum(tf.abs(base_sementic_matrix - hypothesis_sementic_matrix), axis = 1, keepdims = True))
    logit_layer = tf.squeeze(logit_layer, axis = -1)
    
    if PREDICT:
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  predictions={
                      'is_duplicate':logit_layer
                  })
        #prediction 진행 시, None
    if labels is not None:
        labels = tf.to_float(labels)
        
    loss = tf.losses.mean_squared_error(labels=labels, predictions=logit_layer)
    
    if EVAL:
        accuracy = tf.metrics.accuracy(labels, tf.round(logit_layer))
        eval_metric_ops = {'acc': accuracy}
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  eval_metric_ops= eval_metric_ops,
                  loss=loss)
    elif TRAIN:

        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step)

        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  train_op=train_op,
                  loss=loss)
~~~

> 총 4개의 layer를 쌓아 학습 - lstm의 성능을 높이기 위해서는 층이 깊으면 좋음

##### 모델 설명
1. 함수의 입력값을 임베딩 값으로 바꾸기 : 기준 문장과 대상 문장을 임베딩 된 벡터로 바꾼다.
2. 양방향 lstm 사용 : 따라서 fw lstm, bw lstm 정의) 이후, 두 개를 합쳐주기(bidirectional_dynamic_rnn 사용)
3. 양방향 lstm에서 얻을 수 있는 2개의 return 값 : lstm의 return 값, 마지막 은닉 상태 벡터값 --> 우리가 원하는 것은 의미 벡터를 위해 마지막 은닉 상태의 백터값을 원함. 
    - 따라서 마지막 은닉 상태 벡터를 fw_outputs, bw_outputs에 저장
    - `outputs = tf.concat([fw_outputs, bw_outputs], 2)`  : 두 개를 합쳐줌, lstm이 순방향, 역방향을 모두 학습함으로써 성능 개선에 도움
4. 맨하탄 거리 구해서 스칼라 값으로 만들기

유사도(맨하탄 거리)를 측정했으므로 학습 가능