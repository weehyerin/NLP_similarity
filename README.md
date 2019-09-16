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

질문 길이 최대 값: 1169
질문 길이 평균 값: 59.82
질문 길이 표준편차: 31.96
질문 길이 중간 값: 51.0
질문 길이 제 1 사분위: 39.0
질문 길이 제 3 사분위: 72.0

#### 사용 단어 개수 세기
1. 단어 쪼개기
~~~
train_word_counts = train_set.apply(lambda x:len(x.split(' ')))
~~~

![image](https://user-images.githubusercontent.com/37536415/64830831-8bce9000-d60d-11e9-81db-195d16b889f1.png)

질문 단어 개수 최대 값: 237
질문 단어 개수 평균 값: 11.06
질문 단어 개수 표준편차: 5.89
질문 단어 개수 중간 값: 10.0
질문 단어 개수 제 1 사분위: 7.0
질문 단어 개수 제 3 사분위: 13.0
질문 단어 개수 99 퍼센트: 31.0

