---
title: "Compare morphology analysis"
date: 2019-10-20 08:26:28 -0400
categories: NLP
---
<요약><br>
4차 산업혁명의 중심에 명실공히 자리한 인공지능의 정수는 언어라고 할 수 있습니다. 인공지능이 진정한 지능을 발휘하기 위해서는 매개가 되는 언어가 필수이기 때문이지요. 인공지능을 어린아이로 빗대어 보면, 언어를 배우기 위해서는 부모의 풍부한 언어를 듣고 구조를 추론해나가는 과정이 필요합니다. 그러기 위해서는 성능이 우수한 한국어 형태소 분석기와 그를 뒷받침해줄 수 있는 풍부한 부모의 언어가 필요합니다. 그중에서도 한국어 자연어 처리(NLP)에 관심을 갖고 있는 우리는 오픈소스로 공개된 두가지의 형태소 분석기를 비교·분석해보았습니다.<br>
지난 11월, 카카오에서 딥러닝 기술을 기반으로 개발한 형태소 분석기 'khaiii(Kakao Hangul Analyzer III, 카이)'와 『한국어 Word2vec 모델을 위한 최적의 형태소 분석기 선정』에서의 한국어 형태소 분석기 성능 비교에서 최적의 결과값을 보여준 형태소 분석기 mecab-ko에 형태소 분석기가 처리하기 까다로운 유형의 예문으로 실험을 진행해보았습니다.

<데이터 출처><br>
두가지 형태소 분석기의 성능을 비교하기 위해 한국어 형태소 분석기가 쉽게 처리하기 힘든 6가지 종류의 예문을 직접 작성해보았습니다.(3번 문장 제외) 실제로 한국어 사용자가 사용하기에는 다소 무리가 있는 문장이지만, 성능을 비교하기 위한 목적에 맞추기 위한 것으로 이해해주시면 됩니다.

<분석 내용1><br>
작성한 예문을 수행시키기에 앞서, konlpy에서 한국어 형태소 분석기의 성능을 검증하기 위해 쓴 예문을 먼저 수행하게 해보았습니다.
1. "아버지가방에들어가신다"<br>
의미로 보나 형태로 보나 이상적인 분석은 "아버지가/방에/들어가신다"일 것입니다. 띄어쓰기가 행해지지 않은 이 문장을 Mecab과 Khaiii는 어떻게 분석하는지 살펴볼까요?

|Mecab   |Khaiii   |        
|---|---|
| 아버지 / NNG | 아버지 / NNG  |         
|  가 / JKS | 가 / JKS  |         
| 방 / NNG  |  방에 / NNG |   
|에 / JK|들 / VV|
|들어가 / VV|어 / EC|
|신다 / EP+EC|가 / VV|
||시 / EP|
||ㄴ다 / EC|
애꿎은 방에(방아깨비의 평북방언)를 불러내는 Khaiii를 볼 수 있습니다.


2. "나는 밥을 먹는다" VS "하늘을 나는 자동차"<br>
konlpy의 분석을 따르면, 첫번째 문장에서 “나는”은 나/N + 는/J, 두번째 문장에서는 나(-ㄹ다)/V + 는/E 이 되는 것이 바람직하다고 합니다.
과연 각 분석기가 단어의 의미와 주변부를 잘 살피고 있는지 볼까요?

|Mecab   |Khaiii |        
|---|---|
| 나 / NP | 나 / NP  |         
|  는 /  JX | 는 /  JX|         
| 밥 / NNG |  밥 / NNG |   
|을 / JKO	|을 / JKO|
|먹 / VV|먹 / VV|
|는다 / EC|는다 / EC|


|Mecab   |Khaiii |        
|---|---|
| 하늘 / NNG | 하늘 / NNG |         
|  을 / JKO | 을 / JKO|         
|나 / NP|  나 / NP |   
|는 / JX	|는 / JX|
|자동차 / NNG|자동차 / NNG|

다들 밥은 잘 먹지만 하늘을 나는 자동차는 없나봅니다.

3. "아이폰 기다리다 지쳐 애플공홈에서 언락폰질러버렸다6+128기가실버ㅋ"<br>
우리가 흔히 사용할 수 있는, 줄임말과 신조어가 잘 버무려진 문장입니다. 각 형태소 분석기가 사전에 포함되지 않은 단어를 처리하는 방식을 한번 볼까요?

|Mecab   |Khaiii |        
|---|---|
| 아이폰 / NNP | 아이폰 / NNP |         
|  기다리 / VV |기다리 / VV|         
|다 / EC|  다 / EC |   
|지쳐 / VV+EC	|지치 / VV|
|애플 / NNP|어 / EC|
|공홈 / NNG|애플 / NNP|
|에서 / JKB|공홈 / NNG|
|언락 / NNG|에서 / JKB|
|폰 / NNG|언락 / NNG|
|질러 / VV+EC|폰질 / NNG|
|버렸 / VX+EP|리 / VV|
|다 /EC|어 / EC|
|6 / SN|버리 / VX|
|+ / SY|었 / EP|
|128 / SN|다 / EC|
|기 / NNG|  6 / SN |
|가 / JKS|  + / SW|
|실버 / NNP| 128 / SN |
|ㅋ / IC|  기가실 / NNG |
|| 버 / IC |
|| ㅋ / MAG |
실버폰을 질러본 적 없는 Khaiii와 용량을 신경쓰지 않는 Mecab  



<분석 내용2><br>

1. 긴 문장<br>"내가 궁극적으로 그들에게 주고 싶었던 깨달음이라 함은 우주의 언어를 통달하는 것에 이르는 길이 낙타의 걸음으로 고비 사막을 가로지르며 얻는 육체적 고난이 영혼을 불사지를 때 다가오는 고통과 다름아니지만, 티탄의 피로 이루어진 아틀라스가 가이아의 서쪽 끝에서 버티는 안간힘으로는 영원히 이를 수 없는 땅이라는 것이다." <br>
실생활에서 전혀 쓸 것 같지 않은 전형적인 예문입니다. 국어국문학을 전공한 작문능력을 살려, 보다 다양하게 분화된 어미를 써보았습니다.


| Mecab           | Khaiii      |
|-----------------|-------------|
| 내/NP           | 내/NP       |
| 가/JKS          | 가/JKS      |
| 궁금/NNG        | 궁극/NNG    |
| 적/XSN          | 적/XSN      |
| 으로/JKB        | 으로/JKB    |
| 그/NP           | 그/NP       |
| 들/XSN          | 들/XSN      |
| 에게/JKB        | 에게/JKB    |
| 주/VV           | 주/VV       |
| 고/EC           | 고/EC       |
| 싶/VX           | 싶/VX       |
| 었/EP           | 었/EP       |
| 던/ETM          | 던/ETM      |
| 깨달음/NNG      | 깨달음/NNG  |
| 이/VCP          | 이/VCP      |
| 라/EC           | 라/EC       |
| 함/VX+ETN       | 하/VV       |
| 은/JX           | ㅁ/ETN      |
| 우주/NNG        | 은/JX       |
| 의/JKG          | 우주/NNG    |
| 언어/NNG        | 의/JKG      |
| 를/JKO          | 언어/NNG    |
| 통달/NNG        | 를/JKO      |
| 하/XSV          | 통달/NNG    |
| 는/ETM          | 하/XSV      |
| 것/NNB          | 는/ETM      |
| 에/JKB          | 것/NNB      |
| 이르/VV         | 에/JKB      |
| 는/ETM          | 이르/VV     |
| 길/NNG          | 는/ETM      |
| 이/JKS          | 길/NNG      |
| 낙타/NNG        | 이/JKS      |
| 의/JKG          | 낙타/NNG    |
| 걸음/NNG        | 의/JKG      |
| 으로/JKB        | 걸음/NNG    |
| 고비/NNG        | 으로/JKB    |
| 사막/NNG        | 고비/NNG    |
| 을/JKO          | 사막/NNG    |
| 가르지르/VV     | 을/JKO      |
| 며/EC           | 가로지르/VV |
| 얻/VV           | 며/EC       |
| 는/ETM          | 얻/VV       |
| 육체/NNG        | 는/ETM      |
| 적/XSN          | 육체/NNG    |
| 고난/NNG        | 적/XSN      |
| 이/JKS          | 고난/NNG    |
| 영혼/NNG        | 이/JKS      |
| 을/JKO          | 영혼/NNG    |
| 불/XPN          | 을/JKO      |
| 사지/NNG        | 불사지/NNG  |
| 를/JKO          | 를/JKO      |
| 때/NNG          | 때/NNG      |
| 다가오/VV       | 다가오/VV   |
| 는/ETM          | 는/ETM      |
| 고통/NNG        | 고통/NNG    |
| 과/JKB          | 과/JKB      |
| 다름/VA+ETN     | 다르/VA     |
| 아니/VCN        | ㅁ/ETN      |
| 지만/EC         | 아니/VCN    |
| ,/SC            | 지만/EC     |
| 티탄/NNG        | ,/SP        |
| 의/JKG          | 티탄/NNG    |
| 피/NNG          | 의/JKG      |
| 로/JKB          | 피로/NNG    |
| 이루어진/VV+ETM | 이루어지/VV |
| 아틀라스/NNG    | ㄴ/ETM      |
| 가/JKS          | 아/NNG      |
| 가이아/NNP      | 틀라스/NNP  |
| 의/JKG          | 가/JKS      |
| 서쪽/NNG        | 가이아/NNP  |
| 끝/NNG          | 의/JKG      |
| 에서/JKB        | 서쪽/NNG    |
| 버티/VV         | 끝/NNG      |
| 는/ETM          | 에서/JKB    |
| 안간힘/NNG      | 버티/VV     |
| 으로/JKB        | 는/ETM      |
| 는/JX           | 안간힘/NNG  |
| 영원히/MAG      | 으로/JKB    |
| 이를/VV+ETM     | 는/JX       |
| 수/NNB          | 영원히/MAG  |
| 없/VA           | 이/NP       |
| 는/ETM          | 르/VV       |
| 땅/NNG          | ㄹ/ETM      |
| 이/VCP          | 수/NNB      |
| 라는/ETM        | 없/VA       |
| 것/NNB          | 는/ETM      |
| 이/VCP          | 땅/NNG      |
| 다/EF           | 이/VCP      |
| ./SF            | 라는/ETM    |
|                 | 것/NNB      |
|                 | 이/VCP      |
|                 | 다/EF       |
|                 | ./SF        |


2. 줄임말/신조어가 포함된 문장<br>
"이 영화 노잼이야. 갑분싸네. 나 지금 엄근진하다."<br>
'노잼', '갑분싸', '엄근진'과 같은 줄임말과 신조어를 이용한 짧은 문장입니다. Mecab과 Khaiii가 사전에 없는 생소한 단어를 처리하는 능력은 어떨까요?<br>

| Mecab        | Khaiii       |
|--------------|--------------|
| 영화/NNG     | 영화/NNG     |
| 노/NNP       | 노잼/NNG     |
| 잼/NNG       | 이/VCP       |
| 이/VCP       | 야/EF        |
| 야/EF        | ./SF         |
| ./SF         | 갑/MAG       |
| 갑/NNG       | 분/VA        |
| 분/XSN       | 싸/VV        |
| 싸/VA        | 네/EF        |
| 네/EF        | ./SF         |
| ./SF         | 나/NP        |
| 나/NP        | 지금/MAG     |
| 지금/MAG     | 엄근진/XR    |
| 엄/IC        | 하/XSA       |
| 근진/NNG     | 다/EF        |
| 하/XSV       | ./SF         |
| 다/EF        |  |
| ./SF         ||

3. 띄어쓰기가 되지 않은 문장<br>
"늙은이는왜이렇게일찍잠에서깨는지모르겠구나.좀더긴하루를보내고싶어서일까?"<br>
헤밍웨이의 <노인과 바다>에 나오는 문장인데, 하루를 늘려줄 수는 없으니 문장이라도 길게 붙여주고 싶은 마음으로 띄어쓰기를 없애 보았습니다.

| Mecab           | Khaiii      |
|-----------------|-------------|
| 늙은이/NNG     |늙은이/NNG       |
| 는/JX          | 는/JKG     |
| 왜/MAG        | 왜/NNG    |
| 이렇게/MAG          | 이렇/ETN      |
| 일찍/MAG      | 게/NNG |
| 잠/NNG           | 일찍잠/NR       |
| 에서/JKB         | 에서깨/NNG      |
| 깨/VV       | 는/MAG   |
| 는지/EC           | 지/VA       |
| 모르/VV           | 모르/EC       |
| 겠/ EP           | 겠/NNG       |
| 구나/EF          | 구나/NNG       |
| ./SF       | ./NNG    |
| 좀/ MAG     | 좀/NNG  |
| 더/MAG         | 더긴/NNG      |
| 긴/VAETM        | 하루/MAG   |
| 하루/NNG         | 르/VA       |
| 를/JKO           | ㄹ/EC       |
| 보내/VV           | 보내/NNG       |
| 고/EC          | 고/NNG       |
| 싶/VX       | 싶/NNG    |
| 어서/ EC      | 어서/NNG  |
| 일까/ VCPEF      | 이/NNG  |
| ?/SF      | ㄹ까/NNG      |
|      | ?/SF      |


4. 같은 형태, 다른 품사인 단어가 쓰인 문장<br>
"나는 하늘을 나는 돼지가 부러워 그 새를 참지 못하고 새의 등에 올라타 바다에 새겨진 새 도시를 살펴보았다."<br>
아무도 제대로 분석하지 못한 두가지 의미의 '나는'! 그리고 세가지 의미의 '새'를 과연 Khaiii는 분석해낼 수 있을까요?

| Mecab               | Khaiii    |
|---------------------|-----------|
| 나/NP               | 나/NP     |
| 는/JX               | 는/JX     |
| 하늘/NNG            | 하늘/NNG   |
| 을/JKO              | 을/JKO    |
| 나/NP               | 나/NP     |
| 는/JX               | 는/JX     |
| 돼지/NNG            | 돼지/NNG  |
| 가/JKS              | 가/JKS    |
| 부러워/VA+EC        | 부럽/VA   |
| 그/MM               | 어/EC     |
| 새/NNG              | 그/MM     |
| 를/JKO              | 새/NNG    |
| 참/VV               | 를/JKO    |
| 지/EC               | 참/VV     |
| 못하/VX             | 지/EC     |
| 고/EC               | 못하/VX   |
| 새의/NNP            | 고/EC     |
| 등/NNB              | 새/NNG    |
| 에/JKB              | 의/JKG    |
| 올라타/VV+EC        | 등/NNG    |
| 바다/NNG            | 에/JKB    |
| 에/JKB              | 올라타/VV |
| 새겨진/VV+EC+VX+ETM | 아/EC     |
| 새/MM               | 바다/NNG  |
| 도시/NNG            | 에/JKB    |
| 를/JKO              | 새기/VV   |
| 살펴보/VV           | 어/EC     |
| 았/EP               | 지/VX     |
| 다/EF               | ㄴ/ETM    |
| ./SF                | 새/MM     |
|                     | 도시/NNG  |
|                     | 를/JKO    |
|                     | 살펴보/VV |
|                     | 았/EP     |
|                     | 다/EF     |
|                     | ./SF      |

5. 한글 자모가 포함된 문장<br>
"설레ㅁ의 방향으ㄴ ㅈㅣ구오ㅏ 달라서 ㄷㅏ행입니ㄷㅏ"<br>
서둘러 쓰다보면 흔히 저지르는 치명적인 실수지만 우리 눈엔 쉽게 보이지요? 오타를 눈치챌 수 있다면 굉장히 흥미로울텐데요.

| Mecab           | Khaiii      |
|-----------------|-------------|
| 설레/NNG           | 설레ㅁ/NNG       |
| ㅁ/XSN          | 의/JKG     |
| 의/JKG        | 방향으/NNG    |
| 방향/NNG          | ㄴ/ETN      |
| 으/IC        | ㅈ/NNG |
| ㄴ/NNG           | ㅣ/NR       |
| ㅈ/NNG          | 구/NNG      |
| ㅣ/NNG        | 오ㅏ/MAG   |
| 구/NR           | 다르/VA       |
| 오/NR           | 아서/EC       |
| ㅏ/ UNKNOWN           | ㄷ/NNG       |
| 달라서/VAEC          | ㅏ행입/NNG       |
| ㄷ/NNG       | 니ㄷ/NNG    |
| ㅏ행입니다/ UNKNOWN      | ㅏ/NNG  |

6. 사투리가 쓰인 문장<br>
"내 등더리가 어깨 우에가 꼽슬멀꺼디가 후리맞은 싸리빗물로 폭삭 젖을 때꺼정 섰드만 움푹 팬 가심에 애꿎은 빗물만 찬다."<br>
강원도, 경기도, 경상도, 전라도, 충청도, 함경도 방언까지 고루 넣어 만든 한반도 사투리 모음 문장입니다. 우리도 어려운 각 지방의 사투리를 과연 어떻게 분석할 것인지 궁금하지요?

| ﻿Mecab        | Khaiii       |
|--------------|--------------|
| 내 / NP+JKG  | 나/NP        |
| 등더리 / NNG | 의/JKG       |
| 가 / JKS     | 등/NNB       |
| 어깨 / NNG   | 더리/NNG     |
| 우에 / NNP   | 가/JKS       |
| 가 / JKS     | 어깨/NNG     |
| 꼽 / VV      | 우에/NNG     |
| 슬 / MAG     | 가/JKS       |
| 멀 / NP+JKO  | 꼽/NNG       |
| 꺼 / VV+EC   | 슬/VV        |
| 디 / NNG     | 멀/NNG       |
| 가 / JKS     | 꺼디/NNP     |
| 후리 / NNG   | 가/JKS       |
| 맞 / VV      | 후리/NNG     |
| 은 / ETM     | 맞/VV        |
| 싸리 / NNG   | 은/ETM       |
| 빗물 / NNG   | 싸리빗물/NNG |
| 로 / JKB     | 로/JKB       |
| 폭삭 / MAG   | 폭삭/MAG     |
| 젖 / VV      | 젖/VV        |
| 을 / ETM     | 을/ETM       |
| 때 / NNG     | 때꺼정/MAG   |
| 꺼정 / JX    | 서/VV        |
| 섰 / VV+EP   | 었/EP        |
| 드만 / EF    | 드/NNG       |
| 움푹 / MAG   | 만/JX        |
| 팬 / VV+ETM  | 움푹/MAG     |
| 가심 / NNG   | 패/VV        |
| 에 / JKB     | ㄴ/ETM       |
| 애꿎 / VA    | 가심/NNG     |
| 은 / ETM     | 에/JKB       |
| 빗물 / NNG   | 애꿎/VA      |
| 만 / JX      | 은/ETM       |
| 찬다 / VV+EF | 빗물/NNG     |
| . / SF       | 만/JX        |
|              | 차/VV        |
|              | ㄴ다/EF      |
|              | ./SF         |

<결론>