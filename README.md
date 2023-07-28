# 생성요약파트

## 문제 정의
MusicGen 모델의 Input으로 들어가야할 캡션의 구성을 확인했을때, 기본적으로 분위기/악기/장르/템포 등의 구성을 기대하고 있으며 간단한 문장단위의 캡션 또한 입력으로 사용할 수 있습니다. 텍스트에 어울리는 음악 생성을 위해 입력받은 텍스트를 캡션에 반영해야했고 전체 텍스트를 넣을 수는 없었기에 요약 모델을 통하여 문서 압축을 진행하였습니다. 

## 모델 선정
### 한국어 모델
| MODEL_NAME | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
| --- | --- | --- | --- | --- |
| eenzeenee/t5-base-korean-summarization | 0.029444 | 0.005 | 0.029405 | 0.029444 |
| noahkim/KoBigBird-KoBart-News-Summarization | 0.000556 | 0.0 | 0.000556 | 0.000556 |
| gogamza/kobart-summarization | 0.027103 | 0.005222 | 0.0265 | 0.025921 |

### 영어 모델
| MODEL_NAME | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
| --- | --- | --- | --- | --- |
| google/pegasus-cnn_dailymail | 0.251438 | 0.079138 | 0.178306 | 0.177947 |
| google/pegasus-xsum | 0.185752 | 0.040767 | 0.140719 | 0.140715 |
| knlpscience/pegasus-samsum | 0.271225 | 0.094884 | 0.196447 | 0.195872 |
| facebook/bart-cnn_dailymail | 0.258624 | 0.085612 | 0.180948 | 0.180924 |
| pszemraj/pegasus-x-large-book-summary | 0.209288 | 0.051587 | 0.143579 | 0.143580 |

Musicgen 모델의 입력으로 영어로 이루어진 캡션을 기대하므로, 중간에 번역과정이 불가피했고 번역의 위치를 정하기 위해 한국어 모델과 영어 모델을 비교하고자 하였습니다.
최종적으로 ROUGE 지표에서 더 나은 성능을 보이는 모델은 영어 모델이었으며 google의 pegasus를 기반 모델로 선정하였습니다.
따라서, 입력받은 텍스트를 한->영 번역을 거친 후 pegasus모델로 요약을 하는 파이프라인을 구축하였습니다.
우리의 프로젝트에서 입력으로 기대하는 것은 책/소설과 같은 류의 텍스트이므로 이와 유사한 한국 문학 데이터 약 600개로 평가하였으며, 영어 모델은 이 데이터셋을 Papago번역을 통하여 
영문으로 변환시킨 데이터 셋으로 평가되었습니다. 번역이라는 정보 손실의 패널티를 감안하더라도 영어 모델의 성능이 더 좋았음을 위의 평가표에서 확인할 수 있습니다.

## PEGASUS 모델 아키텍처
PEGASUS는 자연어처리 분야의 가장 어려운 과제 중 하나인 생성 요약(Abstractive- summarization) Task를 위해 $GSG^{(Gap-Sentence -Generation)}$라는 새로운 사전학습 기법을 적용하여 소개된 인코더-디코더 모델이며 사전훈련 시 자기지도학습의 목표가 최종적인 다운스트림 과제에 가까울수록 fine-tuning 성능이 좋을 것이라는 가설에서 출발합니다. $GSG$는 기존 BERT처럼 토큰 단위 마스킹이 아닌 문장 단위 마스킹을 한다는 점이 특별합니다. 즉, 몇 개의 온전한 문장들이 입력 문서로부터 제거되고, 모델은 이 문장을 예측 해야합니다.

PEGASUS-X는 PEGASUS를 보완한 버전입니다. 기존 PEGASUS보다 더 긴 텍스트 입력(최대 16K 토큰)을 처리하기위해 PEGASUS 모델의 확장이며 PEGASUS-X보다 훨씬 큰 모델에 필적하는 긴 입력 요약 작업에서 강력한 성능을 달성하는 동시에 추가 매개변수를 거의 추가하지 않고 훈련에 모델 병렬 처리가 필요하지 않은 장점이있습니다. 토크나이저는 PEGASUS와 동일합니다.

## 2차 fine-tuning 데이터 셋
<요약문 및 레포트 생성 데이터 中> 문학 분야 약 5000개 데이터에 대하여 영문으로 번역한 데이터 셋 | AI hub 


---
# 분류 모델

## 문제 정의

텍스트 입력은 블로그, 소설, SNS 게시글 등 모든 장르의 문어체 텍스트로 특정하였습니다. 데이터셋을 직접 구성하는 것은 시간적 제약으로 불가능하여, uchidalab book-dataset을 활용하여 약 20만개의 책 제목을 32개의 장르로 분류하였습니다.

## 데이터셋

데이터셋의 경우 핵심적인 판단 요소는 ‘텍스트의 양’과 ‘장르의 다양성’이었으며, 직접적으로 본문과 장르를 연결한 데이터셋으로 5만개의 movie description-genre dataset과 20만개의 book title-genre dataset 중 텍스트의 길이는 비교적 짧지만 양이 많고 장르적 다양성을 가진 후자를 Fine-tuning 데이터로 선정하였습니다.

- 데이터셋: uchidalab book-dataset
- 데이터 수: 약 20만개의 책 제목
- 장르 수: 32개의 장르

## 모델 구성

음악을 생성하기 위해 사용할 장르를 musicgen 모델에서 사용 가능한 133개의 장르 중에서 유사한 장르를 각각 5개씩 선정하고, 이 중에서 2개의 장르를 랜덤하게 선택하여 사용하는 형태로 모델을 구성하였습니다.

| Label | Category Name | 음악 |
| --- | --- | --- |
| 0 | Arts & Photography | Smooth Rock, Ethereal, Minimalism, Soft Rock, Ambient |
| 1 | Biographies & Memoirs | Lively, Swing, Jazz, Ballad, Piano |
| 2 | Business & Money | Glitch House, Techno, Ambient, EDM, Minimalism |
| 3 | Calendars | Acoustic, Lullaby, Piano, New Age, Meditation |
| 4 | Children's Books | Children, Lullaby, Piano, Folk, World Music |
| 5 | Comics & Graphic Novels | Rocktronica, Industrial, Punk, Heavy Metal, Blues |
| 6 | Computers & Technology | Synth Pop, EDM, Glitch House, Electronic, Techno |
| 7 | Cookbooks, Food & Wine | Smooth Jazz, Lounge, Jazz, Blues, Soul |
| 8 | Crafts, Hobbies & Home | Folk, Indie Folk, Lively, Acoustic, Soft Rock |
| 9 | Christian Books & Bibles | Gospel, Hymns, Inspirational, Christian Music, Piano |
| 10 | Engineering & Transportation | Suspense, Orchestral, Racing, Electronic, Cinematic |
| 11 | Health, Fitness & Dieting | Progressive House, Electronic, Motivational, Ambient, New Age |
| 12 | History | Classical, Orchestral, Piano, Cinematic, Dramedy |
| 13 | Humor & Entertainment | Jazz, Lounge, Quirky, Comedy, Funk |
| 14 | Law | Blues, Suspense, Reggae, Electronic, Cinematic |
| 15 | Literature & Fiction | Cinematic, Orchestral, Fantasy, Piano, Dramedy |
| 16 | Medical Books | Ambient, Synth Pop, Electronic, New Age, Meditation |
| 17 | Mystery, Thriller & Suspense | Suspense, Electronic, Synth Pop, Blues, Cinematic |
| 18 | Parenting & Relationships | Ballad, Lullaby, Piano, Acoustic, Folk |
| 19 | Politics & Social Sciences | Orchestral, Blues, Suspense, Cinematic, Jazz |
| 20 | Reference | World Elements, Jazz, Lounge, Ethnic, Acoustic |
| 21 | Religion & Spirituality | Gospel, Hymns, Inspirational, Christian Music, Piano |
| 22 | Romance | Hawaiian, Ballad, Electronic, Pop, Acoustic |
| 23 | Science & Math | Electronic, Experimental, Synth Pop, Ambient, New Age |
| 24 | Science Fiction & Fantasy | Cinematic, Orchestral, Fantasy, Piano, Dramedy |
| 25 | Self-Help | Ambient, Meditation, New Age, Soft Rock, Piano |
| 26 | Sports & Outdoors | Sports, Electronic, Heavy Metal, Rock, Punk |
| 27 | Teen & Young Adult | Pop, Indie Pop, Lively, Acoustic, Soft Rock |
| 28 | Test Preparation | Trailer, Epic, Cinematic, Action, Dramatic |
| 29 | Travel | World, Lively, Folk, Ethnic, Acoustic |
| 30 | Gay & Lesbian | Pop, Dance, Electronic, Soul, R&B |
| 31 | Education & Teaching | Lively, Children, Piano, Acoustic, Soft Rock |

## 모델 선택

모델 선택 기준은 작동 시간과 정확도로 정했습니다. Roberta-large, Roberta-base, BERT, Distilbert 등을 비교한 결과, BERT-base-uncased 모델이 가장 높은 정확도를 보여주면서도 중간 정도의 작동 시간을 가졌기 때문에 이를 사용하였습니다.

| No. | model | args | accuracy |
| --- | --- | --- | --- |
| 1 | roberta-base | batch64,lr5e-05 | 0.693 ( - ) |
| 2 | roberta-large | batch8, lr1e-05 | 0.694 (+0.001) |
| 3 | distilbert-base-uncased | batch32, lr2e-05 | 0.694 (+0.001) |
| 4 | distilbert-base-uncased | batch16, lr5e-06 | 0.682 (-0.011) |
| 5 | distilbert-base-uncased | batch16, lr1e-05 | 0.695 (+0.002) |
| 6 | bert-base-uncased | batch64, lr5e-05 | 0.708 (+0.015) |

---
# 음악 도메인 특화 감정 분류 모델

## 1. 문제 정의

- 음악은 사람의 마음을 움직이는 힘을 가지고 있습니다. 음악이 가지고 있는 감정은 듣는 사람들로 하여금 그 기분을 간접적으로 경험하게 한다고 생각하고 있습니다.
- 저는 사용자가 입력한 텍스트에 대한 감정 분석(Emotion Analysis)를 통해 음악을 만들기 위한 키워드로 활용하고자 연구를 진행하였습니다.

## 2. 데이터셋

![Untitled](https://file.notion.so/f/s/2e9013a1-3447-4bd5-8d2e-a89ad1774d7d/Untitled.png?id=17baf033-1b72-4e38-88f7-7f0db6588046&table=block&spaceId=0825c815-092a-430c-9d39-95d69099fbe9&expirationTimestamp=1690596000000&signature=cpznrVUjrpsioUervYdL2Wf3jik17CuZIRfStxBBPjI&downloadName=Untitled.png)

- 감정 분류 : 감정을 분류하는 기준은 다양하게 있지만 **폴 에크만**의 감정 분석을 바탕으로 하였습니다.
- 연구에 맞춰 인류의 보편적인 감정 7가지**(역겨움(Disgust), 슬픔(Sadness), 공포(Fear), 분노(Angry), 놀람(Surprise), 기쁨(Happy), 중립(Neutral))**로 나눠 데이터셋을 수집하였습니다.
- **데이터셋** : AIHub ****[감정 분류를 위한 대화 음성 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263)****
    - 약 2만 개의 대화 데이터셋
    - 7가지 감정**(happiness, angry, disgust, fear, neutral, sadness, surprise)**에 대해 라벨링

## 3. 모델

- 감정 분류(Emotion classification)에서 사용하는 모델 조사를 진행하였습니다.
    
    
    | Model | Language |
    | --- | --- |
    | SKTBrain/KoBERT | 한국어 |
    | bhadresh-savani/distilbert-base-uncased-emotion | 영어 |
    | text2emotion | 영어 |
- MusicGen 모델이 영어 캡션 기반이기 때문에 결과물을 영어로 만들어주어야 한다는 점을 들어 한국어 모델과 영어 모델 두가지 경우를 서칭을 진행하였습니다.
    
    ### 한국어 모델
    
    - 한국어 모델 같은 경우에도 NSMC 데이터 기반 감정 분류 모델은 많았지만 이는 긍정/부정만 구분하는 모델이라 적합하다고 생각하지 않았습니다.
    - 이에 KoBERT 모델을 다중 분류 데이터를 사용해 모델 학습 및 튜닝을 진행하였습니다.
    
    ### 영어 모델
    
    - 영어 모델의 경우, 대부분 긍정(Positive)/부정(Negetive)를 보여주는 경우가 많아 다중 분류를 하기 어려웠습니다.
    - 그 가운데 `distilbert-base-uncased-emotion 다중 분류 모델`과 `text2emotion 라이브러리`가 감정 다중 분류 도메인에 적합하다고 판단하고 테스트를 진행하였습니다.
    - 결과로 `KoBERT`보다 좋은 모습을 보여주지는 못했고, 데이터를 `번역`하면 문장의 의미가 달라지는 경우가 있어 적합하지 않다고 판단했습니다.
        
        
        | Model | Accuracy |
        | --- | --- |
        | SKTBrain/KoBERT | 0.72 |
        | bhadresh-savani/distilbert-base-uncased-emotion | 0.64 |
        | text2emotion | 0.62 |
- 이에 KoBERT로 모델 서빙을 진행하기 위해 데이터셋을 기반으로 모델 학습 및 튜닝을 진행하였습니다.
    
    
    | ID | epoch | batch | max_len | Train_acc | Test_acc |
    | --- | --- | --- | --- | --- | --- |
    | 0 | 5 | 16 | 256 | 0.9763 | 0.9328 |
    | 1 | 5 | 32 | 256 | 0.9780 | 0.9313 |
    | 2 | 5 | 64 | 256 | 0.9737 | 0.9342 |
    | 3 | 10 | 16 | 256 | 0.9910 | 0.9320 |
    | 4 | 10 | 32 | 256 | 0.9920 | 0.9328 |
    | 5 | 10 | 64 | 256 | 0.9903 | 0.9316 |
    - Train/Test 결과를 보면 정확도는 매우 높게 나왔지만 Inference를 위해 다른 외부데이터를 사용하면 결과에서 보이는 것과 달리 종종 맞지 않는 결과가 나오곤 하였습니다.
    - 이에 7가지 감정으로 라벨링을 하는 것이 아니라 다른 방식으로 라벨링을 하면 결과를 확실히 얻을 수 있지 않을까 하여 추가 조사를 진행하였습니다.
