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
