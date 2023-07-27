# 생성요약파트

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
