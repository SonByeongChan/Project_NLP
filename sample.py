import pandas as pd

## 불용어가 제거된 train_csv 호출
clean_train_news = pd.read_csv(r'F:\AI_KDT7\NLP\mini\new_news.csv')

clean_train_news.info()

## 클린 val데이터 셋 호출
celans_val_news = pd.read_csv(r'F:\AI_KDT7\NLP\mini\val_new_news.csv')

celans_val_news.info()


## train셋 및 테스트 셋 로드 // 라벨도 같이
train = clean_train_news['cleaned_Content']
test = celans_val_news['cleaned_Content']

train_label = clean_train_news['clickbaitClass']
test_label = celans_val_news['clickbaitClass']


from collections import Counter
import pickle

def build_vocab(corpus, n_vocab, special_tokens, save_path=None, save_path_txt=None):
    counter = Counter()

    print("🛠️ 단어 수 세는 중...")
    for i, tokens in enumerate(corpus):
        counter.update(tokens)
        if (i + 1) % 100000 == 0:
            print(f"  → {i + 1}개 문서 처리 완료")

    vocab = special_tokens.copy()
    for token, count in counter.most_common(n_vocab):
        vocab.append(token)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"✅ Vocab 저장 완료 (pickle): {save_path}")

    if save_path_txt:
        with open(save_path_txt, "w", encoding="utf-8") as f:
            for token in vocab:
                f.write(f"{token}\n")
        print(f"✅ Vocab 저장 완료 (txt): {save_path_txt}")

    print(f"📦 최종 vocab 크기: {len(vocab)}개")
    return vocab

# train 데이터 토큰 처리
train_tokens = []
print("🚀 train 데이터 토큰 처리 중...")
for i, text in enumerate(train):
    train_tokens.extend(text.split())
    if (i + 1) % 10000 == 0:
        print(f"  → {i + 1}개 행 처리 완료")

# test 데이터 토큰 처리
test_tokens = []
print("🚀 test 데이터 토큰 처리 중...")
for i, text in enumerate(test):
    test_tokens.extend(text.split())
    if (i + 1) % 10000 == 0:
        print(f"  → {i + 1}개 행 처리 완료")

# 단어 사전 구축
vocab = build_vocab([train_tokens], n_vocab=500000, special_tokens=["<PAD>", "<UNK>"],
                    save_path="vocab.pkl", save_path_txt="vocab.txt")

# 토큰과 ID 매핑
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

# 결과 출력
print(vocab[:100])  # 상위 100개 단어 출력
print(len(vocab))   # 최종 vocab 크기 출력


import json

# 사전 정의
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

# 파일로 저장
with open('token_to_id.json', 'w', encoding='utf-8') as f:
    json.dump(token_to_id, f, ensure_ascii=False)

with open('id_to_token.json', 'w', encoding='utf-8') as f:
    json.dump(id_to_token, f, ensure_ascii=False)

import numpy as np
# 정수 인코딩 및 패딩
def pad_sequences(sequences, max_length, pad_value):
    result = list()
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)
    return np.asarray(result)

unk_id = token_to_id["<UNK>"]

train_ids = [[token_to_id.get(token, unk_id) for token in news] for news in train]
test_ids = [[token_to_id.get(token, unk_id) for token in news] for news in test]

max_length = 2000
pad_id = token_to_id["<PAD>"]
train_ids = pad_sequences(train_ids, max_length, pad_id)
test_ids = pad_sequences(test_ids, max_length, pad_id)

print(train_ids[0])
print(test_ids[0])

import torch
from torch import nn


class SentenceClassifier(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_dim,
        embedding_dim,
        n_layers,
        dropout=0.5,
        bidirectional=True,
        model_type="lstm",
        pretrained_embedding=None
    ):
        super().__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embedding, dtype=torch.float32)
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=n_vocab,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
        
        if model_type == "rnn":
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        elif model_type == "lstm":
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )

        if bidirectional:
            self.classifier = nn.Linear(hidden_dim * 2, 1)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        return logits
    
from torch.utils.data import TensorDataset, DataLoader

train_ids = torch.tensor(train_ids, dtype=torch.long)
test_ids = torch.tensor(test_ids, dtype=torch.long)

train_labels = torch.tensor(train_label, dtype=torch.float32)
test_labels = torch.tensor(test_label, dtype=torch.float32)

train_dataset = TensorDataset(train_ids, train_labels)
test_dataset = TensorDataset(test_ids, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# a모델 파라미터 설정
from torch import optim

n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

classifier = SentenceClassifier(
   n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers
).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.RMSprop(classifier.parameters(), lr=0.001)

import os
import torch
import numpy as np

# 모델 저장 함수
def save_model(model, epoch, train_accuracy, test_accuracy):
    model_path = f"model_epoch{epoch}_train{train_accuracy:.4f}_test{test_accuracy:.4f}.pth"  # 에포크, 훈련 정확도, 테스트 정확도를 포함한 모델 이름
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, model_path)
    print(f"Model saved at epoch {epoch} with train accuracy {train_accuracy:.4f} and test accuracy {test_accuracy:.4f}.")

# 훈련 함수
def train(model, datasets, criterion, optimizer, device, interval, epoch):
    model.train()
    losses = list()
    corrects = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        labels = labels.unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % interval == 0 or step == 0:  # 첫 번째 스텝도 출력
            print(f"Train Loss {step} : {np.mean(losses) if losses else 0}")

    # 훈련 정확도 계산
    train_accuracy = np.mean(np.array(corrects)) if corrects else 0.0

    # 에포크가 끝난 후 테스트하여 정확도 계산
    test_accuracy = test(model, datasets, criterion, device)

    # 모델 저장
    save_model(model, epoch, train_accuracy, test_accuracy)

# 테스트 함수
def test(model, datasets, criterion, device):
    model.eval()
    losses = list()
    corrects = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        labels = labels.unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        yhat = torch.sigmoid(logits) > 0.45
        corrects.extend(torch.eq(yhat, labels).cpu().tolist())

    accuracy = np.mean(corrects)  # 정확도 계산
    print(f"Val Loss : {np.mean(losses) if losses else 0}, Val Accuracy : {accuracy:.4f}")
    return accuracy  # 정확도 반환


# 훈련 및 테스트 실행
epochs = 30
interval = 1000

for epoch in range(epochs):
    train(classifier, train_loader, criterion, optimizer, device, interval, epoch)
    test(classifier, test_loader, criterion, device)
    
    
    
'''
Microsoft Windows [Version 10.0.26100.3476]
(c) Microsoft Corporation. All rights reserved.

(NLP) F:\AI_KDT7\NLP\mini>C:/Users/KDT/anaconda3/envs/NLP/python.exe f:/AI_KDT7/NLP/mini/sample.py
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 80977 entries, 0 to 80976
Data columns (total 5 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   newsTitle        80977 non-null  object
 1   newsContent      80977 non-null  object
 2   clickbaitClass   80977 non-null  int64 
 3   cleaned_title    80977 non-null  object
 4   cleaned_Content  80977 non-null  object
dtypes: int64(1), object(4)
memory usage: 3.1+ MB
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10122 entries, 0 to 10121
Data columns (total 5 columns):
 #   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   newsTitle        10122 non-null  object
 1   newsContent      10122 non-null  object
 2   clickbaitClass   10122 non-null  int64
 3   cleaned_title    10122 non-null  object
 4   cleaned_Content  10122 non-null  object
dtypes: int64(1), object(4)
memory usage: 395.5+ KB
🚀 train 데이터 토큰 처리 중...
  → 10000개 행 처리 완료
  → 20000개 행 처리 완료
  → 30000개 행 처리 완료
  → 40000개 행 처리 완료
  → 50000개 행 처리 완료
  → 60000개 행 처리 완료
  → 70000개 행 처리 완료
  → 80000개 행 처리 완료
🚀 test 데이터 토큰 처리 중...
  → 10000개 행 처리 완료
🛠️ 단어 수 세는 중...
✅ Vocab 저장 완료 (pickle): vocab.pkl
✅ Vocab 저장 완료 (txt): vocab.txt
📦 최종 vocab 크기: 500002개
['<PAD>', '<UNK>', '있는', '것으로', '말했다', '밝혔다', '국내', '위한', '다양한', '지난해', '등을', '서비스', '글로벌', '관계자는', '올해', '서비스를', '관련', '따르면', '시장', '기존', '계획이다', '5G', '기술', '미국', '새로운', '대표는', '있도록', '이후', '현재', '예정이다', '기술을', '최대', '사업', '세계', 'AI', '데이터', '했다', '모바일', '가장', '이를', '것이라고', '대비', '기업', '주요', '스마트폰', '플랫폼', '이어', '아니라', '특히', '디지털', '설명했다', '게임', '큰', '콘텐츠', '개발', '있다고', '보안', '많은', '것이', '높은', '온라인', '제품', '사업을', '반도체', '등이', '전망이다', '업계', '제공한다', '해외', '있다는', '것은', '것이다', '기능을', '배터리', '것”이라고', '가운데', '모든', '직접', '매출', '오는', '이날', '신규', '강조했다', '때문이다', '한국', '중 이다', '향후', '있어', '이에', '대상으로', '고객', '기반으로', '삼성전자', '등에', '핵심', '전체', '등의', '데이터를', '역시', '시장에서']
500002
[1 1 1 ... 0 0 0]
[1 1 1 ... 0 0 0]
f:\AI_KDT7\NLP\mini\sample.py:191: UserWarning: Failed to initialize NumPy: DLL load failed while importing _multiarray_umath: 지정된 모듈을 찾을 수 없습니다. (Triggered internally at ..\torch\csrc\utils\tensor_numpy.cpp:84.)
  train_ids = torch.tensor(train_ids, dtype=torch.long)
cuda
Train Loss 0 : 0.687701940536499
Train Loss 1000 : 0.690537869156181
Train Loss 2000 : 0.6899993157815719
Train Loss 3000 : 0.6867176660276659
Train Loss 4000 : 0.6819672094780456
Train Loss 5000 : 0.6776217913846926
Val Loss : 0.6517186834377905, Val Accuracy : 0.6257
Model saved at epoch 0 with train accuracy 0.0000 and test accuracy 0.6257.
Val Loss : 0.6557765043742284, Val Accuracy : 0.6180
Train Loss 0 : 0.6969121694564819
Train Loss 1000 : 0.6629427422831704
Train Loss 2000 : 0.6607108939027619
Train Loss 3000 : 0.6584119425440422
Train Loss 4000 : 0.6579226959693196
Train Loss 5000 : 0.6577902464682139
Val Loss : 0.6772013985912868, Val Accuracy : 0.6077
Model saved at epoch 1 with train accuracy 0.0000 and test accuracy 0.6077.
Val Loss : 0.6808826335798508, Val Accuracy : 0.6043
Train Loss 0 : 0.7028490304946899
Train Loss 1000 : 0.649872032852916
Train Loss 2000 : 0.6532877840023527
Train Loss 3000 : 0.652854598778083
Train Loss 4000 : 0.6526635112657573
Train Loss 5000 : 0.6526630612950972
Val Loss : 0.6475759139972589, Val Accuracy : 0.6255
Model saved at epoch 2 with train accuracy 0.0000 and test accuracy 0.6255.
Val Loss : 0.6517598121934592, Val Accuracy : 0.6175
Train Loss 0 : 0.5594261288642883
Train Loss 1000 : 0.6521975702339119
Train Loss 2000 : 0.6492801676953452
Train Loss 3000 : 0.6491659203893699
Train Loss 4000 : 0.6490764432446833
Train Loss 5000 : 0.6489065809980247
Val Loss : 0.6671629432619725, Val Accuracy : 0.6150
Model saved at epoch 3 with train accuracy 0.0000 and test accuracy 0.6150.
Val Loss : 0.6692235231352455, Val Accuracy : 0.6128
Train Loss 0 : 0.5464152097702026
Train Loss 1000 : 0.6448695331841677
Train Loss 2000 : 0.6425227754447295
Train Loss 3000 : 0.6419738600826868
Train Loss 4000 : 0.6421270768095988
Train Loss 5000 : 0.6406770309456061
Val Loss : 0.6514672975455853, Val Accuracy : 0.6186
Model saved at epoch 4 with train accuracy 0.0000 and test accuracy 0.6186.
Val Loss : 0.6583696016673981, Val Accuracy : 0.6133
Train Loss 0 : 0.715004563331604
Train Loss 1000 : 0.6303326947527094
Train Loss 2000 : 0.6333023131578818
Train Loss 3000 : 0.632241023426968
Train Loss 4000 : 0.6311170604490572
Train Loss 5000 : 0.6298522391192461
Val Loss : 0.6575313290439072, Val Accuracy : 0.6028
Model saved at epoch 5 with train accuracy 0.0000 and test accuracy 0.6028.
Val Loss : 0.6636809553674423, Val Accuracy : 0.5981
Train Loss 0 : 0.5322986245155334
Train Loss 1000 : 0.6240622783219302
Train Loss 2000 : 0.6267620827602423
Train Loss 3000 : 0.6260043004181972
Train Loss 4000 : 0.6253645678991915
Train Loss 5000 : 0.6258842333439135
Val Loss : 0.6364723936294389, Val Accuracy : 0.6577
Model saved at epoch 6 with train accuracy 0.0000 and test accuracy 0.6577.
Val Loss : 0.6426021167640625, Val Accuracy : 0.6480
Train Loss 0 : 0.5947551727294922
Train Loss 1000 : 0.6203553901685701
Train Loss 2000 : 0.621757663410345
Train Loss 3000 : 0.6205774984412176
Train Loss 4000 : 0.6211203946318814
Train Loss 5000 : 0.6203057830261245
Val Loss : 0.623354566457572, Val Accuracy : 0.6303
Model saved at epoch 7 with train accuracy 0.0000 and test accuracy 0.6303.
Val Loss : 0.6340256825819212, Val Accuracy : 0.6238
Train Loss 0 : 0.500629723072052
Train Loss 1000 : 0.6161951077627492
Train Loss 2000 : 0.6186711972412736
Train Loss 3000 : 0.6177168713137136
Train Loss 4000 : 0.6169356524765536
Train Loss 5000 : 0.6168708250990297
Val Loss : 0.6629767092448927, Val Accuracy : 0.5929
Model saved at epoch 8 with train accuracy 0.0000 and test accuracy 0.5929.
Val Loss : 0.6746574177937862, Val Accuracy : 0.5865
Train Loss 0 : 0.5662522315979004
Train Loss 1000 : 0.612961124855798
Train Loss 2000 : 0.6124288397303586
Train Loss 3000 : 0.6114024940191687
Train Loss 4000 : 0.6123884715488094
Train Loss 5000 : 0.6122626796874302
Val Loss : 0.6348792752085444, Val Accuracy : 0.6294
Model saved at epoch 9 with train accuracy 0.0000 and test accuracy 0.6294.
Val Loss : 0.644341866363658, Val Accuracy : 0.6215
Train Loss 0 : 0.46028590202331543
Train Loss 1000 : 0.6089334056808517
Train Loss 2000 : 0.6103699411498732
Train Loss 3000 : 0.6108906799278748
Train Loss 4000 : 0.6107298998542501
Train Loss 5000 : 0.6097368119573908
Val Loss : 0.6399121774557032, Val Accuracy : 0.6162
Model saved at epoch 10 with train accuracy 0.0000 and test accuracy 0.6162.
Val Loss : 0.6511543242946431, Val Accuracy : 0.6077
Train Loss 0 : 0.37799739837646484
Train Loss 1000 : 0.6081999961908285
Train Loss 2000 : 0.6093088791913714
Train Loss 3000 : 0.610525431443516
Train Loss 4000 : 0.6088679522774989
Train Loss 5000 : 0.6073496218110008
Val Loss : 0.6512504716321725, Val Accuracy : 0.6188
Model saved at epoch 11 with train accuracy 0.0000 and test accuracy 0.6188.
Val Loss : 0.665433462950479, Val Accuracy : 0.6087
Train Loss 0 : 0.8179389238357544
Train Loss 1000 : 0.6023257813908599
Train Loss 2000 : 0.6021510393604048
Train Loss 3000 : 0.6032683145241196
Train Loss 4000 : 0.6041747847145065
Train Loss 5000 : 0.6036699246857267
Val Loss : 0.6242409951481505, Val Accuracy : 0.6801
Model saved at epoch 12 with train accuracy 0.0000 and test accuracy 0.6801.
Val Loss : 0.6392589258545006, Val Accuracy : 0.6599
Train Loss 0 : 0.4683789610862732
Train Loss 1000 : 0.5983457899653352
Train Loss 2000 : 0.5974502728201043
Train Loss 3000 : 0.5994344590326739
Train Loss 4000 : 0.599297730260061
Train Loss 5000 : 0.5993898839050473
Val Loss : 0.6341561932353718, Val Accuracy : 0.6771
Model saved at epoch 13 with train accuracy 0.0000 and test accuracy 0.6771.
Val Loss : 0.6466369089357096, Val Accuracy : 0.6488
Train Loss 0 : 0.6008052825927734
Train Loss 1000 : 0.591920014027949
Train Loss 2000 : 0.5944472432345048
Train Loss 3000 : 0.5960080048137012
Train Loss 4000 : 0.5959341695042915
Train Loss 5000 : 0.5958766571594891
Val Loss : 0.6188355347282186, Val Accuracy : 0.6698
Model saved at epoch 14 with train accuracy 0.0000 and test accuracy 0.6698.
Val Loss : 0.638411644981559, Val Accuracy : 0.6492
Train Loss 0 : 0.7763878703117371
Train Loss 1000 : 0.5925827694760931
Train Loss 2000 : 0.5922337295650423
Train Loss 3000 : 0.593963029751258
Train Loss 4000 : 0.5933634875581074
Train Loss 5000 : 0.5927122135182377
Val Loss : 0.623892920224952, Val Accuracy : 0.6826
Model saved at epoch 15 with train accuracy 0.0000 and test accuracy 0.6826.
Val Loss : 0.6480049164656779, Val Accuracy : 0.6552
Train Loss 0 : 0.5877659320831299
Train Loss 1000 : 0.5901620028378604
Train Loss 2000 : 0.5924206972717941
Train Loss 3000 : 0.5932957462889479
Train Loss 4000 : 0.5921831079808749
Train Loss 5000 : 0.5911243416576046
Val Loss : 0.6396322078746752, Val Accuracy : 0.6594
Model saved at epoch 16 with train accuracy 0.0000 and test accuracy 0.6594.
Val Loss : 0.6606728496709706, Val Accuracy : 0.6358
Train Loss 0 : 0.9061715006828308
Train Loss 1000 : 0.5889644462030965
Train Loss 2000 : 0.5859770805969171
Train Loss 3000 : 0.586960552742544
Train Loss 4000 : 0.5864727307232998
Train Loss 5000 : 0.5873507101388961
Val Loss : 0.6250975123922705, Val Accuracy : 0.6768
Model saved at epoch 17 with train accuracy 0.0000 and test accuracy 0.6768.
Val Loss : 0.6505245566132698, Val Accuracy : 0.6461
Train Loss 0 : 0.3363959491252899
Train Loss 1000 : 0.5805508523554235
Train Loss 2000 : 0.5817564817323261
Train Loss 3000 : 0.5830365900336326
Train Loss 4000 : 0.5832675092222094
Train Loss 5000 : 0.5839058558801679
Val Loss : 0.5996845177761513, Val Accuracy : 0.6964
Model saved at epoch 18 with train accuracy 0.0000 and test accuracy 0.6964.
Val Loss : 0.6294311915434556, Val Accuracy : 0.6597
Train Loss 0 : 0.4618662893772125
Train Loss 1000 : 0.5793428974789935
Train Loss 2000 : 0.5786387404759248
Train Loss 3000 : 0.5792829046266074
Train Loss 4000 : 0.5804581527068179
Train Loss 5000 : 0.582494982497212
Val Loss : 0.6052961000831925, Val Accuracy : 0.6928
Model saved at epoch 19 with train accuracy 0.0000 and test accuracy 0.6928.
Val Loss : 0.6328012831685668, Val Accuracy : 0.6592
Train Loss 0 : 0.36210960149765015
Train Loss 1000 : 0.5802425738576648
Train Loss 2000 : 0.5803677184709247
Train Loss 3000 : 0.580065438505889
Train Loss 4000 : 0.5805018903597746
Train Loss 5000 : 0.5808907279882924
Val Loss : 0.6241498235973715, Val Accuracy : 0.6828
Model saved at epoch 20 with train accuracy 0.0000 and test accuracy 0.6828.
Val Loss : 0.6520441047859041, Val Accuracy : 0.6485
Train Loss 0 : 0.3917850852012634
Train Loss 1000 : 0.5718511378431653
Train Loss 2000 : 0.5725505911800517
Train Loss 3000 : 0.5733058428126786
Train Loss 4000 : 0.5765340646835483
Train Loss 5000 : 0.5765090177587642
Val Loss : 0.6060542012982479, Val Accuracy : 0.6879
Model saved at epoch 21 with train accuracy 0.0000 and test accuracy 0.6879.
Val Loss : 0.6363874590980686, Val Accuracy : 0.6504
Train Loss 0 : 0.6224562525749207
Train Loss 1000 : 0.5711168164199406
Train Loss 2000 : 0.5693496109574273
Train Loss 3000 : 0.5722091558256852
Train Loss 4000 : 0.5733343629025424
Train Loss 5000 : 0.5731782927867819
Val Loss : 0.6228932967090832, Val Accuracy : 0.6754
Model saved at epoch 22 with train accuracy 0.0000 and test accuracy 0.6754.
Val Loss : 0.6547761400823157, Val Accuracy : 0.6348
Train Loss 0 : 0.5078656673431396
Train Loss 1000 : 0.5680117093242489
Train Loss 2000 : 0.5693060713133652
Train Loss 3000 : 0.5703740650238652
Train Loss 4000 : 0.5704633411736049
Train Loss 5000 : 0.5694436980495212
Val Loss : 0.6018283081909547, Val Accuracy : 0.6994
Model saved at epoch 23 with train accuracy 0.0000 and test accuracy 0.6994.
Val Loss : 0.6355038926028904, Val Accuracy : 0.6599
Train Loss 0 : 0.6614511609077454
Train Loss 1000 : 0.5720989594271371
Train Loss 2000 : 0.5670174156231382
Train Loss 3000 : 0.5654647388991735
Train Loss 4000 : 0.567476742898813
Train Loss 5000 : 0.5681329732256111
Val Loss : 0.6039068839769974, Val Accuracy : 0.6938
Model saved at epoch 24 with train accuracy 0.0000 and test accuracy 0.6938.
Val Loss : 0.6506748794096906, Val Accuracy : 0.6512
Train Loss 0 : 0.6762050986289978
Train Loss 1000 : 0.5606675183141863
Train Loss 2000 : 0.5675748484483782
Train Loss 3000 : 0.5679037925582137
Train Loss 4000 : 0.5658898323111402
Train Loss 5000 : 0.5672503124234963
Val Loss : 0.6203294757125124, Val Accuracy : 0.6901
Model saved at epoch 25 with train accuracy 0.0000 and test accuracy 0.6901.
Val Loss : 0.6612410151280498, Val Accuracy : 0.6446
Train Loss 0 : 0.4408676028251648
Train Loss 1000 : 0.5602232888117656
Train Loss 2000 : 0.5612092846888176
Train Loss 3000 : 0.5644612246525363
Train Loss 4000 : 0.5652481670477664
Train Loss 5000 : 0.5668366843135422
Val Loss : 0.618416112992237, Val Accuracy : 0.6895
Model saved at epoch 26 with train accuracy 0.0000 and test accuracy 0.6895.
Val Loss : 0.6655286725898866, Val Accuracy : 0.6406
Train Loss 0 : 0.3611927032470703
Train Loss 1000 : 0.5629126882100558
Train Loss 2000 : 0.5578263364974289
Train Loss 3000 : 0.5621738509858143
Train Loss 4000 : 0.5639709670993275
Train Loss 5000 : 0.5649153163816375
Val Loss : 0.5959279908464153, Val Accuracy : 0.7090
Model saved at epoch 27 with train accuracy 0.0000 and test accuracy 0.7090.
Val Loss : 0.6373105904125678, Val Accuracy : 0.6553
Train Loss 0 : 0.5378040671348572
Train Loss 1000 : 0.5488551568973077
Train Loss 2000 : 0.5541298100049945
Train Loss 3000 : 0.5570301553163081
Train Loss 4000 : 0.557737349145444
Train Loss 5000 : 0.558203508727695
Val Loss : 0.5841739971451211, Val Accuracy : 0.6965
Model saved at epoch 28 with train accuracy 0.0000 and test accuracy 0.6965.
Val Loss : 0.6339029311945464, Val Accuracy : 0.6496
Train Loss 0 : 0.5608359575271606
Train Loss 1000 : 0.5546917866219531
Train Loss 2000 : 0.5591237923164835
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Val Loss : 0.6203294757125124, Val Accuracy : 0.6901
Model saved at epoch 25 with train accuracy 0.0000 and test accuracy 0.6901.
Val Loss : 0.6612410151280498, Val Accuracy : 0.6446
Train Loss 0 : 0.4408676028251648
Train Loss 1000 : 0.5602232888117656
Train Loss 2000 : 0.5612092846888176
Train Loss 3000 : 0.5644612246525363
Train Loss 4000 : 0.5652481670477664
Train Loss 5000 : 0.5668366843135422
Val Loss : 0.618416112992237, Val Accuracy : 0.6895
Model saved at epoch 26 with train accuracy 0.0000 and test accuracy 0.6895.
Val Loss : 0.6655286725898866, Val Accuracy : 0.6406
Train Loss 0 : 0.3611927032470703
Train Loss 1000 : 0.5629126882100558
Train Loss 2000 : 0.5578263364974289
Train Loss 3000 : 0.5621738509858143
Train Loss 4000 : 0.5639709670993275
Train Loss 5000 : 0.5649153163816375
Val Loss : 0.5959279908464153, Val Accuracy : 0.7090
Model saved at epoch 27 with train accuracy 0.0000 and test accuracy 0.7090.
Val Loss : 0.6373105904125678, Val Accuracy : 0.6553
Train Loss 0 : 0.5378040671348572
Train Loss 1000 : 0.5488551568973077
Train Loss 2000 : 0.5541298100049945
Train Loss 3000 : 0.5570301553163081
Train Loss 4000 : 0.557737349145444
Train Loss 5000 : 0.558203508727695
Val Loss : 0.5841739971451211, Val Accuracy : 0.6965
Model saved at epoch 28 with train accuracy 0.0000 and test accuracy 0.6965.
Val Loss : 0.6339029311945464, Val Accuracy : 0.6496
Train Loss 0 : 0.5608359575271606
Train Loss 1000 : 0.5546917866219531
Train Loss 2000 : 0.5591237923164835
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Train Loss 1000 : 0.5629126882100558
Train Loss 2000 : 0.5578263364974289
Train Loss 3000 : 0.5621738509858143
Train Loss 4000 : 0.5639709670993275
Train Loss 5000 : 0.5649153163816375
Val Loss : 0.5959279908464153, Val Accuracy : 0.7090
Model saved at epoch 27 with train accuracy 0.0000 and test accuracy 0.7090.
Val Loss : 0.6373105904125678, Val Accuracy : 0.6553
Train Loss 0 : 0.5378040671348572
Train Loss 1000 : 0.5488551568973077
Train Loss 2000 : 0.5541298100049945
Train Loss 3000 : 0.5570301553163081
Train Loss 4000 : 0.557737349145444
Train Loss 5000 : 0.558203508727695
Val Loss : 0.5841739971451211, Val Accuracy : 0.6965
Model saved at epoch 28 with train accuracy 0.0000 and test accuracy 0.6965.
Val Loss : 0.6339029311945464, Val Accuracy : 0.6496
Train Loss 0 : 0.5608359575271606
Train Loss 1000 : 0.5546917866219531
Train Loss 2000 : 0.5591237923164835
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Train Loss 2000 : 0.5541298100049945
Train Loss 3000 : 0.5570301553163081
Train Loss 4000 : 0.557737349145444
Train Loss 5000 : 0.558203508727695
Val Loss : 0.5841739971451211, Val Accuracy : 0.6965
Model saved at epoch 28 with train accuracy 0.0000 and test accuracy 0.6965.
Val Loss : 0.6339029311945464, Val Accuracy : 0.6496
Train Loss 0 : 0.5608359575271606
Train Loss 1000 : 0.5546917866219531
Train Loss 2000 : 0.5591237923164835
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Val Loss : 0.5841739971451211, Val Accuracy : 0.6965
Model saved at epoch 28 with train accuracy 0.0000 and test accuracy 0.6965.
Val Loss : 0.6339029311945464, Val Accuracy : 0.6496
Train Loss 0 : 0.5608359575271606
Train Loss 1000 : 0.5546917866219531
Train Loss 2000 : 0.5591237923164835
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Model saved at epoch 28 with train accuracy 0.0000 and test accuracy 0.6965.
Val Loss : 0.6339029311945464, Val Accuracy : 0.6496
Train Loss 0 : 0.5608359575271606
Train Loss 1000 : 0.5546917866219531
Train Loss 2000 : 0.5591237923164835
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Val Loss : 0.6339029311945464, Val Accuracy : 0.6496
Train Loss 0 : 0.5608359575271606
Train Loss 1000 : 0.5546917866219531
Train Loss 2000 : 0.5591237923164835
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Train Loss 0 : 0.5608359575271606
Train Loss 1000 : 0.5546917866219531
Train Loss 2000 : 0.5591237923164835
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Train Loss 3000 : 0.5596514424456235
Train Loss 4000 : 0.5590787714636347
Train Loss 5000 : 0.5584750192340816
Val Loss : 0.6028222197402564, Val Accuracy : 0.6919
Model saved at epoch 29 with train accuracy 0.0000 and test accuracy 0.6919.
Train Loss 5000 : 0.5584750192340816
Val Loss : 0.6028222197402564, Val Accuracy : 0.6919
Model saved at epoch 29 with train accuracy 0.0000 and test accuracy 0.6919.
Val Loss : 0.6028222197402564, Val Accuracy : 0.6919
Model saved at epoch 29 with train accuracy 0.0000 and test accuracy 0.6919.
Val Loss : 0.6537527944804368, Val Accuracy : 0.6401



'''