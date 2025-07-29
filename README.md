# Project_NLP

# 📰 Clickbait News Classifier (클릭베이트<낚시성> 뉴스 분류기)

자연어 처리 기반으로 뉴스 기사의 본문을 분석하여 **클릭베이트 여부**를 분류하는 AI 모델입니다.  
한국어 뉴스 데이터를 정제하고, LSTM 모델을 활용해 이진 분류를 수행합니다.

---

## 📌 프로젝트 개요

- 문제정의: 과도하게 주목을 끌기 위한 ‘낚시성 제목’(clickbait) 뉴스는 독자의 피로도를 높이며, 정보 왜곡의 위험이 있습니다.
- 목표: 뉴스 콘텐츠에서 clickbait 여부를 자동으로 판별하는 모델 개발
- 접근법: 불용어 제거 후 토큰화 → 단어 사전 구축 → 임베딩 → LSTM 분류기 학습

---

## ⚙️ 사용 기술

| 분야 | 사용한 기술 |
|------|--------------|
| 언어/도구 | Python, Pandas, NumPy, PyTorch |
| 모델링 | LSTM, BiLSTM, BCEWithLogitsLoss |
| 기타 | CUDA(GPU), JSON, Tokenizer, Custom Padding |

---

## 🧰 주요 기능

- 뉴스 본문 텍스트 전처리 및 불용어 제거
- 사용자 정의 Vocab 생성 및 ID 매핑 (token_to_id.json)
- 문장 단위 padding 구현 (`pad_sequences`)
- LSTM 기반 분류 모델 학습 및 평가
- 에포크별 모델 저장 및 정확도 로깅

---

## 📁 프로젝트 구조

```bash
Project_NLP/
├── data/                   # 뉴스 데이터 CSV
├── vocab/                  # 단어 사전 및 매핑 파일
├── model/                  # LSTM 모델 정의 및 저장
├── notebook/               # 실험용 Jupyter 노트북
├── sample.py               # 전체 실행 파이썬 스크립트
└── README.md               
