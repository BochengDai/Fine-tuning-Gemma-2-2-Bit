import gc
import os
import re
import numpy as np
import pandas as pd

import nltk
from nltk.util import ngrams
import spacy

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class config:
    root = "/home/PaulDai/Project_milestone/sysllm/sysllm"
    train_path = os.path.join(root, "train_sample.csv")
    test_path = os.path.join(root, "test_sample.csv")
    seed = 42 
    n_splits = 2

train = pd.read_csv(config.train_path)
train = train.iloc[:10000]
test = pd.read_csv(config.test_path)

if test.shape[0] < 10:
    train = train.iloc[:10000]
    
def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)

train["prompt"] = train["prompt"].apply(process)
train["response_a"] = train["response_a"].apply(process)
train["response_b"] = train["response_b"].apply(process)

test["prompt"] = test["prompt"].apply(process)
test["response_a"] = test["response_a"].apply(process)
test["response_b"] = test["response_b"].apply(process)


class Preprocessor:

    def cosine_sim(self, text1: str, text2: str):
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            vectorizer.fit([text1, text2])
            output = vectorizer.transform([text1, text2]).toarray()
            cos_sim = cosine_similarity(output)
            return cos_sim[0][1]
        except:
            return np.nan

    def jaccard_sim(self, text1: str, text2: str):
        set1 = set(text1.split())
        set2 = set(text2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)
    
    def count_new_lines(self, text: str) -> int:
        return text.count('\\n') 
    
    def count_quotes(self, text: str) -> int:
        single_quote_pattern = r"'(.*?)'"
        double_quote_pattern = r'"(.*?)"'
        single_quotes = re.findall(single_quote_pattern, text)
        double_quotes = re.findall(double_quote_pattern, text)
        total_quotes = len(single_quotes) + len(double_quotes)
        return len(single_quotes) + len(double_quotes)

    def tokenize(self, text: str):
        return nltk.word_tokenize(text.lower())

    def generate_ngrams(self, text: str, n: int):
        tokens = self.tokenize(text)
        return list(ngrams(tokens, n))

    def count_ngram_overlaps(self, text1: str, text2: str, n: int) -> int:
        try:
            ngrams1 = self.generate_ngrams(text1, n)
            ngrams2 = self.generate_ngrams(text2, n)
            counter1 = Counter(ngrams1)
            counter2 = Counter(ngrams2)
            overlap = counter1 & counter2
            overlap_count = sum(overlap.values())
            return overlap_count
        except:
            return 0
        
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        
        data["respa_respb_overlap_unigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["response_b"], 1), axis=1)
        data["respa_respb_overlap_bigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["response_b"], 2), axis=1)
        data["respa_respb_overlap_trigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["response_b"], 3), axis=1)

        data["respa_prompt_overlap_unigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["prompt"], 1), axis=1)
        data["respa_prompt_overlap_bigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["prompt"], 2), axis=1)
        data["respa_prompt_overlap_trigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_a"], x["prompt"], 3), axis=1)

        data["respb_prompt_overlap_unigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_b"], x["prompt"], 1), axis=1)
        data["respb_prompt_overlap_bigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_b"], x["prompt"], 2), axis=1)
        data["respb_prompt_overlap_trigram"] = data.apply(lambda x: self.count_ngram_overlaps(x["response_b"], x["prompt"], 3), axis=1)
        
        data["respa_len"] = data["response_a"].apply(lambda x: len(self.tokenize(x)))
        data["respb_len"] = data["response_b"].apply(lambda x: len(self.tokenize(x)))
        data["prompt_len"] = data["prompt"].apply(lambda x: len(self.tokenize(x)))
        
        data["respa_new_lines"] = data["response_a"].apply(lambda x: self.count_new_lines(x))
        data["respb_new_lines"] = data["response_b"].apply(lambda x: self.count_new_lines(x))
        data["prompt_new_lines"] = data["prompt"].apply(lambda x: self.count_new_lines(x))
        
        data["respa_prompt_len_ratio"] = data["respa_len"] / data["prompt_len"]
        data["respb_prompt_len_ratio"] = data["respb_len"] / data["prompt_len"]
        data["respa_respb_len_ratio"] = data["respa_len"] / data["respb_len"]
        
        data["respa_respb_len_diff"] = data["respa_len"] - data["respb_len"]
        data["respa_prompt_len_diff"] = data["respa_len"] - data["prompt_len"]
        data["respb_prompt_len_diff"] = data["respb_len"] - data["prompt_len"]
        
        data["respa_prompt_overlap_unigram_len_ratio"] = data["respa_prompt_overlap_unigram"] / data["prompt_len"]
        data["respa_prompt_overlap_bigram_len_ratio"] = data["respa_prompt_overlap_bigram"] / data["prompt_len"]
        data["respa_prompt_overlap_trigram_len_ratio"] = data["respa_prompt_overlap_trigram"] / data["prompt_len"]

        data["respb_prompt_overlap_unigram_len_ratio"] = data["respb_prompt_overlap_unigram"] / data["prompt_len"]
        data["respb_prompt_overlap_bigram_len_ratio"] = data["respb_prompt_overlap_bigram"] / data["prompt_len"]
        data["respb_prompt_overlap_trigram_len_ratio"] = data["respb_prompt_overlap_trigram"] / data["prompt_len"]
        
        data["overlap_unigram_diff"] = data["respa_prompt_overlap_unigram"] - data["respb_prompt_overlap_unigram"]
        data["overlap_bigram_diff"] = data["respa_prompt_overlap_bigram"] - data["respb_prompt_overlap_bigram"]
        data["overlap_trigram_diff"] = data["respa_prompt_overlap_trigram"] - data["respb_prompt_overlap_trigram"]
        
        data["overlap_unigram_ratio"] = data["respb_prompt_overlap_unigram"] / data["respa_prompt_overlap_unigram"] 
        data["overlap_bigram_ratio"] = data["respb_prompt_overlap_bigram"] / data["respa_prompt_overlap_bigram"] 
        data["overlap_trigram_ratio"] = data["respb_prompt_overlap_trigram"] / data["respa_prompt_overlap_trigram"] 
        
        data["respa_quotes"] = data["response_a"].apply(lambda x: self.count_quotes(x))
        data["respb_quotes"] = data["response_b"].apply(lambda x: self.count_quotes(x))
        data["prompt_quotes"] = data["prompt"].apply(lambda x: self.count_quotes(x))
        
        data["respa_respb_cosine_sim"] = data.apply(lambda x: self.cosine_sim(x["response_a"], x["response_b"]), axis=1)
        data["respa_respb_jaccard_sim"] = data.apply(lambda x: self.jaccard_sim(x["response_a"], x["response_b"]), axis=1)
        
        data["respa_prompt_cosine_sim"] = data.apply(lambda x: self.cosine_sim(x["response_a"], x["prompt"]), axis=1)
        data["respa_prompt_jaccard_sim"] = data.apply(lambda x: self.jaccard_sim(x["response_a"], x["prompt"]), axis=1)
        
        data["respb_prompt_cosine_sim"] = data.apply(lambda x: self.cosine_sim(x["response_b"], x["prompt"]), axis=1)
        data["respb_prompt_jaccard_sim"] = data.apply(lambda x: self.jaccard_sim(x["response_b"], x["prompt"]), axis=1)
        
        data["jaccard_sim_diff"] = data["respa_prompt_jaccard_sim"] - data["respb_prompt_jaccard_sim"]
        data["jaccard_sim_ratio"] = data["respb_prompt_jaccard_sim"] / data["respa_prompt_jaccard_sim"]
        
        return data

print("process data")
preprocessor = Preprocessor()
train = preprocessor.run(train)
test = preprocessor.run(test)

drop_cols = ["id", "response_a", "response_b", "prompt"]
target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
target = "target"

train[target] = np.nan
for idx, t in enumerate(target_cols):
    train.loc[train[t] == 1, target] = idx
train[target] = train[target].astype("int32")
    
X = train.drop(columns=target_cols+drop_cols+[target]+["model_a", "model_b"], axis=1)
y = train[target]
X_test = test.drop(columns=target_cols+drop_cols+["model_a", "model_b"], axis=1)
# X_test = test.drop(columns=drop_cols, axis=1)

X = X.replace([-np.inf, np.inf], np.nan)
X_test = X_test.replace([-np.inf, np.inf], np.nan)

n = X.shape[1]
print(n)
#
X = X.iloc[:, :n//2]
X_test = X_test.iloc[:, :n//2]
print("Done")
cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
test_preds = np.zeros(shape=(X_test.shape[0], y.nunique()))
cv_scores = list()

features = X.columns.tolist()
feat_imp_df = pd.DataFrame({"feature": features})

for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"| Fold {idx+1} |".center(90, "="))
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]

    print(f'train: {X_train.shape}')
    print(f'val: {X_val.shape}')
    print("-"*90)
    print(f"train missing values: {train.isnull().sum().sum()}")
    print(f"test missing values: {test.isnull().sum().sum()}")
    print("-"*90)
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        subsample=0.8,
        n_estimators=650,
        learning_rate=0.045,
        max_depth=5,
        tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
        gpu_id=0 if torch.cuda.is_available() else -1,
        random_state=config.seed
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10
    )
    
    val_preds = model.predict_proba(X_val)
    val_log_loss = log_loss(y_val, val_preds)

    print(f"val log loss: {val_log_loss:.5f}")
    cv_scores.append(val_log_loss)
    
    test_preds += model.predict_proba(X_test) / cv.get_n_splits()
    
    feat_imp_df = feat_imp_df.merge(
        pd.DataFrame(
            {
                "feature": features,
                f"fold_{idx+1}_feat_imp": model.feature_importances_,
            }
        ),
        on=["feature"],
        how="left",
    )

print("="*90)
print(f"CV: {np.mean(cv_scores):.5f}")

feat_imp_df["avg_importance"] = feat_imp_df.iloc[:, 1:].mean(axis=1)
plt.figure(figsize=(12, 10))
sns.barplot(
    data=feat_imp_df.sort_values(by="avg_importance", ascending=False).iloc[
        :50
    ],
    x="avg_importance",
    y="feature",
    color="royalblue",
    width=0.75,
)
plt.title("Average Feature Importances of All Folds", size=12)
plt.show()

for idx, t in enumerate(target_cols):
    test[t] = test_preds[:, idx]
test.head()

test.to_csv("submission_xgb.csv", index=False)
