# gensim 4.0系から使えないコード
# 【A】です。だからこそ、【A】なんです。
#  私の【A】は、全て【A】です。
# 【AはB】です。なので、【AはB】です。
# 【AはB】です。なぜなら、【AはB】だからです。
# 【AはB】です。だからこそ、【AはB】なんです。
# 【Aしたい】ということは、【Aしている】というわけではない。
# TODO: similar.txtを読み込む。毎回modelをloadしない。
# TODO: 上位概念{A}-下位概念{B}を抽出
import datetime
import random
import gensim
import pandas as pd
from pprint import pprint

print("started")
# 学習済モデルのパス
model_path = '../model/cc.ja.300.vec.gz'
# ロードに5分くらいかかる
print("loading model")
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
print("finished")
# 登録している単語の数
# print(len(model.vocab.keys())) # 2000000

now_time = datetime.datetime.now()

# ひとつの単語ベクトルの次元
s = model['猫'].shape # (300,)
pprint(f's={s}')

# 類似度上位10件を取得
pprint(model.most_similar('猫', topn=10))
pprint(model.most_similar('バナナ', topn=10))
pprint(model.most_similar('修行', topn=10))
pprint(model.most_similar('時計', topn=10))
pprint(model.most_similar('時間', topn=10))
match = model.most_similar("企業", topn=10)
match2 = model.most_similar("起業", topn=10)
match3 = model.most_similar("挨拶", topn=10)
pprint(f'match={match}')


def load_noun() -> list:
    f = open('noun.txt', 'r', encoding='UTF-8')
    noun_list = f.readlines()
    f.close()
    return noun_list


def calc_similar(word_list: list) -> list:
    similar_list = []
    for word in word_list:
        similar_list.append(model.most_similar(word.strip(), topn=10))
    df = pd.DataFrame(similar_list)
    df.to_csv(f"similar_{now_time}.csv", encoding="utf-8")

    return similar_list


def gen_shinjiro(noun_list: list, similar_list: list) -> list:
    f = open('shinjiro_template.txt', 'r', encoding='UTF-8')
    template_list = f.readlines()
    f.close()
    shinjiro = []
    for i in range(3):
        for template in template_list:
            rndint = random.randint(0, len(noun_list)-1)
            template = template.replace('{A}', noun_list[rndint].strip())  # 元のクエリ
            template = template.replace('{B}', similar_list[rndint][0][0])  # 最高類似度の単語
            shinjiro.append(template.strip())
        pprint(shinjiro)
    return shinjiro


def main():
    noun_list: list = load_noun()
    similar_list: list = calc_similar(noun_list)
    shinjiro: list = gen_shinjiro(noun_list, similar_list)
    df = pd.DataFrame(shinjiro)
    df.to_csv(f"shinjiro_{now_time}.csv", encoding="utf-8")


main()
