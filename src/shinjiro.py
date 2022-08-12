# gensim 4.0系から使えないコード
# 【A】です。だからこそ、【A】なんです。
#  私の【A】は、全て【A】です。
# 【AはB】です。なので、【AはB】です。
# 【AはB】です。なぜなら、【AはB】だからです。
# 【AはB】です。だからこそ、【AはB】なんです。
# 【Aしたい】ということは、【Aしている】というわけではない。
# TODO: name.txtから読み込んでcsv出力
# TODO: template.txtからテンプレを読み込み
# TODO: {A},{B}にはめ込みtxt出力
# TODO: 上位概念{A}-下位概念{B}を抽出
import gensim
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
pprint(f'match={match}')
