from utils.Voc import *
import os

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data/", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

# 加载/组装voc和对
save_dir = os.path.join("dataset", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# 打印一些对进行验证
print("\npairs:")
for pair in pairs[:10]:
    print(pair)