import jieba

#如果您的電腦同時要使用兩個版本的jieba，請自訂cache檔名，避免兩個cache互相蓋住對方
#jieba.dt.cache_file = 'jieba.cache.new'

seg_list = jieba.cut("新竹的交通大學在新竹的大學路上")
print(" / ".join(seg_list))
# 新竹 / 的 / 交通 / 大學 / 在 / 新竹 / 的 / 大學路 / 上 /