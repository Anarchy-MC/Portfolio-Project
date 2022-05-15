import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
df = pd.read_csv('./movies.csv')

# 空值占比查看
for col in df.columns:
    miss_pre = np.mean(df[col].isnull())
    #print('{} - {}%'.format(col, miss_pre))

#print(df.dtypes)
# name         object
# rating       object
# genre        object
# year          int64
# released     object
# score       float64
# votes       float64
# director     object
# writer       object
# star         object
# country      object
# budget      float64
# gross       float64
# company      object
# runtime     float64

# 正确年份新建
df['year_correct'] = df['released'].astype(str).str.split(' ').str[2]
#print(df.head())

# 排序
df.sort_values(by='gross', ascending=False, inplace=False)
#print(df.head())

# 去重
df['company'].drop_duplicates()

plt.ion()
# 散点图
plt.figure(1)
plt.scatter(x=df['gross'], y=df['budget'])

# 回归曲线拟合
plt.figure(2)
sns.regplot(x='gross', y='budget', data=df, scatter_kws={'color':'red'}, line_kws={'color':'blue'})

# 热力图
plt.figure(3)
correlation = df.corr(method='pearson')
sns.heatmap(correlation, annot=True)
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

df_num = df
plt.figure(4)
for item in df_num.columns:
    if df_num[item].dtype == 'object':
        df_num[item] = df_num[item].astype('category')
        df_num[item] = df_num[item].cat.codes
cor_df_num = df_num.corr()
sns.heatmap(cor_df_num, annot=True)

plt.ioff()
plt.show()

#  correlation pairs
cor_pairs = cor_df_num.unstack()
print(cor_pairs)

# high correlation (sort -> select)
cor_pairs_sort = cor_pairs.sort_values()
high_cor = cor_pairs_sort[(cor_pairs_sort) > 0.5]
print(high_cor)
