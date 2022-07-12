import numpy as np
import pandas as pd
import django
import re

a = np.arange(0, 12).reshape((3, 4))

#取平均数
a_mean = np.mean(a)
a_mean_2 = a.mean()
a_mean_3 = np.average(a)

# 累加和
a_cumsum = np.cumsum(a)

# 临近差，如果为dataframe，第一行显示NAN
a_diff = np.diff(a)

temp = pd.Series([0, 1, 2, 3, 4, 5]).cumsum().shift(1).diff()
temp_cumsum = temp.cumsum()
temp_shift = temp.shift(1)
temp_diff = temp.diff()

# 找出非零元素的位置
a_nonzero = np.nonzero(a)

# 逐行排序, axis=0表示列方向
b = np.array([2, 3, 8, 1, 2, 7, 5, 6]).reshape((2, 4))
b_sort = np.sort(b, axis=1)
print(b_sort)
# 转置
a_t = np.transpose(a)
a_t_2 = a.T

# 截取特定范围的array
a_clip = np.clip(a, 5, 9)
# [[5 5 5 5]
#  [5 5 6 7]
#  [8 9 9 9]]

# 从多维展平为一维
a_flat = a.flatten()
a_flat_2 = a.flat
# <numpy.flatiter object at 0x000001FAC5628180>

# array合并
temp1 = np.array([1, 2, 3])
temp2 = np.array([5, 6, 7])
temp_vstack = np.vstack((temp1, temp2))
temp_hstack = np.hstack((temp1, temp2))
temp_concatenate = np.concatenate((temp1, temp2), axis=0) # axis=0纵向合并
# print(temp_vstack)
# print(temp_hstack)

# 添加新的维度
# print(temp1.shape) #(3,)
# print(temp1.reshape((1, 3)).shape)
# print(temp1[np.newaxis, :].shape) #(1, 3)
print(temp_vstack[:, np.newaxis].shape) #(3, 1)
print(temp_vstack[:, np.newaxis]) # 对于二维array相当于转置

# array分割
a_split = np.split(a, 3, axis=0) #均匀分割
a_split_2 = np.array_split(a, 3, axis=1) #不均匀分割
a_split_3 = np.hsplit(a, 2)
a_split_4 = np.vsplit(a, 3)
print(a_split_3)
print(a_split_4)

# array copy / decopy
b = a.copy()

# Pandas
df = pd.DataFrame(a, columns=['A', 'B', 'C', 'D'])
print(df)
print(df.loc[1, 'A':'C'])
print(df.iloc[1,2:3])
print(df.ix[1, 'A':'C'])

# 添加新的行
df.loc[3] = np.nan
print(df)

# 处理缺失值
df.dropna(axis=0, how='any') #只要存在一个NAN就删除本行，how=‘all’时，本行所有值为NAN才删除
df.fillna(value=0) #用0填充
df.isnull()
for col in df.columns:
    missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, missing))

# print(df.dtypes)
# print(df.columns)
# print(df.index)
# print(df.values)
# print(df.describe())
# print(df.sort_index(axis=1, ascending=False))

arr = np.array(['1.2', '2.', '0.1'])
print(arr.astype(np.float).dtype)

arr = np.array([-1.75, -1.5])
print(arr)
print(np.rint(arr))

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
print(xs)
print(xs.shape, ys.shape, points.shape)

# where的使用
temp = np.random.randn(4, 4)
print(temp)
print(np.where(temp > 0, 1, -1)) # temp中大于0则用1代替，否则用-1代替

# np.in1d / np.intersect1d / np.setxor1d / np.setdiff1d 返回的均为一维
arr1 = np.array([1, 2, 3, 4])
arr2 = np.arange(0, 12).reshape((3, 4))
print(np.in1d(arr2, arr1))
print(np.intersect1d(arr2, arr1)) # 两数组公共元素
print(np.setxor1d(arr2, arr1)) # 两数组的差集
print(np.setdiff1d(arr1, arr2)) # 存在于前者不存在于后者的元素

nsteps = 50
arr = np.random.randint(0, 2, nsteps)
cumsum = np.where(arr > 0, 1, -1).cumsum()
print(cumsum)
check = np.abs(cumsum) > 3
print(check)

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
df = pd.DataFrame(data, columns=['pop', 'year', 's'])
df.name = 'temp'
df.index.name = 'index'
print(df)

# 根据index重新排列,不存在的index使用Nan填充，使用method=‘ffill’实现前向填充，且原来的index必须是有序的，升序或者降序
# reindex也可以指定fill_value
df = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]], index=['a', 'b', 'c'], columns=['11', '22', 'three'])
print(df)
print(df.reindex(['c', 'a', 'b', 'd']))
print(df.reindex(['c', 'a', 'b', 'd'], method='ffill'))

# 删除指定轴上的对象
print(df.drop('11', axis=1))
print(df.drop(['11', '22'], axis=1))

# loc / iloc
print(df.loc['a', ['11', '22']]) # 一行多列
print(df.three)

# 维度不等的两个df相加，只做交集的运算，其他使用Nan代替，可以用add中的fill_value指定特定的代替运算内容
df1 = pd.DataFrame([[1, 2, 3], [2, 3, 5], [3, 6, 5]], index=['a', 'b', 'c'])
df2 = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]], index=['b', 'c', 'd'])
print(df1 + df2)
print(df1.add(df2, fill_value=0))

# 如果某个索引值在DataFrame的列或Series的索引中找不到，则参与运算的两个对象就会被重新索引以形成并集
df3 = pd.DataFrame([[1, 2, 3]], columns=list('abc'))
print(df3)
print(df1 + df3)

# apply的使用
f = lambda x : x.max() - x.min()
print(df1.apply(f))
print(df1.apply(f, axis=1))

# rank的使用
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
print(obj.rank())

# capitalize() 以大写形式返回字符串其中句子的第一个字符为大写其余字符为小写
a = 'alex'
b = a.capitalize()
print(b) # Alex

# ------------------------------Data cleaning------------------------------
# handle Nan
ser = pd.Series([1, 2, 3, np.nan, 4])
print(ser.isnull())

df = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
print(df.dropna()) # 默认删除存在Nan的行, axis=1怎删除列
print(df.dropna(how='all')) # 本行所有值均为Nan则删除

df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan
print(df.dropna())
print(df.dropna(thresh=1)) # 非空元素小于thresh就删除本行/列

print(df.fillna(0)) # 使用0填充Nan
dic = {1:0.5, 2:0.9}
print(df.fillna(dic)) # 指定列使用指定值填充

# 重复值
df = pd.DataFrame({'a':['one', 'two'] * 2 + ['two'], 'b':[1, 2, 3, 4, 4]})
print(df)
print(df.duplicated()) # 整行在之前出现则显示true
print(df.drop_duplicates())
print(df.drop_duplicates(['a'])) # 根据指定列丢弃重复项
print(df.drop_duplicates(['a'], keep='last')) # 默认情况都是保留第一组重复组合，keep则指定保留最后出现的重复组

# 利用函数或映射进行数据转换
df = pd.DataFrame({'food': ['bacon', 'pulled pork', 'Bacon','pastrami','corned beef', 'Bacon', 'pastrami','Honey ham', 'nova lox'],
                   'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_food = {'bacon':'pig', 'pulled pork':'pig', 'bacon':'sheep',
                'pastrami':'cow', 'corned beef':'cow', 'Bacaon':'pig',
                'pastrami':'chicken','honey ham':'chicken', 'nova lox':'fish'}
lowercased = df['food'].str.lower() # 表中可能存在开头为大写字母的food
# df['animal'] = lowercased.map(meat_to_food)
df['animal'] = df['food'].map(lambda x:meat_to_food[x.lower()])
print(df)

# 替换值
df = pd.Series([1., -999., 2., -999., -1000., 3.])
df.replace(-999, np.nan, inplace=True)
print(df)
# df.replace([-999, -1000], np.nan, inplace=True)
# print(df)
df.replace([-999, -1000], [np.nan, 0], inplace=True)
print(df)
df.replace({-999:np.nan, -1000:0})

# 重命名轴索引
df = pd.DataFrame(np.arange(12).reshape(3, 4),
                  index=['Ohio', 'Colorado', 'New York'],
                  columns=['one', 'two', 'three', 'four'])
transform = lambda x:x[:].upper()
df.index = df.index.map(transform)
print(df)

# 离散化和面元（bins）划分，不在范围则显示NAN，没有确切面元范围则自动分配等长范围
bins = [15, 25, 35, 45, 55] # 将年龄划分为16-25， 26-35...左开右闭，可以通过参数right=False设置右端开闭
df = pd.Series([15, 25, 8, 21, 44, 33, 55, 99])
cut = pd.cut(df, bins)
print(cut)
print(pd.value_counts(cut))
cut2 = pd.cut(df, 4, precision=2)
print(cut2) # precision限定小数位数，划分为四个范围
print(pd.value_counts(cut2))
cut3 = pd.qcut(df, 4) # qcut根据样本分位数划分，而不是根据范围平局划分，所以划分结果每个区间样本数都是平均的，不过也可以自定分位数
print(cut3)
print(pd.value_counts(cut3))


# 检查或过滤异常值
df = pd.DataFrame(np.random.randn(1000, 4).reshape(1000, 4))
print(df.describe())
print(df.head())
print(df[(np.abs(df) > 3).any(1)])  # 所有绝对值大于3的

# 排列与随机采样
df = pd.DataFrame(np.arange(5*4).reshape(5, 4))
sampler = np.random.permutation(5) # 随机排序
print(sampler)
print(df.take(sampler)) # 与iloc相似，取对应行
print(df.sample(5))

df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],'data1': range(6)})
print(df)
print(pd.get_dummies(df['key'])) #如果DataFrame的某一列中含有k个不同的值，则可以派生出一个k列矩阵或DataFrame（其值全为1和0）

# 字符串对象方法
val = 'a,b,c,v'
a = [x.strip() for x in val.split(',')]
print(a)
print(':'.join(a))
print('a' in val) # 子串检查的最佳方案
print(val.index(','))
print(val.find(','))
#print(val.index(':')) # index找不到的时候抛出错误，find返回-1
print(val.count(','))
print(val.replace(',', '')) # replace也可用作删除操作
print(val.ljust(15,'+')) # 向右填充
print(val.rjust(15,'+'))

# 正则化
text = "foo    bar\t baz  \tqux"
print(re.split('\s+', text)) # 或者可以先编译，生成可重用的对象
# 如果打算对许多字符串应用同一条正则表达式，强烈建议通过re.compile创建regex对象。这样将可以节省大量的CPU时间。
temp = re.compile('\s+')
temp.split(text)
print(temp.findall(text)) # 找到所有匹配re的模式

text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
# re.IGNORECASE makes the regex case-insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)
print(regex.findall(text)) # 返回所有邮箱地址
print(regex.search(text)) # 返回第一个邮件地址,还有开始结束位置
m = regex.search(text)
print(text[m.start(): m.end()])
print(regex.match(text)) # 只匹配出现在字符串开头的模式
print(regex.sub('666', text)) # 将匹配的模式用另一字串代替，返回新的字串

# 想要分别返回邮箱的用户名，域名，域后缀，用圆括号包起来
pattern_2 = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex_2 = re.compile(pattern_2, flags=re.IGNORECASE)
print(regex_2.findall(text))
# [('dave', 'google', 'com'), ('steve', 'gmail', 'com'), ('rob', 'gmail', 'com'), ('ryan', 'yahoo', 'com')]

# sub还可以用如下表达匹配到各项分组，符号\1对应第一个匹配的组，\2对应第二个匹配的组，以此类推
print(regex_2.sub(r'Username: \1, Domain: \2, Suffix: \3', text))
# Dave Username: dave, Domain: google, Suffix: com
# Steve Username: steve, Domain: gmail, Suffix: com
# Rob Username: rob, Domain: gmail, Suffix: com
# Ryan Username: ryan, Domain: yahoo, Suffix: com

# 层次化索引
df = data = pd.Series(np.random.randn(9),
                      index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                             [1, 2, 3, 1, 3, 1, 2, 2, 3]])
print('-'*20, '层次化索引', '-'*20)
print(df)
print(df['a'])
print(df.loc[:, 2])
print(df.unstack()) # 展开
print(df.unstack().stack()) # unstack的逆运算

frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'],
                              ['Green', 'Red', 'Green']]) # 列名也可以分层
print(frame)
frame.index.names = ['first', 'second']
frame.columns.names = ['state', 'color']
print(frame)

# 重排与分级排序
print('-' * 20, '重排与分级排序', '-' * 20)
print(frame.swaplevel('first', 'second'))
print(frame.sort_index(level=1)) # 按照第二个索引排序

# 根据级别汇总
print('-' * 20, '根据级别汇总', '-' * 20)
print(frame.sum(level = 'first'))
print(frame.sum(level = 'color', axis=1))

# 使用DF的列作为索引
print('-'*20, '使用DF的列作为索引', '-'*20)
df = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two','two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})
print(df)
df1 = df.set_index(['c', 'd'])
print(df1)
print(df.set_index(['c', 'd'], drop=False))
print(df1.reset_index()) # 还原操作

# 合并数据集
print('-'*20, '合并数据集', '-'*20)
df1 = pd.DataFrame({'data1':np.arange(6), 'key':['a', 'c', 'b', 'a', 'r', 'f']})
df2 = pd.DataFrame({'data2':np.arange(3), 'key':['c', 'b', 'e']})
print(pd.merge(df1, df2, on='key')) # 默认为inner join，可以通过how指定
df1 = pd.DataFrame({'data1':np.arange(6), 'lkey':['a', 'c', 'b', 'a', 'r', 'f']})
df2 = pd.DataFrame({'data2':np.arange(3), 'rkey':['c', 'b', 'e']})
print(pd.merge(df1, df2, left_on='lkey', right_on='rkey')) # 列名不同，可以分别指定

df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                    'data1': range(6)})
df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                    'data2': range(5)})
print(pd.merge(df1, df2, how='left', on='key')) # 多对多会生成笛卡尔积，df1中b有三个，df2中b有两个，所以最后输出六个
print(pd.merge(df1, df2))

# 索引维度上的合并
left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
                      'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
print(pd.merge(left1, right1, left_on='key', right_index=True)) # 指定左表使用key作为连接列，右表index列作为连接列

# 轴向连接
arr = np.arange(12).reshape((3, 4))
print(np.concatenate([arr, arr]))
print(np.concatenate([arr, arr], axis=1))  # 横向

# 如果对象在其它轴上的索引不同，我们应该合并这些轴的不同元素还是只使用交集？
# 连接的数据集是否需要在结果对象中可识别？
# 连接轴中保存的数据是否需要保留？许多情况下，DataFrame默认的整数标签最好在连接时删掉。
# concat解决以上问题
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['a', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
print(pd.concat([s1, s2, s3])) # 默认纵向
print(pd.concat([s1, s2, s3], axis=1, sort=False))
print(pd.concat([s1, s2], join='inner', axis=1))
print(pd.concat([s1, s2], axis=1, join_axes=[['a', 'd', 'e']])) # 指定连接的索引
print(pd.concat([s1, s2], keys=['one', 'two'])) # 为详细区分连接的各部分，创建层次化索引， axis=1（水平连接），则keys变为columns

df1 = pd.DataFrame(np.random.randn(2, 2), index=['a', 'b'], columns=['one', 'two'])
df2 = pd.DataFrame(np.random.randn(3, 2), index=['a', 'd', 'c'], columns=['one', 'two'])
print(pd.concat([df1, df2], axis=1, keys=['1', '2'], sort=True))
print(pd.concat({'1':df1, '2':df2}, axis=1)) # 传入字典，则键作为keys

print(pd.concat([df1, df2], ignore_index=True)) # 忽略原有index，用数字代替

# 合并重叠数据
print('-'*20, '合并重叠数据', '-'*20)
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],
              index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64),
              index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
print(np.where(pd.isnull(a), b, a)) # where起到if-else的作用

df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],
                    'b': [np.nan, 2., np.nan, 6.],
                    'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.],
                    'b': [np.nan, 3., 4., 6., 8.]})
print(df1.combine_first(df2)) # combine_first可以看作与where类似，也可看作用传递对象中的数据为调用对象的缺失数据“打补丁”

# 重塑和轴转向
print('-'*20, '重塑和轴转向', '-'*20)
df = pd.DataFrame(np.random.randn(2, 3), index=pd.Index(['Ohio','Colorado'], name='state'),
                  columns=pd.Index(['one', 'two', 'three'], name='number'))
print(df)
print(df.unstack())
print(df.stack())

# 将长格式旋转为宽格式
print('-'*20, '将长格式旋转为宽格式', '-'*20)
df = pd.DataFrame([[1959.0, 1.0, 2710.349, 1707.4, 286.898, 470.045, 1886.9, 28.98, 139.7, 2.82, 5.8, 177.146, 0.00, 0.00],
                   [1959.0, 1.0, 2710.349, 1707.4, 286.898, 470.045, 1886.9, 28.98, 139.7, 2.82, 5.8, 177.146, 0.00, 0.00],
                   [1959.0, 1.0, 2710.349, 1707.4, 286.898, 470.045, 1886.9, 28.98, 139.7, 2.82, 5.8, 177.146, 0.00, 0.00],
                   [1959.0, 1.0, 2710.349, 1707.4, 286.898, 470.045, 1886.9, 28.98, 139.7, 2.82, 5.8, 177.146, 0.00, 0.00],
                   [1959.0, 1.0, 2710.349, 1707.4, 286.898, 470.045, 1886.9, 28.98, 139.7, 2.82, 5.8, 177.146, 0.00, 0.00]],
                  columns=['year', 'quarter', 'realgdp', 'realcons', 'realinv', 'realgovt', 'realdpi', 'cpi',
                           'm1', 'tbilrate', 'unemp', 'pop', 'infl', 'realint'])
print(df)
period = pd.PeriodIndex(year=df.year, quarter=df.quarter, name='date')
print(period)
column = pd.Index(['realgdp', 'infl', 'unemp'], name='item')
print(column)
df = df.reindex(column)

