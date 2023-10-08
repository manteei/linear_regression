from matplotlib import pyplot as plt


def visCount(plt, df):
    plt.figure(figsize=(10, 15))
    df.count().plot(kind='bar', title='Количество',color='skyblue' )
    plt.show()


def addTexit(dfMeans, k):
    for i, value in enumerate(dfMeans):
        x = value + 100
        if i == 0:
            x = value / k
        plt.text(x, i, f'{value:.2f}', fontsize=10)


def visMean(plt, df):
    plt.figure(figsize=(15, 10))
    plt.title('Средние значения признаков')
    dfMeans = df.mean().sort_values(ascending=False)
    plt.barh(dfMeans.index, dfMeans.values, color='skyblue')
    plt.xlim(dfMeans.min(), 4000)
    addTexit(dfMeans, 65)
    plt.show()


def visAll(df):
    df.hist(bins=120, figsize=(10, 10), color='skyblue')
    plt.show()


def visStd(plt, df):
    plt.figure(figsize=(15, 10))
    plt.title('Стандартные отклонения признаков')
    dfStd = df.std().sort_values(ascending=False)
    plt.barh(dfStd.index, dfStd.values,color='skyblue')
    plt.xlim(dfStd.min() * 1.1, 4000)
    addTexit(dfStd, 65)
    plt.show()


def visMin(plt, df):
    plt.figure(figsize=(15, 10))
    plt.title('Минимум')
    dfMin = df.min().sort_values(ascending=False)
    plt.barh(dfMin.index, dfMin.values, color='skyblue')
    plt.xlim(dfMin.min() * 1.1, 200)
    addTexit(dfMin, 100)
    plt.show()

def visMax(plt, df):
    plt.figure(figsize=(15, 10))
    plt.title('максимум')
    dfMax = df.max().sort_values(ascending=False)
    plt.barh(dfMax.index, dfMax.values, color='skyblue')
    plt.xlim(dfMax.min() * 1.1, 40000)
    addTexit(dfMax, 15)
    plt.show()