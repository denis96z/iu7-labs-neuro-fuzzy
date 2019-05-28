import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')

mnist_data = mnist.data

mnist_labels = mnist.target

train_X, test_X, train_Y, test_Y = train_test_split(mnist.data, mnist.target, test_size=1/7, random_state=0)

plt.subplot(121)
plt.imshow(train_X[0,:].reshape(28,28), cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:].reshape(28,28), cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))
plt.savefig('./plot.png')

feat_cols = ['pixel'+str(i) for i in range(train_X.shape[1])]

df_mnist = pd.DataFrame(train_X,columns=feat_cols)

df_mnist['label'] = train_Y
print('Size of the dataframe: {}'.format(df_mnist.shape))

pca_mnist = PCA(n_components=2)
principalComponents_mnist = pca_mnist.fit_transform(df_mnist.iloc[:,:-1])

principal_mnist_Df = pd.DataFrame(data = principalComponents_mnist
             , columns = ['principal component 1', 'principal component 2'])
principal_mnist_Df['y'] = train_Y

principal_mnist_Df.head()

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=principal_mnist_Df,
    legend="full",
    alpha=0.3
)
