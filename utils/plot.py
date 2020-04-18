from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['figure.figsize'] = (13.66,6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

FIGURES_DIR = 'figures/'

def plot_optimal_k(k_range, scores, opt_k, opt_score):
    fig, ax = plt.subplots()
    ax.plot(k_range, scores)
    ax.plot(opt_k, opt_score, 'ro', color='green', markersize=10)
    ax.annotate('   optimal k', (opt_k, opt_score))
    ax.set(xlabel='Possible K neighbour', ylabel='Accuracy',
           title=f'Optimal K: {opt_k} Optimal Score: {opt_score}')
    plt.savefig(FIGURES_DIR + 'Figure_opt_k' + '.png')
    plt.show()


def plot_cm(cm,meth:str=None):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.savefig(FIGURES_DIR + f'Figure_cm_{meth}_' + '.png')
    plt.show()

def plot_data(features, labels):
    x, y, z, p = features['income'], features['age'], features['loan'], labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x, y, z, c=p)
    ax.set_xlabel('Income', fontsize=14)
    ax.set_ylabel('Age', fontsize=14)
    ax.set_zlabel('Loan', fontsize=14)
    fig.colorbar(img)
    plt.savefig(FIGURES_DIR + f'Figure_data' + '.png')
    plt.show()
