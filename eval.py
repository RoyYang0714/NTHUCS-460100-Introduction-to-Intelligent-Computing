import csv
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from tensorflow import keras
from keras import utils as np_utils

csvfile = open('./Team11_Demo.csv')
reader = csv.reader(csvfile)

next(reader)

pred = []

for line in reader:
    pred.append(line[1])

csvfile.close()

csvfile = open('./Program Exam_Answer.csv')
reader = csv.reader(csvfile)

next(reader)

ans = []

for line in reader:
    ans.append(line[1])

csvfile.close()

cm = confusion_matrix(y_true=ans, y_pred=pred)

fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j+1, y=i+1, s=cm[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.show()

count = 0

for i in range(len(pred)):
    if pred[i] == ans[i]:
        count += 1

score = count/len(pred)

print('accurancy: %.2f%s' % (score*100, '%'))

n_classes = 3
y_score = np_utils.to_categorical(pred, num_classes = 3)
y_test = np_utils.to_categorical(ans, num_classes = 3)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i],_ = roc_curve(y_test[:, i],y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"],tpr["micro"],_ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"],tpr["micro"])

plt.rcParams['savefig.dpi'] = 30
plt.rcParams['figure.dpi'] = 300
plt.figure()
# linewidth
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
