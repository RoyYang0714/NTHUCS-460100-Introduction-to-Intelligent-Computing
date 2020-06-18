import csv
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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