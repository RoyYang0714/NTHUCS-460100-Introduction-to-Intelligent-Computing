import numpy as np
from tensorflow import keras

class ADABoost:
    def __init__(self, x_train, y_train, x_test, model_list):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.model_list = model_list
        self.weight_list = np.repeat(float(1/len(self.x_train)), len(self.x_train))
        self.hypothesis_list = []
        self.hypothesis_weight_list = []

    def VotingClassifier_Tensorflow(self, classifier_list) :
        
        idx = 0

        for classifier in classifier_list:
            if idx == 0:
                y_pred = classifier.predict(self.x_test) * self.hypothesis_weight_list[idx]
            else:
                y_pred += classifier.predict(self.x_test) * self.hypothesis_weight_list[idx]
            
            idx += 1

        return y_pred
    
    def adaboost(self):
        for model in self.model_list:
            
            print('Loading {}'.format(model))
            new_model = keras.models.load_model('./checkpoints/{}.h5'.format(model))
            
            self.hypothesis_list.append(new_model)

            y_pred = new_model.predict(self.x_train)

            error = 0
            error_count = 0
            total_error = 0

            for j in range(len(y_pred)) :
                total_error += self.weight_list[j]

                if np.argmax(y_pred[j]) != np.argmax(self.y_train[j]) :
                    error += self.weight_list[j]
                    error_count += 1

            print("Total error is " + str(total_error) + ", current error is " + str(error)) 

            error /= total_error
            error_count /= len(y_pred)
            new_hypothesis_weight = np.log((1 - error) / error)

            print("error of the decision tree is : " + str(error_count))
            print("New hypothesis weight is " + str(new_hypothesis_weight))

            # update training data weight
            for j in range(len(self.weight_list)) :
                if np.argmax(y_pred[j]) != np.argmax(self.y_train[j]) :
                    self.weight_list[j] *= np.exp(new_hypothesis_weight)

            self.hypothesis_weight_list.append(new_hypothesis_weight)

            del new_model

        for weight, model in zip(self.hypothesis_weight_list, self.model_list):
            print("{} hypothesis weight is %.2f".format(model) %weight)

        y_pred = self.VotingClassifier_Tensorflow(self.hypothesis_list)

        return y_pred 
