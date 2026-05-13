import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay,classification_report,roc_curve,auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf


#dataset location paths
train_data_path = "LC25000/train"
test_data_path = "LC25000/test"
val_data_path = "LC25000/val"

def get_train_test_val():
    #load train dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_data_path)
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    #load test dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_data_path,shuffle=False)
    test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)
    #load val dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(val_data_path)
    val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
    return train_dataset,test_dataset,val_dataset

def get_dataset_labels(dataset):
    labels=[]
    #add labels for the images to the list as numpy array
    for image,label in dataset:
        labels.append(label.numpy())
    #concatenate numpy arrays
    y_labels=np.concatenate(labels,axis=0)
    return y_labels

def create_model():
    #define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(256, 256, 3)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    #compile model
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #print model summary
    model.summary()
    return model

def train_model(model,train_dataset,val_dataset,callbacks=None):
    #store model training history
    history = model.fit(train_dataset, validation_data=val_dataset,epochs=10,callbacks=callbacks)
    #plot training history
    plot_model_history(history)

def evaluate_model(model,test_dataset):
    #get the loss and accuracy for test dataset
    test_loss,test_acc = model.evaluate(test_dataset)
    print("test loss:",test_loss)
    print("test accuracy:",test_acc)

def get_model_predictions(model,test_dataset):
    #get model predictions
    predictions = model.predict(test_dataset)
    return predictions

def save_model(model):
    model.save('model')

def plot_model_history(history):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # plot accuracy
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['train', 'val'])
    # plot loss
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xticks(np.arange(len(history.history['accuracy'])))
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['train', 'val'])
    plt.show()

def plot_roc_curve(y_test,y_pred):
    # one hot encode test labels
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    num_classes = len(np.unique(y_test))
    #create dictionaries for fpr,tpr and roc_auc
    fpr,tpr,roc_auc = dict(),dict(),dict()
    #calculate fpr,tpr roc_auc for each class and store in dictionary
    for i in range(num_classes):
        fpr[i],tpr[i],_ = roc_curve(y_test_bin[:,i],y_pred[:,i],drop_intermediate=False)
        roc_auc[i] = auc(fpr[i],tpr[i])

    #colors and classes to be plotted
    colors = ["aqua", "darkorange", "cornflowerblue"]
    classes =['lung_aca','lung_n','lung_scc']
    #plot roc curve for each class
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color,label=f"{classes[i]} ROC curve (area = {roc_auc[i]:0.2f})")
    #plot random chance line
    plt.plot([0, 1], [0, 1], '--', color='black')

    plt.title('Lung Classification ROC Curve OvR')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_test,y_pred):
    # find the max value for class prediction
    y_pred=np.argmax(y_pred,axis=-1)
    # create confusion matrix with predictions
    cm_display=ConfusionMatrixDisplay.from_predictions(y_test,y_pred,display_labels=['lung_aca','lung_n','lung_scc'],cmap=plt.cm.Blues,)
    # set title
    cm_display.ax_.set_title('Lung Classification\nConfusion Matrix')
    plt.show()


def print_classification_report(y_test,y_pred):
    # find the max value for class prediction
    y_pred = np.argmax(y_pred, axis=-1)
    #get classification report
    report = classification_report(y_test,y_pred,digits=4,target_names=['lung_aca','lung_n','lung_scc'])
    print(f'Classification Report\n{report}')


def main():
    #get train, test and val datasets
    train_dataset,test_dataset,val_dataset = get_train_test_val()
    #create model
    model = create_model()
    #define training callbacks
    callbacks =[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True,verbose=1,monitor='val_loss',mode='min')]
    #train model
    train_model(model,train_dataset,val_dataset,callbacks)
    #save model
    save_model(model)
    #evaluate model accuracy and loss on test dataset
    evaluate_model(model,test_dataset)
    #get the labels for the test dataset
    y_test = get_dataset_labels(test_dataset)
    #get the model predictions for test dataset
    y_pred = get_model_predictions(model,test_dataset)
    #plot roc auc
    plot_roc_curve(y_test,y_pred)
    #plot classification report
    plot_confusion_matrix(y_test,y_pred)
    #get classification report
    print_classification_report(y_test,y_pred)


if __name__ == '__main__':
    main()