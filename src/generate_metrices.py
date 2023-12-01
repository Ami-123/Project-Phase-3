import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def generate_accuracy_or_loss_matrix(val_data,train_data,epochs,file_name="",is_loss_matrix=False):
    matrix='Loss' if is_loss_matrix else 'Accuracy'
   
    
    fig, ax = plt.subplots()

    x=np.arange(1,epochs+1)
    # Plot the loss data as a line graph with a label
    ax.plot(x, val_data, label='Validation '+matrix, color='red')
    
    # Plot the accuracy data as a line graph with a label
    ax.plot(x, train_data, label='Training '+matrix, color='blue')
    
    # Add labels and a title
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.set_title(matrix+' Over Epochs')
    
    # Add a legend to differentiate between the lines
    ax.legend()
    
    # Display the plot
    plt.savefig(file_name)
    plt.show()
    




def generate_confusion_matrix(y_true,y_pred,file_name,only_display=False):
  
    
    # constant for classes
    classes = ('angry', 'bored', 'focused', 'neutral')
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix , index = [i for i in classes],
    #                      columns = [i for i in classes])
    # # df_cm = df_cm.applymap(lambda x: '{:.0f}'.format(x))
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted") 
    # plt.ylabel("Actual") 
    # plt.figure(figsize = (12,7))
    
    # ax = sn.heatmap(df_cm, annot=True,fmt="d",square=True, cbar=False)
    # plt.show()
    ConfusionMatrixDisplay(cf_matrix, display_labels=[i for i in classes]).plot()
    if only_display==False:
        plt.savefig(file_name)
    
    