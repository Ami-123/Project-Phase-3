from os import system
import torch
import models
import load_dataset
import train_and_validate_dataset
import generate_metrices
import test_dataset

import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


class Phase2:
    EPOCHS = 2
    BATCH_SIZE = 10
    device = None
    LR = 0.001
    model = None
    saved_model = None
    dataset_path = ""
    train_loader = None
    val_loader = None
    test_loader = None
    dataset = None

    def __init__(self, dir_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_path = dir_path
        self.main()

    def display_menu(self):
        print("1. Train, validate and test model")
        print("2. Test saved model on entire dataset")
        print("3. Test saved model on a single image")
        print("4. Display Model")
        print("5. Exit", end="\n\n")

    def select_model(self):
        print("1. Model 1 with Dropout")
        print("2. Variant 1")
        print("3. Variant 2")

    def main(self):
        while True:
            self.display_menu()
            choice = input("Select : ").strip()

            if choice == '1':
                if self.dataset is None:
                    self.dataset = load_dataset.load_dataset_and_process(self.dataset_path)
                    print("Dataset loaded")
                else:
                    print("Dataset already loaded")

                self.select_model()
                if self.model is not None:
                    models.reset_model_from_GPU()
                    self.model = None

                model_choice = input("Select : ").strip()
                if model_choice == '1':
                    self.model = models.ConvNetWithDropout().to(self.device)
                elif model_choice == '2':
                    self.model = models.ConvNet().to(self.device)
                elif model_choice == '3':
                    self.model = models.ConvNet2().to(self.device)
                else:
                    print("Wrong choice!!")
                    continue

                k_folds = 5
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):

                    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
                    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
                    
                    train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.BATCH_SIZE, sampler=train_sampler)
                    train_loader = [(X.to(self.device), y.to(self.device)) for X, y in train_loader]

                    val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.BATCH_SIZE, sampler=val_sampler)
                    val_loader = [(X.to(self.device), y.to(self.device)) for X, y in val_loader]

                    print(f"\nTraining Fold {fold + 1}/{k_folds}...")
                    model_name = f"best_model_fold_{fold}"
                    [train_corr1, val_corr1, train_loss1, val_loss1] = train_and_validate_dataset.train_and_validate_model(
                        self.model, train_loader, val_loader,fold,self.EPOCHS)
                    
                    print(val_corr1, train_corr1,)
                    generate_metrices.generate_accuracy_or_loss_matrix(val_corr1, train_corr1,self.EPOCHS,
                                                                        model_name + "_accu.png")
                    generate_metrices.generate_accuracy_or_loss_matrix(val_loss1, train_loss1,self.EPOCHS, model_name + "_loss.png",
                                                                   True) 
                    [y_true1, y_pred1] = test_dataset.get_test_results(self.model, val_loader)
                    generate_metrices.generate_confusion_matrix(y_true1, y_pred1, model_name + "_cf.png")
                    print(
                        classification_report(y_true1, y_pred1, target_names=['angry', 'bored', 'focused', 'neutral']))
                    print("\n\nMicro - Precision, Recall, F1 Score \n")
                    print(precision_recall_fscore_support(y_true1, y_pred1, average='micro'))
                    
                    print("\nMacro - Precision, Recall, F1 Score \n")
                    print(precision_recall_fscore_support(y_true1, y_pred1, average='macro'))
                    
                    print("\nAccuracy")
                    print(accuracy_score(y_true1, y_pred1))
                print("\nK-Fold Cross-Validation Completed.")

            elif choice == '2':
                input_model = input("Enter model relative path: (Ex. models/model3.pth) ").strip()
                self.saved_model = load_dataset.load_saved_model(self.device, input_model)
                print("Model loaded")

                self.data_loader = load_dataset.load_entire_dataset(self.device, self.dataset_path, self.BATCH_SIZE)
                print("Dataset loaded")
                [y_true, y_pred] = test_dataset.get_test_results(self.saved_model, self.data_loader)
                generate_metrices.generate_confusion_matrix(y_true, y_pred, "", True)
                print(classification_report(y_true, y_pred, target_names=['angry', 'bored', 'focused', 'neutral']))
                print("\n\nMicro - Precision, Recall, F1 Score \n")
                print(precision_recall_fscore_support(y_true, y_pred, average='micro'))

                print("\nMacro - Precision, Recall, F1 Score \n")
                print(precision_recall_fscore_support(y_true, y_pred, average='macro'))

                print("\nAccuracy")
                print(accuracy_score(y_true, y_pred))

            elif choice == '3':
                input_model = input("Enter model relative path: (Ex. models/model3.pth) ").strip()
                self.saved_model = load_dataset.load_saved_model(self.device, input_model)

                input_image = input(
                    "Enter image relative path: (Ex. processed_data/neutral/PrivateTest_1960069.jpg) ").strip()
                test_dataset.predict_emotion(self.device, input_image, self.saved_model)

            elif choice == '4':
                self.select_model()
                if self.model is not None:
                    models.reset_model_from_GPU()
                    self.model = None

                model_choice = input("Select : ").strip()
                if model_choice == '1':
                    print("\n", models.ConvNetWithDropout())
                    models.print_model_parameters(models.ConvNetWithDropout())
                elif model_choice == '2':
                    print("\n", models.ConvNet())
                    models.print_model_parameters(models.ConvNet())
                elif model_choice == '3':
                    print("\n", models.ConvNet2())
                    models.print_model_parameters(models.ConvNet2())
                else:
                    print("Wrong choice!!")
                    continue

            elif choice == '5':
                system('cls||clear')
                return

            else:
                print("Wrong choice!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion Recognition Dataset Training and Testing')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset directory', required=True)
    Phase2(parser.parse_args().dataset_path)
