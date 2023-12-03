import time
import torch
import torch.nn as nn

def train_and_validate_model(model, train_loader, val_loader, fold_number, epochs, max_train_batch=500, max_val_batch=100, patience=3):

    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    start_time = time.time()

    train_losses = []
    val_losses = []
    train_correct = []
    val_correct = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        all = 0
        model.train()

        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1
            if b == max_train_batch:
                break

            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            all += y_pred.size()[0]

            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr.item()  # Fix: use item() to get the scalar value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = trn_corr / all
        train_losses.append(loss.item() / max_train_batch)
        train_correct.append(train_acc)

        model.eval()
        val_loss = 0
        val_all = 0

        with torch.no_grad():
            for b, (X_val, y) in enumerate(val_loader):
                if b == max_val_batch:
                    break

                y_val = model(X_val)
                val_all += y_val.size()[0]

                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y).sum()

                loss = criterion(y_val, y)
                val_loss += loss.item()

        val_acc = tst_corr.item() / val_all
        val_loss /= max_val_batch

        scheduler.step(val_loss)
        val_losses.append(val_loss)
        val_correct.append(val_acc)

        print(f'Epoch: {i + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_acc:.4f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save the best model weights based on the fold number
            save_path = f"saved_model/best_model_fold_{fold_number}.pth"
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping! No improvement in validation loss for {patience} epochs.')
                break

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    return [train_correct, val_correct, train_losses, val_losses]
