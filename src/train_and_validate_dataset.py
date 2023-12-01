import time
import torch
import torch.nn as nn

def train_and_validate_model(model,train_loader,val_loader,epochs=1,max_train_batch=500,max_val_batch=100):

    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    start_time = time.time()

    train_losses = []
    val_losses = []
    train_correct = []
    val_correct = []
    
    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0
        all=0
        model.train()
        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            b+=1
            if b==max_train_batch:
              break
            # X_train, y_train = X_train.to('cuda'), y_train.to('cuda')
            # Apply the model
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            all+=y_pred.size()[0]
            # print(y_pred.size()[0)
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            # print( torch.max(y_pred.data, 1))
            trn_corr += batch_corr
            # print(batch_corr)
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(predicted,trn_corr,torch.max(y_pred.data, 1)[1])
            # Print interim results
            # if b%200 == 0:
                # print(f'epoch: {i:2}  batch: {b:4}  loss: {loss.item():10.8f}  \ accuracy: {trn_corr.item()/(200):7.3f}%')
        train_acc = trn_corr.item() / all
        # print(all/trn_corr.item())
        train_losses.append(loss.item() / max_train_batch)
        # train_correct.append(train_acc)
    
        # train_losses.append(loss)
        train_correct.append(trn_corr)
        model.eval()
        # Run the validation batches
        with torch.no_grad():
            val_loss = 0
            val_all=0
            for b, (X_val, y) in enumerate(val_loader):
                if b==max_val_batch:
                  break
                # X_val, y_val = X_val.to('cuda'), y_val.to('cuda')
                # Apply the model
                y_val = model(X_val)
                val_all+=y_val.size()[0]
                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y).sum()
    
                
                 # Accumulate validation loss
                loss = criterion(y_val, y)
                val_loss += loss.item()
                # if b%600 == 0:
                #     print(f'tes epoch: {i:2}  batch: {b:4} [{10*b:6}/60000]    \ accuracy: {tst_corr.item()*100/(10*b):7.3f}%')
        val_acc = tst_corr.item() / val_all
        val_loss /= max_val_batch
        # print(tst_corr.item())
        scheduler.step(val_loss)
        val_losses.append(val_loss)
        val_correct.append(tst_corr)
       
    
        print(f'Epoch: {i + 1}/{epochs}, Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_acc:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    
        #  # Early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1
        #     if epochs_without_improvement >= patience:
        #         print(f'Early stopping! No improvement in validation loss for {patience} epochs.')
        #         early_stop = True
        #         break
    
        # if not early_stop:
        #     print(f'Training completed after {epochs} epochs.')
    
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
    
    train_corr=[trn_corr.item()/all for trn_corr in train_correct]
    val_corr=[trn_corr.item()/val_all for trn_corr in val_correct]
    train_loss=[trn_loss for trn_loss in train_losses]
    val_loss=[trn_loss for trn_loss in val_losses]

    return [train_corr ,val_corr, train_loss, val_loss]
    