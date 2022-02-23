model.eval()
start_time = time.time()
array1 = []
running_TP = 0
running_FP = 0
running_FN = 0
running_TN = 0

with torch.no_grad():
    running_loss = 0.
    running_corrects = 0


    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        array1.append([preds,labels.data])

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if preds == labels.data == b :
            running_TP += 1
        if preds == b and labels.data == a :
            running_FP += 1
        if preds == a and labels.data == b :
            running_FN += 1
        if preds == labels.data == a :
            running_TN += 1

    recall = running_TP / (running_TP + running_FN)
    precision = running_TP / (running_TP + running_FP)
    f1_score = 2*recall*precision / (precision + recall)
    print('recall:{} precision:{} f1_score{}'.format(recall, precision, f1_score))
    epoch_loss = running_loss / len(test_datasets)
    epoch_acc = running_corrects / len(test_datasets) * 100.
    print('[Val Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc, time.time() - start_time))