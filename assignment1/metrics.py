def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    
    for i in range(len(prediction)):
        if prediction[i] == True and ground_truth[i] == True:
            precision += 1
            recall += 1
            accuracy += 1
        elif prediction[i] == True and ground_truth[i] == False:
            precision += 1
        elif prediction[i] == False and ground_truth[i] == True:
            recall += 1
        else:
            accuracy += 1
        
    precision = precision / len(prediction)
    recall = recall / len(prediction)
    accuracy = accuracy / len(prediction)
    f1 = 2 * (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            accuracy += 1
    accuracy = accuracy / len(prediction)
    return accuracy
