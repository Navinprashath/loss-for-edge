from keras import backend as K

def negative_max_soft_log_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    #Clip the value to avoid zero division
    y_pred_f = K.clip(y_pred_f, K.epsilon(), 1 - K.epsilon())

    #Get the true label index
    true_index = K.argmax(y_true_f)

    #Convert to tf.int64 to tf.int32 
    true_index_32 = tf.to_int32(true_index)

    #Get the probability of the true index
    pred_value = y_pred_f[true_index_32]

    """Imposing that all other probabilities should 
    be less than 50 percent of actual probability"""
    subfactor = y_pred_f[true_index_32]/2
    pred_value_shifted = (pred_value - subfactor)

    #Find the relative difference between the probabilities and rest of the probabilities
    y_pred_wrong = tf.subtract(y_pred_f,pred_value_shifted)
    y_pred_wrong_f = K.clip(y_pred_wrong, K.epsilon(), 1 - K.epsilon())

    #Mask the index values that are greater than the true label probability
    y_wrong_preds = tf.to_float(tf.greater(y_pred_f,pred_value_shifted))
 
    #Singlemax probability
    elemid = tf.to_int32(K.argmax(y_pred_wrong_f))
    logdiff = (1 - y_true_f[elemid]) * K.log(1-y_pred_wrong_f[elemid]) * y_wrong_preds[elemid]

    #Sum of all log differences
    error = -(K.sum(logdiff))
    return error