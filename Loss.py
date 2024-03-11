import numpy as np

def loss_catcrossentrop(y_pred,y_true) -> np.float64:
    samp_len = len(y_pred)
    y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
    
    if len(y_true.shape)==1:
        correct_confidence = y_pred_clipped[range(samp_len), y_true]
    elif len(y_true.shape) == 2:
        correct_confidence  = np.sum(y_pred_clipped*y_true,axis=1)
    
    neg_log_poss = -np.log(correct_confidence)
    return neg_log_poss

def loss(output, y) -> np.float64:
    sample_loss = loss_catcrossentrop(output, y)
    data_loss = np.mean(sample_loss)
    return data_loss