from keras import optimizers
import numpy
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
import os
from tensorflow.python.ops import math_ops

def calculateScore(y_pred, y):
    prediction_pos = y_pred[:,0]<y_pred[:,1]
    y_pos = y[:,0]<y[:,1]
    
    # print prediction_pos
    # print y_pos

    tp = numpy.sum(prediction_pos & y_pos); # y_pred[idx,0]==1 & self.y[:,1]==1
    fp = numpy.sum(prediction_pos & ~y_pos);#y_pred[idx,1]==1&self.y[:,1]==0
    fn = numpy.sum(~prediction_pos & y_pos); #y_pred[idx,1]==0&self.y[:,1]==1
    count = len(y);
    # print (tp, fp, fn)
    precision = tp/(0.0+tp+fp)
    recall = tp/(0.0+tp+fn)
    f1 = 2.0*precision*recall/(precision+recall)
    return {'precision':precision, 'recall':recall, 'f1':f1, 'tp':tp/count, 'fn':fn/count, 'fp':fp/count}

class LRTensorBoard(TensorBoard):
    def __init__(self, trainSet, testSet, log_dir, write_grads, write_images, histogram_freq):  # add other arguments to __init__ if you need
        training_log_dir = os.path.join(log_dir, 'training')
        super().__init__(log_dir=training_log_dir, write_grads=write_grads, write_images = write_images, histogram_freq = histogram_freq)
        self.trainSet = trainSet;
        self.testSet = testSet;

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(LRTensorBoard, self).set_model(model)

    def customMetrics(self, data, logs):
        y_pred = self.model.predict(data['x']);
        score = calculateScore(y_pred, data['y']);
        logs.update({'Score / Precision': score['precision'],
                    'Score / Recall': score['recall'],
                    'Score / F1': score['f1'],
                    'TF / True Positive': score['tp'],
                    'TF / False Positive': score['fp'],
                    'TF / False Negative': score['fn'],})

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        self.customMetrics(self.testSet, val_logs)
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)


        
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        self.customMetrics(self.trainSet, logs);
        lr = self.model.optimizer.getCurrentLeaningRate();
        
        logs.update({'Learning rate': lr})
        super().on_epoch_end(epoch, logs)


class MyRMSpropOptimizer(optimizers.RMSprop):
    def __init__(self, initEpochs, **kwargs):
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            K.set_value(self.iterations, initEpochs);
    def getCurrentLeaningRate(self):
        return K.eval(self.lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
            K.dtype(self.decay)))))
   
