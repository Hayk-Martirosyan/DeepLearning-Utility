from keras import optimizers
import numpy
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
import os
from tensorflow.python.ops import math_ops
from keras.metrics import top_k_categorical_accuracy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import io

def calculateScore(y_pred, y):
    prediction_pos = y_pred[:,0]<y_pred[:,1]
    y_pos = y[:,0]<y[:,1]
    
    # print prediction_pos
    # print y_pos

    tp = numpy.sum(prediction_pos & y_pos); # y_pred[idx,0]==1 & self.y[:,1]==1
    tn = numpy.sum(~prediction_pos & ~y_pos); # y_pred[idx,0]==0 & self.y[:,1]==0
    fp = numpy.sum(prediction_pos & ~y_pos);#y_pred[idx,1]==1&self.y[:,1]==0
    fn = numpy.sum(~prediction_pos & y_pos); #y_pred[idx,1]==0&self.y[:,1]==1
    count = len(y);
    # print (tp, fp, fn)
    precision = tp/(0.0+tp+fp)
    recall = tp/(0.0+tp+fn)
    f1 = 2.0*precision*recall/(precision+recall)
    return {'precision':precision, 'recall':recall, 'f1':f1, 
            'tp':tp/(tp+fn), 'tn':tn/(tn+fp), 'fn':fn/count, 'fp':fp/count}

class LRTensorBoard(TensorBoard):
    def __init__(self, trainSet, testSet, log_dir, write_grads, write_images, histogram_freq):  # add other arguments to __init__ if you need
        training_log_dir = os.path.join(log_dir, 'training')
        super(LRTensorBoard, self).__init__(log_dir=training_log_dir, write_grads=write_grads, write_images = write_images, histogram_freq = histogram_freq)
        self.trainSet = trainSet;
        self.testSet = testSet;
        self.frequency = histogram_freq;
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(LRTensorBoard, self).set_model(model)

    def customMetrics(self, data, logs):
        y_pred = self.model.predict(data['x']);
        score = calculateScore(y_pred, data['y']);
        logs.update({'1. Score / Precision': score['precision'],
                    '1. Score / Recall': score['recall'],
                    '1. Score / F1': score['f1'],
                    '2. True / Positive': score['tp'],
                    '2. True / Negative': score['tn'],
                    '3. False / Positive': score['fp'],
                    '3. False / Negative': score['fn'],})

    def on_epoch_end(self, epoch, logs=None):
        if not (self.frequency==0 or epoch % self.frequency==0):
            return
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        self.customMetrics(self.testSet, val_logs)
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)


        
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        self.customMetrics(self.trainSet, logs);

        
        lr = self.model.optimizer.getCurrentLearningRate();
        logs.update({'4. Learning rate': lr})
        super(LRTensorBoard, self).on_epoch_end(epoch, logs)


class MultiClassTensorBoard(LRTensorBoard):

    def calculateScores(self, confusionMatrix):
        scores = []

        for i in range(len(confusionMatrix)):
            score = {};
            tp = confusionMatrix[i,i]+0.0
            
            allLabelPredictions = numpy.sum(confusionMatrix[:,i])
            if allLabelPredictions==0:
                precision=0.0
            else:
                precision = tp / allLabelPredictions

            realLabelCount = numpy.sum(confusionMatrix[i,:], 0)
            if realLabelCount==0:
                recall = 0.0
            else:  
                recall = tp / realLabelCount
            if precision+recall==0:
                f1 = 0.0;
            else:
                f1 = 2.0 * precision*recall / (precision+recall)
            score =  {'precision':precision, 'recall':recall, 'f1':f1, 'label':i}
            scores.append(score)
        return scores

    def customMetrics(self, data, logs):
        if(len(data['x'])==0):
            return

        y_pred = self.model.predict(data['x']);
        labelCount = data['y'].shape[1]
        # print(numpy.argmax(y_pred[0:1,:],1))
        # print(data['y'][0:1,:])

        confusionMatrix = confusion_matrix(numpy.argmax(data['y'], axis=1), numpy.argmax(y_pred, axis=1), range(0, labelCount));
        # print(confusionMatrix)
        scores = self.calculateScores(confusionMatrix)
        # scores.reverse()
        for score in scores:
            # print (score)
            logs.update({'1. Score / Precision_' +str(score['label']): score['precision'],
                        '1. Score / Recall_' +str(score['label']): score['recall'],
                        '1. Score / F1_' +str(score['label']): score['f1'],})

class RegressionTensorBoard(TensorBoard):
    def __init__(self, log_dir, write_grads, write_images, histogram_freq):  # add other arguments to __init__ if you need
        training_log_dir = os.path.join(log_dir, 'training')
        super(RegressionTensorBoard, self).__init__(log_dir=training_log_dir, write_grads=write_grads, write_images = write_images, histogram_freq = histogram_freq)
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(RegressionTensorBoard, self).set_model(model)

   

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)

       
        
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        if hasattr(self.model.optimizer, 'getCurrentLearningRate'):
            lr = self.model.optimizer.getCurrentLearningRate();
            logs.update({'4. Learning rate': lr})

        super(RegressionTensorBoard, self).on_epoch_end(epoch, logs)

     

class DataGraphTensorBoard(Callback):
    def __init__(self, log_dir, trainSet, byIndex = False, frequency=0):  # add other arguments to __init__ if you need
        
        super(DataGraphTensorBoard, self).__init__()
        self.data_log_dir = os.path.join(log_dir, 'images')
        self.trainSet = trainSet;
        self.frequency = frequency
        self.byIndex = byIndex;

    def set_model(self, model):
        # Setup writer for validation metrics
        self.writer = tf.summary.FileWriter(self.data_log_dir)
        super(DataGraphTensorBoard, self).set_model(model)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.frequency==0 or epoch % self.frequency==0:
            img = self.renderGraph( self.trainSet);
            self.writeGraph(img, self.writer)

    
    def renderGraph(self, data):
        x = data['x']
        y = data['y']

        y_pred = self.model.predict(x);
        # x = x[:,0]
        plt.figure(figsize=(10,5))
        
        if self.byIndex:
            minx = 0
            maxx = len(x)
        else:
            minx = numpy.min(x)
            maxx = numpy.max(x)
        
        miny = min(numpy.min(y), numpy.min(y_pred))
        maxy = max(numpy.max(y), numpy.max(y_pred))
        plt.axis([minx, maxx, miny, maxy])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.yscale('symlog')

        if self.byIndex:
            for i in range(len(x)):
                plt.plot(i, y[i], 'b.')
                plt.plot(i, y_pred[i], 'r.')
        else:
            for i in range(len(x)):
                plt.plot(x[i], y[i], 'b,')
                plt.plot(x[i], y_pred[i], 'r,')
        plt.title("Data & Prediction")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def writeGraph(self, img, writer):
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(img.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Add image summary
        summary_op = tf.summary.image("plot", image)
        # Session
        with tf.Session() as sess:
            # Run
            summary = sess.run(summary_op)
            # Write summary
            self.writer.add_summary(summary)
            self.writer.flush()
   


class MyRMSpropOptimizer(optimizers.RMSprop):
    def __init__(self, initEpochs, **kwargs):
        super(MyRMSpropOptimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            K.set_value(self.iterations, initEpochs);
    def getCurrentLearningRate(self):
        return K.eval(self.lr * (  # pylint: disable=g-no-augmented-assignment
          1. / (1. + self.decay * math_ops.cast(self.iterations,
            K.dtype(self.decay)))))

class MySGDOptimizer(optimizers.SGD):
    def __init__(self, initEpochs, **kwargs):
        super(MySGDOptimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            K.set_value(self.iterations, initEpochs);
    def getCurrentLearningRate(self):
        return K.eval(self.lr * (1. / (1. + self.decay * K.cast(self.iterations, 
            K.dtype(self.decay)))))

class MyAdamOptimizer(optimizers.Adam):
    def __init__(self, initEpochs, **kwargs):
        super(MyAdamOptimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            K.set_value(self.iterations, initEpochs);
    def getCurrentLearningRate(self):
        return K.eval(self.lr * (1. / (1. + self.decay * K.cast(self.iterations, 
            K.dtype(self.decay)))))

   
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_4_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=4)

def top_n_accuracy(n):
    return lambda y_true, y_pred: top_k_categorical_accuracy(y_true, y_pred, k=n)