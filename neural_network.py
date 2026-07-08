#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════╗
║  🧠 شبكة عصبية من الصفر - برينياك وسايبر      ║
║  Neural Network from Scratch                   ║
║  تعمل على GitHub Actions                      ║
╚══════════════════════════════════════════════════╝
"""

import numpy as np
import struct
import os
import gzip
import urllib.request
from datetime import datetime

# ==================== 1. تحميل بيانات MNIST ====================

class MNISTLoader:
    def __init__(self, data_dir='./mnist_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.urls = {
            'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
            'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
            'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
            'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
        }
    
    def download(self):
        for name, url in self.urls.items():
            filepath = os.path.join(self.data_dir, os.path.basename(url))
            if not os.path.exists(filepath):
                print(f"📥 تحميل {name}...")
                urllib.request.urlretrieve(url, filepath)
        print("✅ اكتمل التحميل!")
    
    def load_images(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
            images = images.astype(np.float32) / 255.0
        return images
    
    def load_labels(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with gzip.open(filepath, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
    
    def load_all(self):
        self.download()
        self.train_images = self.load_images('train-images-idx3-ubyte.gz')
        self.train_labels = self.load_labels('train-labels-idx1-ubyte.gz')
        self.test_images = self.load_images('t10k-images-idx3-ubyte.gz')
        self.test_labels = self.load_labels('t10k-labels-idx1-ubyte.gz')
        print(f"📊 تدريب: {len(self.train_images)} | اختبار: {len(self.test_images)}")
        return self

# ==================== 2. طبقات الشبكة العصبية ====================

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        if activation == 'relu':
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)
        
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.z = None
        self.output = None
    
    def forward(self, inputs):
        self.input = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.output = 1.0 / (1.0 + np.exp(-np.clip(self.z, -500, 500)))
        elif self.activation == 'softmax':
            shifted_z = self.z - np.max(self.z, axis=1, keepdims=True)
            exp_z = np.exp(shifted_z)
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            self.output = self.z
        
        return self.output
    
    def backward(self, d_output, learning_rate):
        batch_size = self.input.shape[0]
        
        if self.activation == 'relu':
            d_z = d_output * (self.z > 0).astype(np.float32)
        elif self.activation == 'sigmoid':
            sig = self.output
            d_z = d_output * sig * (1 - sig)
        elif self.activation == 'softmax':
            d_z = d_output
        else:
            d_z = d_output
        
        d_weights = np.dot(self.input.T, d_z) / batch_size
        d_bias = np.sum(d_z, axis=0, keepdims=True) / batch_size
        d_input = np.dot(d_z, self.weights.T)
        
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        
        return d_input

class Dropout:
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True
    
    def forward(self, inputs):
        if self.training:
            self.mask = (np.random.rand(*inputs.shape) > self.drop_rate) / (1 - self.drop_rate)
            return inputs * self.mask
        return inputs
    
    def backward(self, d_output, learning_rate):
        if self.training:
            return d_output * self.mask
        return d_output

class CrossEntropyLoss:
    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        correct_probs = predictions[np.arange(batch_size), targets]
        loss = -np.mean(np.log(correct_probs))
        return loss
    
    def backward(self, predictions, targets):
        batch_size = predictions.shape[0]
        grad = predictions.copy()
        grad[np.arange(batch_size), targets] -= 1
        grad /= batch_size
        return grad

# ==================== 3. النموذج الكامل ====================

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_fn = CrossEntropyLoss()
        self.train_losses = []
        self.val_accuracies = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs, training=True):
        x = inputs
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training
            x = layer.forward(x)
        return x
    
    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
    
    def train(self, train_images, train_labels, val_images, val_labels,
              epochs=15, batch_size=64, learning_rate=0.01):
        num_samples = len(train_images)
        num_batches = num_samples // batch_size
        
        print(f"\n{'='*60}")
        print(f"🚀 بدء التدريب: {epochs} دورة, {num_batches} دفعة/دورة")
        print(f"📊 الأوزان: {sum(l.weights.size for l in self.layers if hasattr(l, 'weights')):,}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            train_images = train_images[indices]
            train_labels = train_labels[indices]
            
            epoch_loss = 0.0
            
            for batch in range(num_batches):
                start = batch * batch_size
                end = start + batch_size
                batch_images = train_images[start:end]
                batch_labels = train_labels[start:end]
                
                predictions = self.forward(batch_images, training=True)
                loss = self.loss_fn.forward(predictions, batch_labels)
                epoch_loss += loss
                
                grad = self.loss_fn.backward(predictions, batch_labels)
                self.backward(grad, learning_rate)
            
            avg_loss = epoch_loss / num_batches
            self.train_losses.append(avg_loss)
            
            val_acc = self.evaluate(val_images[:500], val_labels[:500])
            self.val_accuracies.append(val_acc)
            
            bar = '█' * int(val_acc * 20) + '░' * (20 - int(val_acc * 20))
            print(f"📊 دورة {epoch+1:2d}/{epochs} | {bar} | خسارة: {avg_loss:.4f} | دقة: {val_acc:.2%}")
        
        print(f"\n✅ اكتمل التدريب!")
    
    def predict(self, images):
        predictions = self.forward(images, training=False)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, images, labels):
        predictions = self.predict(images)
        return np.mean(predictions == labels)

# ==================== 4. التشغيل ====================

def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  🧠 شبكة عصبية من الصفر - GitHub Actions      ║")
    print("╚══════════════════════════════════════════════════╝\n")
    
    # تحميل البيانات
    mnist = MNISTLoader()
    mnist.load_all()
    
    # استخدام 20,000 للتدريب (GitHub Actions أسرع من الموبايل)
    train_size = 20000
    test_size = 5000
    
    train_images = mnist.train_images[:train_size]
    train_labels = mnist.train_labels[:train_size]
    test_images = mnist.test_images[:test_size]
    test_labels = mnist.test_labels[:test_size]
    
    # بناء النموذج
    model = NeuralNetwork()
    model.add(DenseLayer(784, 256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(DenseLayer(256, 128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(DenseLayer(128, 10, activation='softmax'))
    
    total_weights = 784*256 + 256*128 + 128*10
    print(f"🏗️ هيكل النموذج: 784 -> 256 -> 128 -> 10")
    print(f"📊 إجمالي الأوزان: {total_weights:,}\n")
    
    # تدريب
    model.train(train_images, train_labels, test_images, test_labels,
                epochs=15, batch_size=128, learning_rate=0.01)
    
    # تقييم نهائي
    final_acc = model.evaluate(test_images, test_labels)
    
    print(f"\n{'='*60}")
    print(f"🎯 النتيجة النهائية:")
    print(f"   الدقة على بيانات الاختبار: {final_acc:.2%}")
    print(f"   عدد الأوزان المدربة: {total_weights:,}")
    print(f"   عدد الصور التدريبية: {train_size:,}")
    print(f"{'='*60}\n")
    
    # حفظ النتائج
    with open('results.txt', 'w') as f:
        f.write(f"الدقة النهائية: {final_acc:.2%}\n")
        f.write(f"عدد الأوزان: {total_weights:,}\n")
        f.write(f"الدقة عبر الدورات: {model.val_accuracies}\n")
        f.write(f"الخسارة عبر الدورات: {model.train_losses}\n")

if __name__ == "__main__":
    main()
