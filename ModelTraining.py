# -*- coding: utf-8 -*-
__author__ = '翁飞龙'
import cv2 as cv
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from FaceDateSet import load_dataset, resize_image, IMAGE_SIZE
import warnings
warnings.filterwarnings('ignore')
 
 
class Dataset:
    def __init__(self, path_name, person_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        # 验证集
        # self.valid_images = None
        # self.valid_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None
        # 数据加载路径
        self.path_name = path_name
        # 当前库采用的维度顺序
        self.input_shape = None
        # Current person name
        self.person_name = person_name
 
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        # 加载数据集至内存
        print("The path is:", self.path_name)
        images, labels = load_dataset(self.path_name, self.person_name)
        
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 10))
        
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
 
            # 输出训练集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(test_images.shape[0], 'test samples')
            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
            # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)
            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            test_images = test_images.astype('float32')
            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255.0
            test_images /= 255.0
            self.train_images = train_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.test_labels = test_labels
 
 
# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None
        self.person_name = None
        # self.predicting_models = []
 
    # 建立模型
    def build_model(self, dataset, nb_classes=2):
        # 构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()

        # 原始代码（一定要保留）
        # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Conv2D(32, 3, 3, padding='same',
                                     input_shape=dataset.input_shape))  # 1 2维卷积层
        self.model.add(Activation('relu'))  # 2 激活函数层
 
        self.model.add(Conv2D(32, 3, 3, padding='same'))  # 3 2维卷积层
        self.model.add(Activation('relu'))  # 4 激活函数层
 
        self.model.add(MaxPool2D(pool_size=(2, 2)))  # 5 池化层
        self.model.add(Dropout(0.25))  # 6 Dropout层
 
        self.model.add(Conv2D(64, 3, 3, padding='same'))  # 7  2维卷积层
        self.model.add(Activation('relu'))  # 8  激活函数层
 
        self.model.add(Conv2D(64, 3, 3, padding='same'))  # 9  2维卷积层
        self.model.add(Activation('relu'))  # 10 激活函数层
 
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='same'))  # 11 池化层
        self.model.add(Dropout(0.25))  # 12 Dropout层
 
        self.model.add(Flatten())  # 13 Flatten层
        self.model.add(Dense(512))  # 14 Dense层,又被称作全连接层
        self.model.add(Activation('relu'))  # 15 激活函数层
        self.model.add(Dropout(0.5))  # 16 Dropout层
        self.model.add(Dense(nb_classes))  # 17 Dense层
        self.model.add(Activation('softmax'))  # 18 分类层，输出最终结果

        # 输出模型概况
        self.model.summary()
 
    # 训练模型
    def train(self, dataset, batch_size=20, nb_epoch=200, data_augmentation=False):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作
 
        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_data=(dataset.test_images, dataset.test_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            print("使用了数据提升")
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转
 
            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)
 
            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                                epochs = nb_epoch,
                                                validation_data=(dataset.test_images, dataset.test_labels))
        return 'Finished'
 
    MODEL_PATH = './Model/face.model.h5'
 
    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)
 
    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)
    
    def load_person_name(self, person_name):
        self.person_name =  person_name  

    def get_person_name(self):
        return self.person_name

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        # print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print(f'{self.model.metrics_names[1]}:{score[1] * 100}%')
 

    # 识别人脸
    def face_predict(self, image):
        # 依然是根据后端系统确定维度顺序
        #if K.image_dim_ordering() == 'th' 
        if K.image_data_format() == 'channels_first'and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 与模型训练不同，这次只是针对1张图片进行预测
        #elif K.image_dim_ordering() == 'tf' 
        elif K.image_data_format() == 'channels_last'and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
 
            # 浮点并归一化
        image = image.astype('float32')
        image /= 255.0

        result = self.model.predict(image)

        print("This is the model of: ", self.get_person_name())
        print("The data type of results:", result)
        # 返回类别预测结果
        if result[0][0] > result[0][1]:
            return 'T', result[0][0]
        else:
            return 'F', result[0][1]
 
if __name__ == '__main__':
    print("The program has reached in here!!!-----")
    dataset = Dataset('FaceImageDate', 'mingyang')
    dataset.load()
 
    # # 训练模型
    model = Model()
    model.build_model(dataset)
    
    # 测试训练函数的代码
    model.train(dataset)
    
    model.save_model(file_path='./Model/mingyang.face.model.h5')

    # # 评估模型
    model = Model()
    model.load_model(file_path='./Model/mingyang.face.model.h5')

    # model_2 = Model()
    # model_2.load_model(file_path='./Model/mingyang.face.model.h5')
    # model_3 = Model()
    # model_3.load_model(file_path='./Model/jinbo.face.model.h5')

    # image2 = cv.imread("11_197.jpg")
    # result = model.face_predict(image2)
    # # result = model_2.face_predict(image2)
    # # result = model_3.face_predict(image2)

    # print("The final result is:", result)
    model.evaluate(dataset)
    del dataset