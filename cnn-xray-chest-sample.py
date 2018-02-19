# Importa bibliotecas
from numpy.random import seed

seed(7)  # Define uma semente para a reprodutibilidade dos resultados do modelo.
import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
# K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.layers.normalization import BatchNormalization
from keras import regularizers

import datetime

import csv


class CNN_XRayChest(object):
    def __init__(self):
        # Define o caminho onde o dataset está.
        self.PATH = '/home/paulo/mestrado/dataset/x-ray-chest'
        self.data_path = self.PATH + '/data-sample'
        self.data_dir_list = os.listdir(self.data_path)

        self.img_rows = 64
        self.img_cols = 64
        self.num_channel = 1
        # Número de épocas do modelo
        self.num_epoch = 50
        # Define o tamanho do batch.
        self.batch_size = 30
        # Tamanho dos dados de validacao
        self.validation_split = 0.2
        # otimizador
        # self.otimizador = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.otimizador = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0005, amsgrad=True)
        # self.otimizador = RMSprop(lr=0.001, decay=1e-6)
        # self.otimizador = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # Lista com as imagens do dataset
        self.img_data_list = []
        self.img_data = []
        # Variável que armazena a quantidade de imagens do dataset.
        self.numero_de_imagens = 0
        # Cria array de rótulos
        self.labels = []

        # Define o número de classes classes
        self.num_classes = 5
        # Define um dicionário com as classes e seus valores.
        self.classes = {'Atelectasis': 0,  # 4212 *
                        'Effusion': 1,  # 3959 *
                        'Nodule': 2,
                        'Consolidation': 3,
                        'Infiltration': 4  # 2199 *
                        }
        # Define os vetores que receberão os datasets de treinamento e teste (imagens e rótulos).
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.x = []
        self.y = []
        # Armazena valores das classes no formato one-hot encoding
        self.Y = []
        # Define o modelo sequencial
        self.model = Sequential()

        # variável que guarda valores de perda de treinamento (loss) e métricas (acurácia) em épocas sucessivas,
        # bem como valores de perda de validação e valores de métrica de validação(se aplicável).
        self.history = []

        # Define listas para armazenar as imagens e suas respectivas classes para ser
        self.imagens = []
        self.classes_imagens = []
        self.verbose = 1

        self.qtde_imagens = 1845  # 638

    def open_CSV(self):
        # Abre arquivo csv com o nome das imagens e suas respectivas classes.
        with open('/home/paulo/mestrado/dataset/x-ray-chest/sample_labels.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            i = 0
            # define um dicionario para armazenar o nome das imagens e as suas respectivas classes do arquivo csv.
            # imagens_classes = {}

            Atelectasis = "Atelectasis"
            Effusion = "Effusion"
            Nodule = "Nodule"
            Consolidation = "Consolidation"
            Infiltration = "Infiltration"
            # Laço que percorre os registros do arquivo csv e seta as listas de imagens e classes.
            for row in reader:
                imagem = row['Image Index']
                classe = row['Finding Labels']

                if Atelectasis in classe:
                    self.imagens.append(imagem)
                    self.classes_imagens.append(Atelectasis)
                elif Effusion in classe:
                    self.imagens.append(imagem)
                    self.classes_imagens.append(Effusion)
                elif Nodule in classe:
                    self.imagens.append(imagem)
                    self.classes_imagens.append(Nodule)
                elif Consolidation in classe:
                    self.imagens.append(imagem)
                    self.classes_imagens.append(Consolidation)
                elif Infiltration in classe:
                    self.imagens.append(imagem)
                    self.classes_imagens.append(Infiltration)

            # imagens_classes[i] = row
            # i = i + 1
            # print(row['Image Index'], row['Finding Labels'])

    def load_dataset(self):
        self.img_data = np.array(self.img_data_list)

        # Recupera a quantidade de imagens do vetor img_data.
        numero_de_imagens = self.img_data.shape[0]
        numero_de_imagens = self.qtde_imagens
        # Seta array de rótulos com a quantidade de imagens.
        self.labels = np.ones((numero_de_imagens,), dtype='float32')
        i = 0
        # Carrega as imagens do dataset.
        # Laço que percorre o diretório com as imagens do dataset.
        for dataset in self.data_dir_list:
            # Carrega as imagens do diretório.
            img_list = os.listdir(self.data_path + '/' + dataset)
            # print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
            # Percorre a lista de imagens para processá-las e adicioná-las à lista de imagens (img_data_list).
            for img in img_list:
                # print(img)
                input_img = cv2.imread(self.data_path + '/' + dataset + '/' + img)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize = cv2.resize(input_img, (self.img_rows, self.img_cols), interpolation=cv2.INTER_CUBIC)
                self.img_data_list.append(input_img_resize)

                # Recupera o índice da imagem na lista de imagens.
                image_index = self.imagens.index(img)
                # Recupera a classe correspondente à imagem recuperada acima.
                classe = self.classes_imagens[image_index]
                # Retorna o numero da classe para a descricao informada (key).
                if classe not in self.classes:
                    rotulo = 0
                else:
                    rotulo = self.classes[classe]

                # Adiciona o rotulo encontrado no array de rótulos.
                self.labels[i] = rotulo
                i += 1

    def convert_class_one_hot_encoding(self):
        # converte as classes para one-hot encoding
        self.Y = np_utils.to_categorical(self.labels)

    def augmenting_dataset(self):
        # Método que cria imagens distorcidas de parte do dataset de treinamento do cifar-10 para aumentar
        # o número de imagens deste dataset e conseguentemente melhorar o aprendizado da rede.
        # pega 200 imagens do dataset de treinamento.
        print("Aumentando o dataset de treinamento...")
        img_flip_ud = self.img_data[:self.qtde_imagens]
        # gira as imagens horizontalmente.
        img_flip_ud = np.flipud(img_flip_ud)
        # pega 200 imagens do dataset de treinamento.
        img_flip_lr = self.img_data[:self.qtde_imagens]
        # gira as imagens verticalmente.
        img_flip_lr = np.fliplr(img_flip_lr)
        # pega 200 imagens do dataset de treinamento
        img_flip = self.img_data[:self.qtde_imagens]
        # inverte a ordem dos elementos de um array ao longo de um eixo.
        img_flip = np.flip(img_flip, axis=0)
        # img_rot90 = self.img_data[:638]
        # img_rot90 = np.rot90(img_rot90, 1, (1, 2))

        # adiciona as imagens distorcidas ao dataset de treinamento.
        self.img_data = np.concatenate((self.img_data, img_flip_ud, img_flip_lr, img_flip))

        # pega os 5000 últimos rótulos do dataset de rótulos de treinamento.
        lbl_flip = self.Y[:self.qtde_imagens]
        lbl_flip1 = self.Y[:self.qtde_imagens]
        lbl_flip2 = self.Y[:self.qtde_imagens]
        # lbl_flip3 = self.Y[:638]
        # adiciona os rótulos ao dataset de rótulos de treinamento.
        self.Y = np.concatenate((self.Y, lbl_flip, lbl_flip1, lbl_flip2), axis=0)

    def convert_dataset(self):
        # Transforma o tipo de dados de int para float32.
        self.img_data = np.array(self.img_data_list)
        self.img_data = self.img_data.astype('float32')
        # Normalize os valores dos pixels de 0 a 1 dividindo o valor original por 255
        self.img_data /= 255
        print(self.img_data.shape)

        if self.num_channel == 1:
            if K.image_dim_ordering() == 'th':
                self.img_data = np.expand_dims(self.img_data, axis=1)
                print(self.img_data.shape)
            else:
                self.img_data = np.expand_dims(self.img_data, axis=4)
                print(self.img_data.shape)

        else:
            if K.image_dim_ordering() == 'th':
                self.img_data = np.rollaxis(self.img_data, 3, 1)
                print(self.img_data.shape)

    def split_dataset(self):
        # print('aquii')
        # Embaralha o dataset
        self.x, self.y = shuffle(self.img_data, self.Y, random_state=2)
        # self.x = self.img_data
        # self.y = self.Y
        # print('aquiie')
        # Divide o dataset em treinamento e teste, sendo 20% para teste.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3,
                                                                                random_state=2)
        # print('aquia')

    def create_model(self):
        # Cria o modelo
        # input_shape = self.img_data[0].shape
        # self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.linhas_img, self.colunas_img, self.canais_img)
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format=None,
                              dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                              activity_regularizer=None,
                              kernel_constraint=None, bias_constraint=None,
                              input_shape=(self.img_rows, self.img_cols, self.num_channel)))
        # self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.img_rows, self.img_cols, self.num_channel )))
        # self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation('softmax'))

    def create_model1(self):
        # Método que cria o modelo da rede com suas camadas (arquitetura da rede).
        reg = 0.001
        # print('aqui')
        # padding='same' inclui uma borda na imagem de entrada para que a saída seja do mesmo tamanho da entrada após aplicado o fitro.
        # padding='valid' sem padding (a imagem de sáida será diminuída em relação à imagem de entrada).
        # Cria camada convolucional com 32 filtros de tramnho 3x3 com padding de tamanho que a saída seja do mesmo tamanho da entrada.
        self.model.add(
            Conv2D(32, (3, 3), padding='same', input_shape=(self.img_rows, self.img_cols, self.num_channel),
                   kernel_regularizer=regularizers.l2(reg)))
        self.model.add(BatchNormalization())
        # Cria camada com função de ativação ReLU.
        self.model.add(Activation('relu'))
        # Cria camada convolucional com 32 filtros de tramnho 3x3 com padding de tamanho que a saída seja do mesmo tamanho da entrada.
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.img_rows, self.img_cols, self.num_channel),
                              kernel_regularizer=regularizers.l2(reg)))
        self.model.add(BatchNormalization())
        # Cria camada com função de ativação ReLU.
        self.model.add(Activation('relu'))
        # Cria camada de pooling para dividir pela metade a saída da camada anterior.
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(BatchNormalization())
        # Cria camada de dropout para desligar neurônios.
        self.model.add(Dropout(0.25))
        # print('aqui1')

        # Cria camada convolucional com 64 filtros de tramnho 3x3 com padding de tamanho que a saída seja do mesmo tamanho da entrada.
        # self.model.add(Conv2D(64, (3, 3), padding='same', input_shape=(self.img_rows, self.img_cols, self.num_channel), kernel_regularizer=regularizers.l2(reg)))
        # Cria camada com função de ativação ReLU.
        # self.model.add(Activation('relu'))
        # self.model.add(BatchNormalization())
        # Cria camada convolucional com 32 filtros de tramnho 3x3 com padding de tamanho que a saída seja do mesmo tamanho da entrada.
        # self.model.add(Conv2D(64, (3, 3), padding='same', input_shape=(self.img_rows, self.img_cols, self.num_channel), kernel_regularizer=regularizers.l2(reg)))
        # Cria camada com função de ativação ReLU.
        # self.model.add(Activation('relu'))
        # self.model.add(BatchNormalization())
        # Cria camada de pooling para dividir pela metade a saída da camada anterior.
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # Cria camada de dropout para desligar neurônios.
        # self.model.add(Dropout(0.25))

        # Converte as os mapas de características 3D para vetores de características (1D).
        self.model.add(Flatten())
        # Cria camada densa (fully connected) com 512 neurônios.
        self.model.add(Dense(128, kernel_regularizer=regularizers.l2(reg)))
        self.model.add(BatchNormalization())
        # Cria camada com função de ativação ReLU.
        self.model.add(Activation('relu'))
        # Cria camada de dropout para desligar neurônios.
        self.model.add(Dropout(0.5))
        # Cria camada densa (fully connected) com 5 neurônios para fazer a classificação dos dados.
        self.model.add(Dense(self.num_classes))
        # Cria camada com função de ativação softmax para que a imagem seja classificada em uma das 5 classes definidas.
        self.model.add(Activation('softmax'))
        # print('aqui2')

    def create_model2(self):
        input_shape = (self.img_rows, self.img_cols, self.num_channel)
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                              activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=86, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
        self.model.add(Conv2D(filters=86, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        # model.add(Dense(1024, activation = "relu"))
        # model.add(Dropout(0.5))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation="softmax"))

    def sumary_model(self):
        self.model.summary()

    def cnn(self):
        # class_weight1 = class_weight.compute_class_weight('balanced', np.unique(self.Y), self.Y)
        # print(class_weight1)
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
        self.model.compile(loss='categorical_crossentropy', optimizer=self.otimizador, metrics=["accuracy"])

        # treinamento
        # self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epocas, validation_split=self.validation_split, verbose=self.verbose)
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.num_epoch,
                                      verbose=self.verbose, validation_split=self.validation_split)

        # testa o modelo com os dados separados para teste com o tamanho de batch definido.
        score = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=self.verbose)
        print("Pontuação no teste:", score[0])
        print('Acurácia no teste:', score[1])

    def print_accuracy_loss(self):
        # Plota os gráficos de treino e validação com as curvas de acurácia e função de perda.
        plt.figure(0)
        plt.plot(self.history.history['acc'], 'r')
        plt.plot(self.history.history['val_acc'], 'g')
        plt.xticks(np.arange(0, self.num_epoch + 1, 2.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Épocas")
        plt.ylabel("Acurácia")
        plt.title("Acurácia no treino vs Acurácia no teste")
        plt.legend(['treino', 'teste'])
        plt.savefig('/tmp/grafico_acuracia.png')

        plt.figure(1)
        plt.plot(self.history.history['loss'], 'r')
        plt.plot(self.history.history['val_loss'], 'g')
        plt.xticks(np.arange(0, self.num_epoch + 1, 2.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Épocas")
        plt.ylabel("Função de Perda")
        plt.title("Perda no treino vs Perda no teste")
        plt.legend(['treino', 'teste'])

        plt.savefig('/tmp/grafico_perda.png')

        plt.show()

    def save_model(self):
        # salva o modelo
        model_json = self.model.to_json()
        open('xray-chest_architecture.json', 'w').write(model_json)
        # salva os pesos aprendidos pela rede nos dados de treinamento.
        self.model.save_weights('xray-chest_weights.h5', overwrite=True)


# keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
# visualizar com linha de comando.
# tensorboard --logdir=/full_path_to_your_logs


xray = CNN_XRayChest()
xray.open_CSV()
xray.load_dataset()
xray.convert_class_one_hot_encoding()
xray.convert_dataset()
xray.augmenting_dataset()
xray.split_dataset()
# xray.create_model()
xray.create_model1()
print("Início: ", datetime.datetime.now())
xray.sumary_model()
xray.cnn()
xray.print_accuracy_loss()
print("Fim: ", datetime.datetime.now())
xray.save_model()