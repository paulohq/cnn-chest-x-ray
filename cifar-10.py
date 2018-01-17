from numpy.random import seed
seed(1) #informa uma semente para a reprodutibilidade dos resultados.
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import datetime

class CifarCNN(object):
    def __init__(self):
        #define as variáveis que serão usadas na classe.

        #Número de vezes que uma determinada imagem será aumentada
        self.NUM_TO_AUGMENT=5
        # CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
        #Número de canais da imagem.
        self.canais_img = 3
        #Número de pixels da imagem.
        self.linhas_img = 32
        self.colunas_img = 32
        #Tamanho do batch para atualização dos pesos.
        self.batch_size = 100
        #Número de épocas.
        self.epocas = 20
        #Número de classes de saída.
        self.numero_classes = 10
        self.verbose = 1
        #Define o percentual de dados para validação.
        self.validation_split = 0.2
        #Define o otimizador RMSProp para ser usado na rede.
        self.otimizador = RMSprop()
        #define os vetores que receberão os dataset de treinamento de teste (das imagens e labels).
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        #variável com o modelo da rede.
        self.model = Sequential()
        #Otimizador sgd
        self.sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        #variável que guarda valores de perda de treinamento (loss) e métricas (acurácia) em épocas sucessivas,
        # bem como valores de perda de validação e valores de métrica de validação(se aplicável).
        self.history = []


    def LoadDataSet(self):
        #Carrega o dataset do cifar para vetores de treinamento (X_train) e de teste (X_test).
        #Nos vetores X_train e X_test serão carregadas as imagens de treinamento e teste
        #e nos vetores y_train e y_test serão carregados os rótulos das imagens (labels) de treinamento e teste.
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = cifar10.load_data()
        #Reduz o tamanho do dataset para rodar na máquina sem GPU.
        #self.X_train = self.X_train[:10000]
        #self.X_test = self.X_test[:2000]
        #self.Y_train = self.Y_train[:10000]
        #self.Y_test = self.Y_test[:2000]

    def PrintRandomImage(self):
        fig = plt.figure(figsize=(8, 3))
        for i in range(self.numero_classes):
            ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
            idx = np.where(self.Y_train[:] == i)[0]
            features_idx = self.X_train[idx, ::]
            img_num = np.random.randint(features_idx.shape[0])
            im = np.transpose(features_idx[img_num, ::], (1, 2, 0))
            #ax.set_title(class_names[i])
            plt.imshow(im)
        plt.show()

    def AugmentingDataSet(self):
        # Método que cria imagens distorcidas de parte do dataset de treinamento do cifar-10 para aumentar
        # o número de imagens deste dataset e conseguentemente melhorar o aprendizado da rede.
        # pega 2000 imagens do dataset de treinamento.
        print("Aumentando o dataset de treinamento...")
        img_flip_ud = self.X_train[46000:50000]
        #gira as imagens horizontalmente.
        img_flip_ud = np.flipud(img_flip_ud)
        # pega 2000 imagens do dataset de treinamento.
        img_flip_lr = self.X_train[42000:46000]
        # gira as imagens verticalmente.
        img_flip_lr = np.fliplr(img_flip_lr)
        #pega 1000 imagens do dataset de treinamento
        img_flip = self.X_train[40000:42000]
        #inverte a ordem dos elementos de um array ao longo de um eixo.
        img_flip = np.flip(img_flip, axis=0)

        # adiciona as imagens distorcidas ao dataset de treinamento.
        self.X_train = np.concatenate((self.X_train, img_flip_ud, img_flip_lr, img_flip))

        #pega os 5000 últimos rótulos do dataset de rótulos de treinamento.
        lbl_flip = self.Y_train[:10000]
        #adiciona os rótulos ao dataset de rótulos de treinamento.
        self.Y_train = np.concatenate((self.Y_train, lbl_flip), axis=0)

    def Augmenting(self):
        #Método que cria imagens distorcidas de parte do dataset de treinamento do cifar-10 para aumentar
        # o número de imagens deste dataset e conseguentemente melhorar o aprendizado da rede.
        print("Aumenando dataset de treinamento...")
        datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                     zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
        qtde_imagens_modificadas = 10
        #datagen.fit(self.X_train)

        #xtas, ytas = [], []
        for i in range(qtde_imagens_modificadas):
            num_aug = 0
            x = self.X_train[i] # (3, 32, 32)
            x = x.reshape((1,) + x.shape) # (1, 3, 32, 32)
            y = self.Y_train[i]
            y = y.reshape((1,) + y.shape)
            #
            for x_aug in datagen.flow(x, batch_size=1): #, save_to_dir = '/home/paulo/desenv/cifar-10/extra', save_prefix = 'cifar', save_format = 'jpeg'):

                #xtas.append(x_aug[0])
                num_aug += 1
                a = np.concatenate((a, x_aug), axis=0)
                b = np.concatenate((b, y), axis=0)

                #np.flip()
                #np.rot90()
                if num_aug >= 1: #self.NUM_TO_AUGMENT:
                    break
        self.X_train = np.concatenate((self.X_train, a), axis=0)
        self.Y_train = np.concatenate((self.Y_train, b), axis=0)

    def PrintDataSet(self):
        #Imprime a quantidade de elementos nos vetores de treino e teste.
        print('X_train shape:', self.X_train.shape)
        print('X_test shape:', self.X_test.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

    def ConvertDataSet(self):
        #Pre-procesamento das imagens.

        # Converte os rótulos de numérico para vetor de categorias one-hot-encode.
        self.Y_train = np_utils.to_categorical(self.Y_train, self.numero_classes)
        self.Y_test = np_utils.to_categorical(self.Y_test, self.numero_classes)
        print(self.Y_train.shape, 'train samples')
        print(self.Y_test.shape, 'test samples')
        # transforma os dados de treinamento e teste de integer para o tipo de dados float32
        # para fazer a divisão abaixo.
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        #Como os pixels tem valores que variam de 0 a 255, conforme sua intensidade, vamos normalizar os valores
        #para a faixa de 0 a 1 dividindo cada valor de pixel pelo valor máximo que ele pode ter (255).
        self.X_train /= 255
        self.X_test /= 255


    def CreateModel(self):
        #Método que cria o modelo da rede com suas camadas (arquitetura da rede).

        #padding='same' inclui uma borda na imagem de entrada para que a saída seja do mesmo tamanho da entrada após aplicado o fitro.
        #padding='valid' sem padding (a imagem de sáida será diminuída em relação à imagem de entrada).
        #Cria camada convolucional com 32 filtros de tramnho 3x3 com padding de tamanho que a saída seja do mesmo tamanho da entrada.
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.linhas_img, self.colunas_img, self.canais_img)))
        #Cria camada com função de ativação ReLU.
        self.model.add(Activation('relu'))
        #Cria camada convolucional com 32 filtros de tramnho 3x3 com padding de tamanho que a saída seja do mesmo tamanho da entrada.
        self.model.add(Conv2D(32, (3, 3), padding='same'))
        #Cria camada com função de ativação ReLU.
        self.model.add(Activation('relu'))
        #Cria camada de pooling para dividir pela metade a saída da camada anterior.
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #Cria camada de dropout para desligar neurônios.
        self.model.add(Dropout(0.25))
        #Cria camada convolucional com 64 filtros de tramnho 3x3 com padding de tamanho que a saída seja do mesmo tamanho da entrada.
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        #Cria camada com função de ativação ReLU.
        self.model.add(Activation('relu'))
        #Cria camada convolucional com 32 filtros de tramnho 3x3 com padding de tamanho que a saída seja do mesmo tamanho da entrada.
        self.model.add(Conv2D(64, (3, 3)))
        #Cria camada com função de ativação ReLU.
        self.model.add(Activation('relu'))
        #Cria camada de pooling para dividir pela metade a saída da camada anterior.
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #Cria camada de dropout para desligar neurônios.
        self.model.add(Dropout(0.25))
        #Converte as os mapas de características 3D para vetores de características (1D).
        self.model.add(Flatten())
        #Cria camada densa (fully connected) com 512 neurônios.
        self.model.add(Dense(512))
        #Cria camada com função de ativação ReLU.
        self.model.add(Activation('relu'))
        #Cria camada de dropout para desligar neurônios.
        self.model.add(Dropout(0.5))
        #Cria camada densa (fully connected) com 10 neurônios para fazer a classificação dos dados.
        self.model.add(Dense(self.numero_classes))
        #Cria camada com função de ativação softmax para que a imagem seja classificada em uma das 10 classes definidas.
        self.model.add(Activation('softmax'))

    def SummaryModel(self):
        #Imprime resumo do modelo da rede. As camadas com suas respectivas saídas e número de parâmetros.
        self.model.summary()

    def CNN(self):
        #Método que compila o modelo criado, treina a rede e faz o teste.
        #Compila o modelo
        self.model.compile(loss='categorical_crossentropy', optimizer=self.otimizador,metrics=['accuracy'])
        #treina o modelo com os dados de treinamento, para o tamanho de batch e número de épocas definadas.
        #Separa os dados de treinamento em treinamento e validação.
        self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size,epochs=self.epocas, validation_split=self.validation_split,verbose=self.verbose)
        #testa o modelo com os dados separadas para teste com o tamanho de batch definido.
        score = self.model.evaluate(self.X_test, self.Y_test, batch_size=self.batch_size, verbose=self.verbose)
        print("Pontuação no teste:", score[0])
        print('Acurácia no teste:', score[1])

    def PrintAccuracyLoss(self):
        #Plota os gráficos de treino e validação com as curvas de acurácia e função de perda.
        plt.figure(0)
        plt.plot(self.history.history['acc'], 'r')
        plt.plot(self.history.history['val_acc'], 'g')
        plt.xticks(np.arange(0, self.epocas + 1, 2.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Épocas")
        plt.ylabel("Acurácia")
        plt.title("Acurácia no treino vs Acurácia na validação")
        plt.legend(['treino', 'validação'])
        plt.savefig('/tmp/grafico_acuracia.png')

        plt.figure(1)
        plt.plot(self.history.history['loss'], 'r')
        plt.plot(self.history.history['val_loss'], 'g')
        plt.xticks(np.arange(0, self.epocas + 1, 2.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Épocas")
        plt.ylabel("Função de Perda")
        plt.title("Perda no treino vs Perda na validação")
        plt.legend(['treino', 'validação'])

        plt.savefig('/tmp/grafico_perda.png')

        plt.show()

    def SaveModel(self):
        #salva o modelo
        model_json = self.model.to_json()
        open('cifar10_architecture.json', 'w').write(model_json)
        #salva os pesos aprendidos pela rede nos dados de treinamento.
        self.model.save_weights('cifar10_weights.h5', overwrite=True)
        #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        #visualizar com linha de comando.
        #tensorboard --logdir=/full_path_to_your_logs

#Instancia a classe.
cifar = CifarCNN()
#Chama método para ler o dataset.
cifar.LoadDataSet()
cifar.AugmentingDataSet()
cifar.PrintDataSet()
cifar.ConvertDataSet()
cifar.CreateModel()
print("Inicio:", datetime.datetime.now())
cifar.SummaryModel()
cifar.CNN()
cifar.PrintAccuracyLoss()
print("Fim:", datetime.datetime.now())
cifar.SaveModel()
