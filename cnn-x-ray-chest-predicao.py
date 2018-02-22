from keras.models import model_from_json
import numpy as np
import os, os.path, cv2
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from glob import glob

import csv

#% matplotlib inline
class predicao(object):
    def __init__(self):

        self.PATH = '/home/paulo/mestrado/dataset/x-ray-chest'
        self.data_path = self.PATH + '/data-1000'
        self.data_dir_list = os.listdir(self.data_path)

        # simple version for working with CWD
        #print
        #len([name for name in os.listdir('.') if os.path.isfile(name)])

        # path joining version for other paths
        #DIR = self.data_path



        self.img_rows = 64
        self.img_cols = 64
        self.num_channel = 1
        self.qtde_imagens = 1000 #len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        self.qtde_imagens_predicao = 10

        # Lista com as imagens do dataset
        self.img_data_list = []
        self.img_data = []
        # Cria array de rótulos
        self.labels = []

        # Define o número de classes classes
        self.num_classes = 5
        # Define um dicionário com as classes e seus valores.
        self.classes = {'Atelectasis': 0,  # 4212 *
                        'Effusion': 1,  # 3959 *
                        'Infiltration': 2,  # 9551 *
                        'Nodule': 3,  # 2706 *
                        'Pneumothorax': 4  # 2199 *
                        }
        self.x = []
        self.y = []
        # Armazena valores das classes no formato one-hot encoding
        self.Y = []

        # Define listas para armazenar as imagens e suas respectivas classes para ser
        self.imagens = []
        self.classes_imagens = []

        self.lista_nome_imagens = []

    def open_CSV(self):
        # Abre arquivo csv com o nome das imagens e suas respectivas classes.
        with open(self.PATH + '/lista-imagem-classe-1000.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            i = 0
            # define um dicionario para armazenar o nome das imagens e as suas respectivas classes do arquivo csv.
            # imagens_classes = {}

            # Laço que percorre os registros do arquivo csv e seta as listas de imagens e classes.
            for row in reader:
                imagem = row['Image Index']
                classe = row['Finding Labels']

                self.imagens.append(imagem)
                self.classes_imagens.append(classe)
            # imagens_classes[i] = row
            # i = i + 1
            # print(row['Image Index'], row['Finding Labels'])

    def load_dataset(self):
        self.img_data = np.array(self.img_data_list)

        # Recupera a quantidade de imagens do vetor img_data.
        numero_de_imagens = self.qtde_imagens
        # Seta array de rótulos com a quantidade de imagens.
        self.labels = np.ones((numero_de_imagens,), dtype='float')
        i = 0
        # Carrega as imagens do dataset.
        # Laço que percorre o diretório com as imagens do dataset.
        for dataset in self.data_dir_list:
            # Carrega as imagens do diretório.
            img_list = os.listdir(self.data_path + '/' + dataset)
            # print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
            # Percorre a lista de imagens para processá-las e adicioná-las à lista de imagens (img_data_list).
            for img in img_list:
                self.lista_nome_imagens.append(img)

                input_img = cv2.imread(self.data_path + '/' + dataset + '/' + img)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize = cv2.resize(input_img, (self.img_rows, self.img_cols), interpolation=cv2.INTER_CUBIC)
                self.img_data_list.append(input_img_resize)

                # Recupera o índice da imagem na lista de imagens.
                image_index = self.imagens.index(img)
                # Recupera a classe correspondente à imagem recuperada acima.
                classe = self.classes_imagens[image_index]
                #print(img, classe)
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
        #print(self.lista_imagens_nome )
        #print(self.labels)

    def convert_dataset(self):
        # Transforma o tipo de dados de int para float32.
        self.img_data = np.array(self.img_data_list)
        self.img_data = self.img_data.astype('float32')
        # Normalize os valores dos pixels de 0 a 1 dividindo o valor original por 255
        self.img_data /= 255
        #print(self.img_data.shape)

        if self.num_channel == 1:
            if K.image_dim_ordering() == 'th':
                self.img_data = np.expand_dims(self.img_data, axis=1)
                #print(self.img_data.shape)
            else:
                self.img_data = np.expand_dims(self.img_data, axis=4)
                #print(self.img_data.shape)

        else:
            if K.image_dim_ordering() == 'th':
                self.img_data = np.rollaxis(self.img_data, 3, 1)
                #print(self.img_data.shape)

    def plot_image(self):

        #imagens = []
        #imagens[0] = ''
        #imagens[1] = ''
        #imagens[2] = ''
        #imagens[3] = ''
        #imagens[4] = ''

        for i in range(5):
            nome_imagem = self.lista_nome_imagens[i]

            input_img = cv2.imread(self.data_path + '/data/' + nome_imagem)

            image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))

            plt.figure(i)
            #plt.plot(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(nome_imagem + ' Classe: ' + self.classes_imagens[i])
            #plt.show()
            # plt.savefig('/tmp/imagem.png')
            plt.imshow(image)
            plt.show()


    def predict(self):
        # carrega a arquitetura da rede.
        model_architecture = '/home/paulo/PycharmProjects/cnn-chest-x-ray/xray-chest_architecture-89-perc.json'
        # carrega os pesos aprendidos pela rede no treinamento
        model_weights = '/home/paulo/PycharmProjects/cnn-chest-x-ray/xray-chest_weights-89-perc.h5'
        # Lê o modelo
        model = model_from_json(open(model_architecture).read())
        # Lê os pesos
        model.load_weights(model_weights)

        #nome_imagem =[]
        #classes = []
        #imgs = []
        #data_path = '/home/paulo/mestrado/dataset/x-ray-chest/data-predicao/'
        #nome_imagem.append('00000181_061.png')
        #classes.append('Atelectasis')
        #nome_imagem.append('00012973_005.png')
        #classes.append('Effusion')

        #input_img = cv2.imread(data_path + nome_imagem[0])
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        #input_img_resize = cv2.resize(input_img, (self.img_rows, self.img_cols), interpolation=cv2.INTER_CUBIC)
        #imgs.append(input_img_resize)

        #input_img = cv2.imread(data_path + nome_imagem[1])
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        #input_img_resize = cv2.resize(input_img, (self.img_rows, self.img_cols), interpolation=cv2.INTER_CUBIC)
        #imgs.append(input_img_resize)

        #imgs = np.array(imgs)
        #imgs = imgs.astype('float32')
        # Normalize os valores dos pixels de 0 a 1 dividindo o valor original por 255
        #imgs /= 255
        #imgs = np.expand_dims(imgs, axis=4)

        imgs = self.img_data[0:self.qtde_imagens_predicao]


        #Faz a predição para as imagens do vetor.
        predictions = model.predict_classes(imgs)

        print('Predição:')
        for i in range(self.qtde_imagens_predicao):

            target = predictions[i]

            for key, value in self.classes.items():
                if value == target:
                    predicao = key
                    break
            #print('Imagem: ' + self.lista_nome_imagens[i] + ' Classe: ' + predicao)

            #Carrega a imagem
            input_img = cv2.imread(self.data_path + '/data/' + self.lista_nome_imagens[i])
            #input_img = cv2.imread(self.data_path + '/data/' + nome_imagem[i])
            #input_img = cv2.imread(data_path + nome_imagem[i])
            image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            #Faz o resize da imagem para 512x512
            image = cv2.resize(image, (512, 512))

            plt.figure(i)
            plt.title(self.lista_nome_imagens[i] + ' Classe: ' + self.classes_imagens[i])
            #plt.title(nome_imagem[i] + ' Classe: ' + classes[i])
            plt.xlabel('Classe Predição: ' + predicao)
            plt.imshow(image)
            plt.show()

        #print(self.lista_nome_imagens[0:5])
        #print(self.Y[0:5])
        # print('[0 1]')
        print(predictions)


pr = predicao()
pr.open_CSV()
pr.load_dataset()
pr.convert_class_one_hot_encoding()
pr.convert_dataset()
#pr.plot_image()
pr.predict()