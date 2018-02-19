from shutil import copyfile
import csv
import os
import datetime


class CopyFile(object):
    def __init__(self):
        # Define o caminho onde o dataset está.
        #self.PATH = '/media/sf_paulo/RNP/dataset/X-ray-chest'
        self.PATH = '/media/paulo/84DAF282DAF26FB2/paulo/RNP/dataset/X-ray-chest'
        self.data_path = self.PATH + '/imagens'
        self.data_dir_list = os.listdir(self.data_path)

        self.imagens = []
        self.classes = []

    def open_CSV(self):
        # Abre arquivo csv com o nome das imagens e suas respectivas classes.
        #with open('/media/sf_paulo/RNP/dataset/X-ray-chest/image-class.csv', newline='') as csvfile:
        with open('/home/paulo/mestrado/dataset/x-ray-chest/BBox_List_2017.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            i = 0
            # define um dicionario para armazenar o nome das imagens e as suas respectivas classes do arquivo csv.
            # imagens_classes = {}

            # Laço que percorre os registros do arquivo csv e seta as listas de imagens e classes.
            for row in reader:

                imagem = row['Image Index']
                classe = row['Finding Label']

                self.imagens.append(imagem)
                self.classes.append(classe)
                print(i,imagem + ' - ' + classe)
                i = i + 1

    def copy_file(self):
        #self.classe = ""
        #classe = 'a'
        for dataset in self.data_dir_list:

            # Carrega as imagens do diretório.
            #img_list = os.listdir(self.data_path + '/' + dataset)
            # print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
            # Percorre a lista de imagens para processá-las e adicioná-las à lista de imagens (img_data_list).
            for img in self.imagens:
                src = self.data_path + '/' + dataset + '/' + img
                try:
                    # Recupera o índice da imagem na lista de imagens.
                    image_index = self.imagens.index(img)
                except ValueError:
                    print('Nao encontrado:' + img)
                else:
                    # Recupera a classe correspondente à imagem recuperada acima.
                    classe1 = self.classes[image_index]

                    classe = classe1.replace("|", "-")
                    #Monta o diretorio onde será gravada a imagem (com o nome da classe).
                    #dst_dir = self.data_path + '/' + dataset + '/' + classe
                    dst_dir = '/home/paulo/mestrado/dataset/x-ray-chest' + '/' + dataset + '/' + classe
                    #Monta o diretório e o nome de destino da imagem.
                    dst = '/home/paulo/mestrado/dataset/x-ray-chest' + '/' + '/' + dataset + '/' + classe + '/' + img

                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)

                    copyfile(src, dst)


print('Inicio:', datetime.datetime.now())
x = CopyFile()
x.open_CSV()
x.copy_file()
print('Fim:', datetime.datetime.now())