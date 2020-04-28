from models import *

class LoadData:
    original_image = [] # the 1000 label image
    train_data= []
    validation_data = []
    test_data = []
    positive_image = []
    negative_image = []
    new_train_date = []

    def __init__(self, labels_path = './project_train/label_train.txt'):
        self.labels_train = pd.read_csv(labels_path, sep=' ', names=["n_image", "i", "j", "h", "l"], header=None)
    '''
        get all images in train folder and instance them to type Image
        args:
            path to train fold
        return:
            a list of type Images
    '''
    def get_original_images(self, path='./project_train/train'):
        for i in range(1000):
            im = color.rgb2gray(imread(f'{path}/{i+1:04d}.jpg'))
            labels = self.labels_train.loc[self.labels_train['n_image'] == i+1]
            im_object = Image(i+1, im)
            for idx, label in labels.iterrows():
                im_object.add_face(Box(label[1], label[2], label[3], label[4]))
            self.original_image.append(im_object)
        return  self.original_image
    '''
        Add all detected boxs to the correspnding image of a list of type image 
        args:
            images: 
                list of type Image
            path: 
                path to the detection.txt
    '''
    def get_detected_box(self, images, path):
        self.labels_train = pd.read_csv(path, sep=' ', names=["n_image", "i", "j", "h", "l"], header=None)
        self.labels_train['n_image'].astype(int)
        # self.labels_train['i'].astype(int)
        # self.labels_train['j'].astype(int)
        # self.labels_train['h'].astype(int)
        # self.labels_train['l'].astype(int)
        i = 0
        for image in images:
            n_image  = image.number
            boxs = self.labels_train.loc[self.labels_train['n_image'] == n_image]
            for idx, label in boxs.iterrows():
                b = Box(label[1], label[2], label[3], label[4])
                image.add_detected(b)
                i += 1
        print(f'{i} images added in detected')

    def get_test_images(self, path='./project_test'):
        for i in range(500):
            im = color.rgb2gray(imread(f'{path}/{i+1:04d}.jpg'))
            im_object = Image(i+1, im)
            self.test_data.append(im_object)
        return self.test_data

    def split_data(self, ratio):
        '''split 1000 training data'''
        idx = round(len(self.original_image) * ratio)
        self.train_data = self.original_image[0:idx]
        self.validation_data = self.original_image[idx:]
        return self.train_data, self.test_data


class GenerateData:
    def generate_positive(self, images, h=HEIGHT, w=WIDTH):
        """generate a set of positive training samples from a dataset lebelled
        Args:
            images: array of Images
        Returns:
            array of image matrix of the same size
        """
        positive = []
        i = 0
        for image in images:
            im = image.image
            for face in image.face:
                im_face = im[face.y:face.y + face.height, face.x:face.x + face.width]
                im_face = transform.resize(im_face, (HEIGHT, WIDTH))
                positive.append(im_face)
                i += 1
        print(f'{i} positive image generated.')
        return positive

    def generate_neg_FP(images, h=HEIGHT, w=WIDTH):
        """generate a set of positive training samples from a dataset lebelled
        Args:
            images: array of Images
        Returns:
            array of image matrix of the same size
        """
        negative = []
        i = 0
        for image in images:
            im = image.image
            for face in image.detected:
                FP = True
                for TP in image.face:
                    if face.test_true_positive(TP) == True:
                        FP = False
                if FP == True:
                    im_face = im[int(face.y):int(face.y + face.height), int(face.x):int(face.x + face.width)]
                    im_face = transform.resize(im_face, (HEIGHT, WIDTH))
                    negative.append(im_face)
                    i += 1
        print(f'{i} negative image generated.')
        return negative

    def generate_neg_position(self, x_max, y_max, width, height):
        '''generate the position of the top left of the negative image'''
        x = np.random.randint(0, x_max - width)
        y = np.random.randint(0, y_max - height)
        return (x, y)

    def generate_negative(self, images, h=HEIGHT, w=WIDTH, n_sample_per_image=5):
        """generate a set of negative training samples from a dataset lebelled
        Args:
            images: array of type Images
        Returns:
            array of image matrix of the same size
        """
        negative = []
        i = 0
        for image in images:
            im = image.image
            x_max = im.shape[1]
            y_max = im.shape[0]
            for k in range(5):
                pos = self.generate_neg_position(x_max, y_max, w, h)
                box_generated = Box(pos[1], pos[0], h, w, s=0)
                for face in image.face:
                    box_generated.test_true_positive(face)
                if box_generated.true_positive == False:
                    neg_im = im[box_generated.y:box_generated.y + box_generated.height,
                             box_generated.x:box_generated.x + box_generated.width]
                    negative.append(neg_im)
                    i += 1
        print(f'{i} negative image generated.')
        return negative

    def save_data_to_image(image_array, path):
        '''save nparray as image in folder
            Args:
                image_array: list of numpy arrays
            path:
                folder path
        '''
        for idx, image in enumerate(image_array):
            imsave(f'{path}/{idx:05d}.jpg', image)
        print(f'{idx + 1} image saved in {path}.')

def hog_svm(date, X_train, y_train):
    load_data = LoadData()
    generate_data = GenerateData()
    # type Image
    images = load_data.get_original_images()
    X_train_hog = Model.hog(X_train)
    clf = svm.SVC(class_weight='balanced')
    clf.fit(X_train_hog, y_train)
    #svm = Model.SVM(X_train_hog, y_train)
    Model.save_model(svm, f'./result/hog_svm_{date}.sav')

def augment_neg_FP(path_detection, path_folder):
    load_data = LoadData()
    load_data.get_original_images()
    load_data.get_detected_box(load_data.original_image, path_detection)
    neg_image = GenerateData.generate_neg_FP(load_data.original_image)
    GenerateData.save_data_to_image(neg_image, path_folder)
if __name__ == '__main__':
    # hog_svm('4_25')
    # augment_neg_FP('./result/result_4_27/train_result/detection.txt', './project_train/neg_svm_hog3')
    t = time.time()
    path_pos = ['./project_train/positive']
    path_neg = ['./project_train/negative', './project_train/neg_svm_hog', './project_train/neg_svm_hog2', './project_train/neg_svm_hog3']
    X_train, y_train = Model.load_combine_pos_neg_data(path_pos, path_neg)
    hog_svm('4_27', X_train, y_train)
    time = time.time() - t
    print(f'time used: {time}')
