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

    def get_detected_box(self, path):
        self.labels_train = pd.read_csv(path, sep=' ', names=["n_image", "i", "j", "h", "l"], header=None)
        for idx, label in self.labels_train.iterrows():
            self.original_image[label[0]].dec
            print(f'{label[0]}  {label[1]}  {label[2]} {label[3]}  {label[4]}')

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

    def save_data_to_image(self, image_array, path):
        '''save nparray as image in folder
            Args:
                image_array: list of numpy arrays
            path:
                folder path
        '''
        for idx, image in enumerate(image_array):
            imsave(f'{path}/{idx:05d}.jpg', image)
        print(f'{idx + 1} image saved in {path}.')

def hog_svm(date):
    load_data = LoadData()
    generate_data = GenerateData()
    # type Image
    images = load_data.get_original_images()
    X_train, y_train = Model.load_pos_neg_from_images(generate_data.generate_positive(images), generate_data.generate_negative(images))
    X_train_hog = Model.hog(X_train)
    svm = Model.SVM(X_train_hog, y_train)
    Model.save_model(svm, f'./result/hog_svm_{date}.sav')

if __name__ == '__main__':
    t = time.time()
    # hog_svm('4_25')
    load_data = LoadData()
    load_data.get_detected_box('./result/result_4_25/detection.txt')


    time = time.time() - t
    print(f'time used: {time}')
