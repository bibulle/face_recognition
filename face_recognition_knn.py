"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""

import math
from sklearn import neighbors
import os
import os.path
import re
import unicodedata
import progressbar
import time
import pickle
import json
import datetime
import logging
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

TRAIN_PATH = "/train_dir"
TEST_PATH = "/test_dir"

MODEL_FILE = "trained_knn_model.clf"

PROGRESS_FILE = os.path.join(TRAIN_PATH, 'progress.json')
progress = {
    'status': 'Starting',
    'countImagesCurrent': 0,
    'countImagesMax': 0,
    'countImagesTotal': 0,
    'nameImageCurrent': '',
    'nameImageMax': '',
    'newFacesCount': 0,
    'newFaces': '',
    'newFacesName': '',
    'lastStartTime': datetime.datetime.now()
}

CONFIG_FILE = os.path.join(TRAIN_PATH, 'config.json')
config = {
    "logProgress": True,
    "sortImages": False,
    "restartAskedTime": datetime.datetime.now(),
    "logLevel": 10
}


def trainModel(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    logging.info('Train Model')
    setStatus('Train Model')

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        if class_dir == "ignore":
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    logging.debug("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            logging.debug("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    logging.info("Training complete!")

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


def getImagesToTestFromDir(img_path):
    setStatus('Get images to test')
    dirs = []
    files = []

    for image_file in os.listdir(os.path.join(TEST_PATH, img_path)):
        full_file_path = os.path.join(TEST_PATH, img_path, image_file)
        if os.path.isdir(full_file_path):
            dirs.append(os.path.join(img_path, image_file))
        elif os.path.isfile(full_file_path) and os.path.splitext(full_file_path)[1][1:] in ALLOWED_EXTENSIONS:
            files.append(re.sub('^[.]/', '', os.path.join(img_path, image_file)))

    for d in dirs:
        for f in getImagesToTestFromDir(d):
            files.append(re.sub('^[.]/', '', f))

    return files


def getTrainedFaces(trainPath):
    logging.debug('getTrainedFaces')
    setStatus('Loading trained faces')
    trainedFace = {}
    for class_dir in os.listdir(trainPath):
        if not os.path.isdir(os.path.join(trainPath, class_dir)):
            continue
        for image_name in os.listdir(os.path.join(trainPath, class_dir)):
            trainedFace[unicodedata.normalize('NFC', re.sub("[.][^.]*$", "", image_name))] = class_dir
    # logging.debug(training[unicodedata.normalize('NFC',"Boite 1 - Juin 2001 ?_Numériser 13.jpeg (705, 1380, 757, 1432)")])
    logging.debug("getTrainedFaces done")
    return trainedFace


def getNewerTrainedFaceDate(trainPath):
    #logging.debug('getNewerTrainedFaceDate')
    maxdate = datetime.datetime.min
    for class_dir in os.listdir(trainPath):
        if not os.path.isdir(os.path.join(trainPath, class_dir)):
            continue
        if (datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(trainPath, class_dir))) > maxdate):
            maxdate = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(trainPath, class_dir)))
            #logging.debug(maxdate.strftime('%c')+' '+os.path.join(trainPath, class_dir))
    #logging.debug("getNewerTrainedFaceDate done")
    return maxdate


def saveProgress():
    global progress
    if (progress['countImagesCurrent'] > progress['countImagesMax']):
        progress['countImagesMax'] = progress['countImagesCurrent']
        progress['nameImageMax'] = progress['nameImageCurrent']

    with open(PROGRESS_FILE, 'w+') as outfile:
        json.dump(progress, outfile, indent=4, sort_keys=True, default=str)


def setStatus(status):
    global progress
    progress['status'] = status
    saveProgress()


def loadProgress():
    global progress
    try:
        with open(PROGRESS_FILE, 'r') as json_file:
            progress_tmp = json.load(json_file)
            if 'newFacesName' in progress_tmp:
                progress = progress_tmp
    except IOError:
        logging.error('Do not exist today ('+PROGRESS_FILE+')')
        saveProgress()


def loadConfig():
    global config
    try:
        with open(CONFIG_FILE, 'r') as json_file:
            config_tmp = json.load(json_file)
            if 'restartAskedTime' in config_tmp:
                config = config_tmp
                config['restartAskedTime'] = datetime.datetime.fromisoformat(config['restartAskedTime'])
                # logging.debug(config)
            if 'logLevel' in config:
                logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=config['logLevel'])
            else:
                logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=10)
    except IOError:
        logging.error('Do not exist today ('+CONFIG_FILE+')')


if __name__ == "__main__":

    loadConfig()
    if (config['logProgress']):
        bar = progressbar.ProgressBar()

    # read json progression
    loadProgress()
    # logging.debug(progress)

    restartAsked = False
    counter = 0

    # Boucle tant qu'on a des modif
    while (counter == 0) or (restartAsked and counter < 100):

        startingDate = datetime.datetime.now()
        logging.info("Starting --------("+startingDate.isoformat()+")")
        setStatus('Starting')

        restartAsked = False
        counter += 1

        # get the file list
        logging.info('Get images to test')
        files = getImagesToTestFromDir(".")
        # logging.debug(files.count(object))

        if (config['logProgress']):
            bar.max_value = len(files)
        progress['countImagesTotal'] = len(files)
        progress['newFacesCount'] = 0
        progress['lastStartTime'] = datetime.datetime.now()
        cpt = 0

        if (config['sortImages']):
            files.sort()

        # --- Let's do the job (foreach image)
        logging.info("Predict...")
        for image_file in files:
            full_file_path = os.path.join(TEST_PATH, image_file)

            cpt += 1

            # ignore non image
            if not os.path.isfile(full_file_path) or os.path.splitext(full_file_path)[1][1:] not in ALLOWED_EXTENSIONS:
                # logging.debug("Ignore a file : "+full_file_path)
                continue

            # Is training needed
            if (not os.path.exists(MODEL_FILE)) or (getNewerTrainedFaceDate(TRAIN_PATH) > datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_FILE))):
                #if (os.path.exists(MODEL_FILE)):
                #    logging.debug(getNewerTrainedFaceDate(TRAIN_PATH).strftime('%c')+' > '+datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_FILE)).strftime('%c'))
                # read already found
                trainedFace = getTrainedFaces(TRAIN_PATH)

                # Train the KNN classifier and save it to disk
                # Once the model is trained and saved, you can skip this step next time.
                classifier = trainModel(TRAIN_PATH, model_save_path=MODEL_FILE, n_neighbors=2)

            # manage progress
            if (config['logProgress']):
                bar.update(cpt)
            progress['countImagesCurrent'] = cpt
            progress['nameImageCurrent'] = full_file_path
            setStatus("Recognition")
            # logging.debug("Looking for faces in {}".format(image_file))

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predict(full_file_path, model_path=MODEL_FILE)

            # logging.debug results on the console
            for name, (top, right, bottom, left) in predictions:
                # logging.debug("- Found {} at ({}, {})".format(name, left, top))

                face_key = unicodedata.normalize('NFC', ("%s (%d, %d, %d, %d)" % (image_file, left, top, right, bottom)).replace('/', '_'))

                # if already managed ignore
                if face_key in trainedFace:
                    continue

                progress['newFacesCount'] += 1
                progress['newFaces'] = face_key
                progress['newFacesName'] = name
                progress['newFacesValidated'] = True
                logging.debug(face_key+" -> "+name)

                # if unknown... add it
                if name == "unknown":

                    # get the next unknow number
                    i = 1
                    unknown_dir = os.path.join(TRAIN_PATH, "Unknown_"+str(i).zfill(6))
                    while os.path.isdir(unknown_dir):
                        i += 1
                        unknown_dir = os.path.join(TRAIN_PATH, "Unknown_"+str(i).zfill(6))
                    os.mkdir(unknown_dir)
                    # logging.debug(unknown_dir)

                    # get the face and save it
                    pil_image = Image.open(os.path.join(TEST_PATH, image_file)).convert("RGB")
                    cropped_image = pil_image.crop((left, top, right, bottom))
                    cropped_image.save(os.path.join(unknown_dir, face_key+".jpg"), "JPEG")

                else:
                    # if not already there
                    if not os.path.isfile(os.path.join(TRAIN_PATH, name, face_key+".jpg")):
                        # save it to the 'tovalidate' dir
                        # get the good directory
                        progress['newFacesValidated'] = False
                        to_be_validate_dir = os.path.join(TRAIN_PATH, name, 'tovalidate')
                        if not os.path.isdir(to_be_validate_dir):
                            os.mkdir(to_be_validate_dir)

                        saved_face_path = os.path.join(to_be_validate_dir, face_key+".jpg")
                        if not os.path.exists(saved_face_path):
                            # get the face and save it
                            pil_image = Image.open(os.path.join(TEST_PATH, image_file)).convert("RGB")
                            cropped_image = pil_image.crop((left, top, right, bottom))
                            cropped_image.save(saved_face_path, "JPEG")
                saveProgress()

            loadConfig()
            if (startingDate < config['restartAskedTime']):
                logging.info("Restart asked.("+config['restartAskedTime'].isoformat()+" > "+startingDate.isoformat()+")")
                restartAsked = True
                break
