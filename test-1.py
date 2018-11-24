import os
import sys
import tensorflow as tf

from InceptionModel.inception_utils import load_model, load_labels_vocabulary, make_predictions_and_gradients, top_label_id_and_score
from IntegratedGradients.integrated_gradients import integrated_gradients, random_baseline_integrated_gradients
from VisualizationLibrary.visualization_lib import Visualize, show_pil_image, pil_image


MODEL_LOC='./InceptionModel/tensorflow_inception_graph.pb'
LABELS_LOC='./InceptionModel/imagenet_comp_graph_label_strings.txt'

# Load the Inception model.
sess, graph = load_model(MODEL_LOC)

# Load the Labels vocabulary.
labels = load_labels_vocabulary(LABELS_LOC)

# Make the predictions_and_gradients function
inception_predictions_and_gradients = make_predictions_and_gradients(sess, graph)



def load_image(img_path):
    # "/Users/pin-jutien/Integrated-Gradients/Images/70bfca4555cca92e.jpg"
    image = tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    img = sess.run(image)
    return img


img = load_image("/Users/pin-jutien/Integrated-Gradients/Images/70bfca4555cca92e.jpg")

# Determine top label and score.
top_label_id, score = top_label_id_and_score(img, inception_predictions_and_gradients)
print("Top label: %s, score: %f" % (labels[top_label_id], score))

# Compute attributions based on just the gradients.
_, gradients = inception_predictions_and_gradients([img], top_label_id)

attributions = random_baseline_integrated_gradients(
    img,
    top_label_id,
    inception_predictions_and_gradients,
    steps=50,
    num_random_trials=10)

Visualize(
    attributions, img,
    clip_above_percentile=95,
    clip_below_percentile=58,
    morphological_cleanup=True,
    outlines=True,
    overlay=True)
