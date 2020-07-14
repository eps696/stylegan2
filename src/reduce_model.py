import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow; tf = tensorflow.compat.v1 if hasattr(tensorflow.compat,'v1') else tensorflow

import pickle

flags = tf.flags
flags.DEFINE_string('model', None, 'Full model path')
flags.DEFINE_string('out_dir', './', 'Output directory for reduced model')
a = flags.FLAGS

def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

tf.InteractiveSession()

with open(a.model, 'rb') as file:
    _, _, Gs = pickle.load(file, encoding='latin1')
    save_pkl((Gs), os.path.join(a.out_dir, os.path.basename(a.model[:-4]) + "-Gs.pkl"))
    # G,D = generator and discriminator snapshots to resume training 
    # Gs = Long-term average of the generator, yielding higher-quality results than snapshots
