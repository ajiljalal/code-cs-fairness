import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.core.framework import graph_pb2
import copy
import tensorflow.contrib.graph_editor as ge
import os
from threading import Lock


def flatten_eps(eps):
    # [BS, eps_size]
    return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)


def unflatten_eps(feps):
    index = 0
    eps = []
    bs = feps.shape[0]  # feps.size // eps_size
    for shape in eps_shapes:
        eps.append(np.reshape(
            feps[:, index: index+np.prod(shape)], (bs, *shape)))
        index += np.prod(shape)
    return eps

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def_optimized, name='')
        return graph


def run(sess, fetches, feed_dict):
    lock = Lock()
    with lock:
        # Locked tensorflow so average server response time to user is lower
        result = sess.run(fetches, feed_dict)
    return result

def get_model(model_path, batch_size, z_sdev, fixed_init=False):
    assert os.path.exists(model_path), f'model_path does not exist: {model_path}'

    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def)


    inputs = {
        'dec_eps_0': 'Placeholder',
        'dec_eps_1': 'Placeholder_1',
        'dec_eps_2': 'Placeholder_2',
        'dec_eps_3': 'Placeholder_3',
        'dec_eps_4': 'Placeholder_4',
        'dec_eps_5': 'Placeholder_5',
        'enc_x': 'input/image',
        'enc_x_d': 'input/downsampled_image',
        'enc_y': 'input/label'
    }
    outputs = {
        'dec_x': 'model_1/Reshape_4',
        'enc_eps_0': 'model/pool0/truediv_1',
        'enc_eps_1': 'model/pool1/truediv_1',
        'enc_eps_2': 'model/pool2/truediv_1',
        'enc_eps_3': 'model/pool3/truediv_1',
        'enc_eps_4': 'model/pool4/truediv_1',
        'enc_eps_5': 'model/truediv_4'
    }

    eps_shapes = [(128, 128, 6), (64, 64, 12), (32, 32, 24),
                  (16, 16, 48), (8, 8, 96), (4, 4, 384)]
    eps_sizes = [np.prod(e) for e in eps_shapes]
    eps_size = 256 * 256 * 3


    dec_eps = []
    dec_eps_shapes = [(batch_size,128, 128, 6), (batch_size,64, 64, 12), (batch_size,32, 32, 24),
          (batch_size,16, 16, 48), (batch_size,8, 8, 96), (batch_size,4, 4, 384)]

    # replace the decoder placeholders with differentiable variables
    target_var_name_pairs = []
    for i in range(6):
        # name of i-th decoder placeholder
        name = 'import/' + inputs[f'dec_eps_{i}']
        var_shape = dec_eps_shapes[i]

        # Give each variable a name that doesn't already exist in the graph
        var_name = f'dec_eps_{i}_turned_var'
        # Create TensorFlow variable initialized by values of original const.
    #         var = tf.get_variable(name=var_name, dtype='float32', shape=var_shape,initializer=tf.constant_initializer(tensor_as_numpy_array))
        if not fixed_init:
            var = tf.get_variable(name=var_name, dtype='float32', shape=var_shape,initializer=tf.random_normal_initializer(stddev=z_sdev))
        else:
            init_value = np.load(f'./initializations/dec_eps_{i}_sdev_1.npy')
            init_value = init_value.repeat(batch_size, 0)
            var = tf.get_variable(name=var_name, dtype='float32', shape=var_shape,initializer=tf.constant_initializer(z_sdev * init_value))

        # We want to keep track of our variables names for later.
        target_var_name_pairs.append((name, var_name))

        # add new variable to list
        dec_eps.append(var)

    # At this point, we added a bunch of tf.Variables to the graph, but they're
    # not connected to anything.

    # The magic: we use TF Graph Editor to swap the Constant nodes' outputs with
    # the outputs of our newly created Variables.

    for const_name, var_name in target_var_name_pairs:
        const_op = tf.get_default_graph().get_operation_by_name(const_name)
        var_reader_op = tf.get_default_graph().get_operation_by_name(var_name + '/read')
        ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))


    # remove floor operations from the graph
    floor_op = tf.get_default_graph().get_operation_by_name('import/model/Floor_1')
    div_op = tf.get_default_graph().get_operation_by_name('import/model/truediv_2')

    ge.swap_outputs(ge.sgv(floor_op), ge.sgv(div_op))

    # remove random noise from encoder
    c_op = tf.get_default_graph().get_operation_by_name('import/model/random_uniform/sub/_4108__cf__4108')
    c_  = tf.constant(0, dtype=tf.float32)
    c_zero_op = tf.get_default_graph().get_operation_by_name('Const')
    ge.swap_outputs(ge.sgv(c_op), ge.sgv(c_zero_op))

    n_eps = 6


    def get(name):
        return tf.get_default_graph().get_tensor_by_name('import/' + name + ':0')

    # Encoder
    enc_x = get(inputs['enc_x'])
    enc_eps = [get(outputs['enc_eps_' + str(i)]) for i in range(n_eps)]
    enc_x_d = get(inputs['enc_x_d'])
    enc_y = get(inputs['enc_y'])

    # Decoder
    dec_x = get(outputs['dec_x'])



    def encode(img):
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        bs = img.shape[0]
        assert img.shape[1:] == (256, 256, 3)
        feed_dict = {enc_x: img}
        update_feed(feed_dict, bs)  # For unoptimized model

        return flatten_eps(run(sess, enc_eps, feed_dict))


    def decode(feps):
        if len(feps.shape) == 1:
            feps = np.expand_dims(feps, 0)
        bs = feps.shape[0]
        # assert len(eps) == n_eps
        # for i in range(n_eps):
        #     shape = (BATCH_SIZE, 128 // (2 ** i), 128 // (2 ** i), 6 * (2 ** i) * (2 ** (i == (n_eps - 1))))
        #     assert eps[i].shape == shape
        eps = unflatten_eps(feps)

        feed_dict = {}
        for i in range(n_eps):
            feed_dict[dec_eps[i]] = eps[i]
        update_feed(feed_dict, bs)  # For unoptimized model

        return run(sess, dec_x, feed_dict)

    def random(bs=1, eps_std=0.7):
        feps = np.random.normal(scale=eps_std, size=[bs, eps_size])
        return decode(feps), feps
    # function that updates the feed_dict to include a downsampled image
    # and a conditional label set to all zeros.
    def update_feed(feed_dict, bs):
        x_d = 128 * np.ones([bs, 128, 128, 3], dtype=np.uint8)
        y = np.zeros([bs], dtype=np.int32)
        feed_dict[enc_x_d] = x_d
        feed_dict[enc_y] = y
        return feed_dict

    feed_dict = {}
    update_feed(feed_dict, batch_size)
    return dec_x, dec_eps, feed_dict, run


