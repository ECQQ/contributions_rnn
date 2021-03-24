import tensorflow as tf
import os

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _parse_ft(sample, n_cls):
    feat_keys = dict() # features for record
    for k in range(300):
        feat_keys['dim_tok_{}'.format(k)] = tf.io.FixedLenSequenceFeature([],
                                                        dtype=tf.float32,
                                                        allow_missing=True)

    feat_keys['text'] = tf.io.FixedLenSequenceFeature([],
                                                    dtype=tf.string,
                                                    allow_missing=True)
    feat_keys['category'] = tf.io.FixedLenSequenceFeature([],
                                                    dtype=tf.string,
                                                    allow_missing=True)
    feat_keys['label'] = tf.io.FixedLenFeature([], tf.int64)

    feat_keys['length'] = tf.io.FixedLenFeature([], tf.int64)
    ex1 = tf.io.parse_example(sample, feat_keys)

    examples = [ex1['dim_tok_{}'.format(k)] for k in range(300)]
    inputs = tf.concat(examples, 0)
    inputs = tf.reshape(inputs, [tf.cast(ex1['length'], tf.int32), 300])
    return inputs, tf.cast(ex1['length'], tf.int32), tf.one_hot(ex1['label'], n_cls), ex1['text'], ex1['category']

def _parse(sample, n_cls):
    feat_keys = dict() # features for record
    feat_keys['input'] = tf.io.FixedLenSequenceFeature([],
                                                    dtype=tf.float32,
                                                    allow_missing=True)
    feat_keys['text'] = tf.io.FixedLenSequenceFeature([],
                                                    dtype=tf.string,
                                                    allow_missing=True)
    feat_keys['category'] = tf.io.FixedLenSequenceFeature([],
                                                    dtype=tf.string,
                                                    allow_missing=True)
    feat_keys['label'] = tf.io.FixedLenFeature([], tf.int64)

    feat_keys['length'] = tf.io.FixedLenFeature([], tf.int64)
    ex1 = tf.io.parse_example(sample, feat_keys)

    inputs = tf.expand_dims(ex1['input'], 1)
    return inputs, tf.cast(ex1['length'], tf.int32), tf.one_hot(ex1['label'], n_cls), ex1['text'], ex1['category']

def load_records(source, batch_size, return_cls=False, get_all=False, n_cls=13):

    if get_all:
        datasets = [os.path.join(source, x) for x in os.listdir(source)]
        dataset  = tf.data.TFRecordDataset(datasets)
        dataset  = dataset.map(lambda x: _parse_ft(x, n_cls), num_parallel_calls=8)
        dataset  = dataset.padded_batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if return_cls:
            return dataset, n_cls
        return dataset
    else:
        datasets = [tf.data.TFRecordDataset(os.path.join(source, x)) for x in os.listdir(source)]
        n_cls = len(datasets)

        datasets = [
            dataset.map(
                lambda x: _parse_ft(x, n_cls), num_parallel_calls=8) for dataset in datasets
        ]

        datasets = [dataset.repeat() for dataset in datasets]
        datasets = [dataset.shuffle(5000, reshuffle_each_iteration=True) for dataset in datasets]
        dataset  = tf.data.experimental.sample_from_datasets(datasets)
        dataset  = dataset.padded_batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if return_cls:
            return dataset, n_cls
        return dataset
