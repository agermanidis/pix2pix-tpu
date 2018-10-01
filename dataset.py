import os
import random
import tensorflow as tf

CROP_SIZE = 256

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def load_dataset(args, params):
    seed = random.randint(0, 2**31 - 1)

    def transform(image):
        r = image
        if args.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)
        r = tf.image.resize_images(
            r, [args.scale_size, args.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform(
            [2], 0, args.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        r = tf.image.crop_to_bounding_box(
             r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        return r

    def parser(filename):
        img_file = tf.read_file(filename)
        inp = tf.image.decode_image(img_file, channels=3)
        inp = tf.image.convert_image_dtype(inp, dtype=tf.float32)
        assertion = tf.assert_equal(
            tf.shape(inp)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            inp = tf.identity(inp)
        inp.set_shape([None, None, 3])
        width = tf.shape(inp)[1]
        a_img = preprocess(inp[:, :width // 2, :])
        b_img = preprocess(inp[:, width // 2:, :])
        if args.which_direction == "AtoB":
            input_img, target_img = a_img, b_img
        else:
            input_img, target_img = b_img, a_img
        return transform(input_img), transform(target_img)

    file_pattern = os.path.join(args.data_dir, "train/*")
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.repeat()
    dataset = dataset.map(parser)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    
    images, targets = dataset.make_one_shot_iterator().get_next()
    return images, targets
