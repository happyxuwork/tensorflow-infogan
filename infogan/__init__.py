import argparse

from os.path import join, realpath, dirname, basename

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

# import sys
# sys.path.append('infogan')
from infogan.categorical_grid_plots import CategoricalPlotter


# from .categorical_grid_plots import categorical_grid_plots
# from .categorical_grid_plots import categorical_grid_plots import CategoricalPlotter
# import categorical_grid_plots.CategoricalPlotter
from infogan.tf_utils import (
    scope_variables,
    NOOP,
    load_mnist_dataset,
    run_network,
    leaky_rectify,
)
from infogan.misc_utils import (
    next_unused_name,
    add_boolean_cli_arg,
    create_progress_bar,
    load_image_dataset,
)
from infogan.noise_utils import (
    create_infogan_noise_sample,
    create_gan_noise_sample,
)

SCRIPT_DIR = dirname(realpath(__file__))
PROJECT_DIR = dirname(SCRIPT_DIR)
TINY = 1e-6
# z为噪声的维度，如果单纯是GAN的话，为62，如果是InfoGAN的话，为74。
def generator_forward(z,
                      network_description,
                      is_training,
                      reuse=None,
                      name="generator",
                      use_batch_norm=True,
                      debug=False):
    with tf.variable_scope(name, reuse=reuse):
        return run_network(z,
                           network_description,
                           is_training=is_training,
                           use_batch_norm=use_batch_norm,
                           debug=debug,
                           strip_batchnorm_from_last_layer=True)

# 返回两个参数：一个是判断图像为真的概率==prob，一个是输出前一层的隐藏层===hidden
def discriminator_forward(img,
                          network_description,
                          is_training,
                          reuse=None,
                          name="discriminator",
                          use_batch_norm=True,
                          debug=False):
    with tf.variable_scope(name, reuse=reuse):
        # fake_images==[64,28,28,1]
        # discriminator_desc==conv:4:2:64:lrelu,conv:4:2:128:lrelu,fc:1024:lrelu
        # is_training == True
        out = run_network(img,
                          network_description,
                          is_training=is_training,
                          use_batch_norm=use_batch_norm,
                          debug=debug)
        out = layers.flatten(out)
        prob = layers.fully_connected(
            out,
            num_outputs=1,
            activation_fn=tf.nn.sigmoid,
            scope="prob_projection"
        )
    # out的输出为[64,1024]
    return {"prob":prob, "hidden":out}

# true_categoricals==[64,10]# 实质就是如果是使用InfoGAN则将前10列取出，行保留所有==这个就是类别latend code
# true_continuous== [64,2]# 将随机噪声中的11-12列取出，行保留所有===这个就是符合均匀分布[-1,1]的连续的latend code
# categorical_lambda==1
# continuous_lambda==1
# fix_std==Fix continuous var standard deviation to 1
# hidden的输出为[64,1024]
def reconstruct_mutual_info(true_categoricals,
                            true_continuous,
                            categorical_lambda,
                            continuous_lambda,
                            fix_std,
                            hidden,
                            is_training,
                            reuse=None,
                            name="mutual_info"):
    with tf.variable_scope(name, reuse=reuse):
        # hidden的输出为[64,1024]
        # 下面out的输出为[64,128]
        out = layers.fully_connected(
            hidden,
            num_outputs=128,
            activation_fn=leaky_rectify,
            normalizer_fn=layers.batch_norm,
            normalizer_params={"is_training":is_training}
        )
        # num_categorical=10
        # num_continuous = 2
        num_categorical = sum([true_categorical.get_shape()[1].value for true_categorical in true_categoricals])
        num_continuous = true_continuous.get_shape()[1].value

        # tf.identity是返回了一个一模一样新的tensor，再control_dependencies的作用块下，需要增加一个新节点到gragh中
        # out输出[64,12]
        out = layers.fully_connected(
            out,
            num_outputs=num_categorical + (num_continuous if fix_std else (num_continuous * 2)),
            activation_fn=tf.identity
        )

        # distribution logic
        offset = 0
        ll_categorical = None
        for true_categorical in true_categoricals:
            # cardinality=10
            cardinality = true_categorical.get_shape()[1].value
            # 得到前10个的预测
            prob_categorical = tf.nn.softmax(out[:, offset:offset + cardinality])
            # 关于categorical latent code的目标函数为G(z,c)对应于categorical的输出的softmax与categorical latent code的交叉熵。
            ll_categorical_new = tf.reduce_sum(tf.log(prob_categorical + TINY) * true_categorical,
                reduction_indices=1
            )
            if ll_categorical is None:
                ll_categorical = ll_categorical_new
            else:
                ll_categorical = ll_categorical + ll_categorical_new
            offset += cardinality

        # 关于continuous latent code的目标函数，将continuous latent code以均值为G(z,c)对应于continuous的输出，
        # 方差为1进行标准化，然后计算它以正态分布的概率密度作为目标函数。
        mean_contig = out[:, num_categorical:num_categorical + num_continuous]

        if fix_std:
            std_contig = tf.ones_like(mean_contig)
        else:
            std_contig = tf.sqrt(tf.exp(out[:, num_categorical + num_continuous:num_categorical + num_continuous * 2]))

        epsilon = (true_continuous - mean_contig) / (std_contig + TINY)
        ll_continuous = tf.reduce_sum(
            - 0.5 * np.log(2 * np.pi) - tf.log(std_contig + TINY) - 0.5 * tf.square(epsilon),
            reduction_indices=1,
        )
        if ll_categorical is None:
            ll_categorical = tf.constant(0.0, dtype=tf.float32)
        mutual_info_lb = continuous_lambda * ll_continuous + categorical_lambda * ll_categorical
    return {
        "mutual_info": tf.reduce_mean(mutual_info_lb),
        "ll_categorical": tf.reduce_mean(ll_categorical),
        "ll_continuous": tf.reduce_mean(ll_continuous),
        "std_contig": tf.reduce_mean(std_contig)
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--scale_dataset", type=int, nargs=2, default=[28,28])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--generator_lr", type=float, default=1e-3)
    parser.add_argument("--discriminator_lr", type=float, default=2e-4)
    parser.add_argument("--categorical_lambda", type=float, default=1.0)
    parser.add_argument("--continuous_lambda", type=float, default=1.0)

    parser.add_argument("--categorical_cardinality", nargs="*", type=int, default=[10],
                        help="Cardinality of the categorical variables used in the generator.")
    parser.add_argument("--generator",
                        type=str,
                        default="fc:1024,fc:7x7x128,reshape:7:7:128,deconv:4:2:64,deconv:4:2:1:sigmoid",
                        help="Generator network architecture (call tech support).")
    parser.add_argument("--discriminator",
                        type=str,
                        default="conv:4:2:64:lrelu,conv:4:2:128:lrelu,fc:1024:lrelu",
                        help="Discriminator network architecture, except last layer (call tech support).")
    parser.add_argument("--num_continuous", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--style_size", type=int, default=62)
    parser.add_argument("--plot_every", type=int, default=200,
                        help="How often should plots be made (note: slow + costly).")
    add_boolean_cli_arg(parser, "infogan", default=True, help="Train GAN or InfoGAN")
    # add_boolean_cli_arg(parser, "infogan", default=False, help="Train GAN or InfoGAN")
    add_boolean_cli_arg(parser, "use_batch_norm", default=True, help="Use batch normalization.")
    # add_boolean_cli_arg(parser, "use_batch_norm", default=False, help="Use batch normalization.")
    add_boolean_cli_arg(parser, "fix_std", default=True, help="Fix continuous var standard deviation to 1.")
    add_boolean_cli_arg(parser, "force_grayscale", default=False, help="Convert images to single grayscale output channel.")
    return parser.parse_args()


def train():
    args = parse_args()

    # 从1234中随机取值
    np.random.seed(args.seed)

    #批大小为64
    batch_size = args.batch_size
    n_epochs = args.epochs
    use_batch_norm = args.use_batch_norm
    fix_std = args.fix_std
    plot_every = args.plot_every
    use_infogan = args.infogan

    # 大小为62
    style_size = args.style_size

    # [10]
    categorical_cardinality = args.categorical_cardinality

    # 大小为2
    num_continuous = args.num_continuous

    # "fc:1024,fc:7x7x128,reshape:7:7:128,deconv:4:2:64,deconv:4:2:1:sigmoid"
    generator_desc = args.generator

    # "conv:4:2:64:lrelu,conv:4:2:128:lrelu,fc:1024:lrelu"不包括最后一层
    discriminator_desc = args.discriminator

    # 在开发一个程序时候，与其让它运行时崩溃，不如在它出现错误条件时就崩溃（返回错误）。这时候断言assert
    # 就显得非常有用。

    #如果没有指定训练的数据集，默认使用mnist数据集
    if args.dataset is None:
        assert args.scale_dataset == [28, 28]
        X = load_mnist_dataset()
        if args.max_images is not None:
            X = X[:args.max_images]
        dataset_name = "mnist"

    #如果指定了训练的数据集，就一定要指定图像的大小
    else:
        scaled_image_width, scaled_image_height = args.scale_dataset
        # load pngs and jpegs here
        # 加载特定数据库，特定大小，特定数量，并且归一化的4维数组（数量，宽度，高度，通道数）
        # 这里加载图像的方式值得借鉴
        X = load_image_dataset(
            args.dataset,
            desired_width=scaled_image_width, # TODO(jonathan): pick up from generator or add a command line arg (either or)...
            desired_height=scaled_image_height,
            value_range=(0.0, 1.0),
            max_images=args.max_images,
            force_grayscale=args.force_grayscale
        )
        dataset_name = basename(args.dataset.rstrip("/"))



    if use_infogan:
        # z_size=74==如果使用infoGAN，这输入噪音的中会增加其它的latend code(连续与离散)
        # 其中style_size为62，categorical_cardinality为[10]，num_continuous为2
        z_size = style_size + sum(categorical_cardinality) + num_continuous
        sample_noise = create_infogan_noise_sample(
            categorical_cardinality,
            num_continuous,
            style_size
        )
        # sample_noise.shape=[64 74]===其中这74列中，有62列符合标准正态分布，2列符合-1到1的均匀分布，10列符合类别分布
    else:
        # 否则就进行[64,62]维度的二维列表的生成----gan的随机噪声
        z_size = style_size
        sample_noise = create_gan_noise_sample(style_size)

    # parser.add_argument("--generator_lr", type=float, default=1e-3)
    # parser.add_argument("--discriminator_lr", type=float, default=2e-4)
    discriminator_lr = tf.get_variable(
        "discriminator_lr", (),
        initializer=tf.constant_initializer(args.discriminator_lr)
    )
    generator_lr = tf.get_variable(
        "generator_lr", (),
        initializer=tf.constant_initializer(args.generator_lr)
    )

    n_images, image_height, image_width, n_channels = X.shape

    discriminator_lr_placeholder = tf.placeholder(tf.float32, (), name="discriminator_lr")
    generator_lr_placeholder = tf.placeholder(tf.float32, (), name="generator_lr")
    assign_discriminator_lr_op = discriminator_lr.assign(discriminator_lr_placeholder)
    assign_generator_lr_op = generator_lr.assign(generator_lr_placeholder)

    ## begin model
    true_images = tf.placeholder(
        tf.float32,
        [None, image_height, image_width, n_channels],
        name="true_images"
    )
    zc_vectors = tf.placeholder(
        tf.float32,
        [None, z_size],
        name="zc_vectors"
    )
    # 是否训练鉴别器
    is_training_discriminator = tf.placeholder(
        tf.bool,
        [],
        name="is_training_discriminator"
    )
    # 是否训练生成器
    is_training_generator = tf.placeholder(
        tf.bool,
        [],
        name="is_training_generator"
    )

    # tf.get_variable_scope()==用于得到当前的scope
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        # 从该生成网络中得到的是[64,28,28,1]
        fake_images = generator_forward(
            zc_vectors,
            generator_desc,
            is_training=is_training_generator,
            name="generator",
            debug=True
        )

    print("Generator produced images of shape %s" % (fake_images.get_shape()[1:]))
    print("")

    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        # fake_images==[64,28,28,1]
        # discriminator_desc==conv:4:2:64:lrelu,conv:4:2:128:lrelu,fc:1024:lrelu
        # is_training == True
        discriminator_fake = discriminator_forward(
            fake_images,
            discriminator_desc,
            is_training=is_training_discriminator,
            name="discriminator",
            use_batch_norm=use_batch_norm,
            debug=True
        )
    # return {"prob":prob, "hidden":out}
    prob_fake = discriminator_fake["prob"]

    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        discriminator_true = discriminator_forward(
            true_images,
            discriminator_desc,
            is_training=is_training_discriminator,
            reuse=True,
            name="discriminator",
            use_batch_norm=use_batch_norm
        )
    prob_true = discriminator_true["prob"]

    # discriminator should maximize:即鉴别器需要能很好地区别真的样本和假的样本
    ll_believing_fake_images_are_fake = tf.log(1.0 - prob_fake + TINY)
    ll_true_images = tf.log(prob_true + TINY)
    # 这里使用tf.reduce_mean的原因很简单===出来的可不是一张图像，而是一个批次的图像
    discriminator_obj = (
        tf.reduce_mean(ll_believing_fake_images_are_fake) +
        tf.reduce_mean(ll_true_images)
    )

    # generator should maximize:即生成器希望产生的图像通过鉴别器判别得到的为真的概率越高越好
    ll_believing_fake_images_are_real = tf.reduce_mean(tf.log(prob_fake + TINY))
    generator_obj = ll_believing_fake_images_are_real

    discriminator_solver = tf.train.AdamOptimizer(
        learning_rate=discriminator_lr,
        beta1=0.5
    )
    generator_solver = tf.train.AdamOptimizer(
        learning_rate=generator_lr,
        beta1=0.5
    )

    discriminator_variables = scope_variables("discriminator")
    generator_variables = scope_variables("generator")

    train_discriminator = discriminator_solver.minimize(-discriminator_obj, var_list=discriminator_variables)
    train_generator = generator_solver.minimize(-generator_obj, var_list=generator_variables)
    discriminator_obj_summary = tf.summary.scalar("discriminator_objective", discriminator_obj)
    generator_obj_summary = tf.summary.scalar("generator_objective", generator_obj)

    if use_infogan:
        categorical_c_vectors = []
        offset = 0
        for cardinality in categorical_cardinality:
            # 实质就是如果是使用InfoGAN则将前10列取出，行保留所有==这个就是类别latend code
            categorical_c_vectors.append(
                zc_vectors[:, offset:offset+cardinality]
            )
            offset += cardinality
        # 将随机噪声中的11-12列取出，行保留所有===这个就是符合均匀分布[-1,1]的连续的latend code
        continuous_c_vector = zc_vectors[:, offset:offset + num_continuous]

        # 互信息函数
        q_output = reconstruct_mutual_info(
            categorical_c_vectors,
            continuous_c_vector,
            categorical_lambda=args.categorical_lambda,
            continuous_lambda=args.continuous_lambda,
            fix_std=fix_std,
            hidden=discriminator_fake["hidden"],
            is_training=is_training_discriminator,
            name="mutual_info"
        )

        mutual_info_objective = q_output["mutual_info"]
        mutual_info_variables = scope_variables("mutual_info")
        neg_mutual_info_objective = -mutual_info_objective
        train_mutual_info = generator_solver.minimize(
            neg_mutual_info_objective,
            var_list=generator_variables + discriminator_variables + mutual_info_variables
        )
        ll_categorical = q_output["ll_categorical"]
        ll_continuous = q_output["ll_continuous"]
        std_contig = q_output["std_contig"]

        mutual_info_obj_summary = tf.summary.scalar("mutual_info_objective", mutual_info_objective)
        ll_categorical_obj_summary = tf.summary.scalar("ll_categorical_objective", ll_categorical)
        ll_continuous_obj_summary = tf.summary.scalar("ll_continuous_objective", ll_continuous)
        std_contig_summary = tf.summary.scalar("std_contig", std_contig)
        generator_obj_summary = tf.summary.merge([
            generator_obj_summary,
            mutual_info_obj_summary,
            ll_categorical_obj_summary,
            ll_continuous_obj_summary,
            std_contig_summary
        ])
    else:
        neg_mutual_info_objective = NOOP
        mutual_info_objective = NOOP
        train_mutual_info = NOOP
        ll_categorical = NOOP
        ll_continuous = NOOP
        std_contig = NOOP
        entropy = NOOP


    log_dir = next_unused_name(
        join(
            PROJECT_DIR,
            "%s_log" % (dataset_name,),
            "infogan" if use_infogan else "gan"
        )
    )
    journalist = tf.summary.FileWriter(
        log_dir,
        flush_secs=10
    )
    print("Saving tensorboard logs to %r" % (log_dir,))

    img_summaries = {}
    if use_infogan:
        plotter = CategoricalPlotter(
            categorical_cardinality=categorical_cardinality,
            num_continuous=num_continuous,
            style_size=style_size,
            journalist=journalist,
            generate=lambda sess, x: sess.run(
                fake_images,
                {zc_vectors:x, is_training_discriminator:False, is_training_generator:False}
            )
        )
    else:
        image_placeholder = None
        plotter = None
        # img_summaries["fake_images"] = tf.summary.image("fake images", fake_images, max_images=10)
        img_summaries["fake_images"] = tf.summary.image("fake images", fake_images, max_outputs=10)
    image_summary_op = tf.summary.merge(list(img_summaries.values())) if len(img_summaries) else NOOP

    idxes = np.arange(n_images, dtype=np.int32)
    iters = 0
    with tf.Session() as sess:
        # pleasure
        sess.run(tf.global_variables_initializer())
        # content

        for epoch in range(n_epochs):
            disc_epoch_obj = []
            gen_epoch_obj = []
            infogan_epoch_obj = []

            np.random.shuffle(idxes)
            pbar = create_progress_bar("epoch %d >> " % (epoch,))


            for idx in pbar(range(0, n_images, batch_size)):
                batch = X[idxes[idx:idx + batch_size]]
                # train discriminator
                noise = sample_noise(batch_size)
                _, summary_result1, disc_obj, infogan_obj = sess.run(
                    [train_discriminator, discriminator_obj_summary, discriminator_obj, neg_mutual_info_objective],
                    feed_dict={
                        true_images:batch,
                        zc_vectors:noise,
                        is_training_discriminator:True,
                        is_training_generator:True
                    }
                )

                disc_epoch_obj.append(disc_obj)

                if use_infogan:
                    infogan_epoch_obj.append(infogan_obj)

                # train generator
                noise = sample_noise(batch_size)
                _, _, summary_result2, gen_obj, infogan_obj = sess.run(
                    [train_generator, train_mutual_info, generator_obj_summary, generator_obj, neg_mutual_info_objective],
                    feed_dict={
                        zc_vectors:noise,
                        is_training_discriminator:True,
                        is_training_generator:True
                    }
                )

                journalist.add_summary(summary_result1, iters)
                journalist.add_summary(summary_result2, iters)
                journalist.flush()
                gen_epoch_obj.append(gen_obj)

                if use_infogan:
                    infogan_epoch_obj.append(infogan_obj)

                iters += 1

                if iters % plot_every == 0:
                    if use_infogan:
                        plotter.generate_images(sess, 10, iteration=iters)
                    else:
                        noise = sample_noise(batch_size)
                        current_summary = sess.run(
                            image_summary_op,
                            {
                                zc_vectors:noise,
                                is_training_discriminator:False,
                                is_training_generator:False
                            }
                        )
                        journalist.add_summary(current_summary, iters)
                    journalist.flush()

            msg = "epoch %d >> discriminator LL %.2f (lr=%.6f), generator LL %.2f (lr=%.6f)" % (
                epoch,
                np.mean(disc_epoch_obj), sess.run(discriminator_lr),
                np.mean(gen_epoch_obj), sess.run(generator_lr)
            )
            if use_infogan:
                msg = msg + ", infogan loss %.2f" % (np.mean(infogan_epoch_obj),)
            print(msg)

if __name__ == "__main__":
    train()