# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import argparse
import numpy as np
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
    # add_boolean_cli_arg(parser, "infogan", default=True, help="Train GAN or InfoGAN")
    # # add_boolean_cli_arg(parser, "infogan", default=False, help="Train GAN or InfoGAN")
    # add_boolean_cli_arg(parser, "use_batch_norm", default=True, help="Use batch normalization.")
    # add_boolean_cli_arg(parser, "fix_std", default=True, help="Fix continuous var standard deviation to 1.")
    # add_boolean_cli_arg(parser, "force_grayscale", default=False, help="Convert images to single grayscale output channel.")
    return parser.parse_args()

# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# print(np.hstack([a,b]))

def create_categorical_noise(categorical_cardinality, size):
    noise = []
    for cardinality in categorical_cardinality:
        noise.append(
            np.random.randint(0, cardinality, size=size)
        )
    return noise

def make_one_hot(indices, size):
    as_one_hot = np.zeros((indices.shape[0], size))
    as_one_hot[np.arange(0, indices.shape[0]), indices] = 1.0
    return as_one_hot

def main():
    noise = create_categorical_noise(1,64)
    kk = make_one_hot(noise,1)
    print(kk)


if __name__ == '__main__':
    main()