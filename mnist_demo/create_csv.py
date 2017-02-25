# Util to convert MNIST original data files to csv files.

def convert(img_file, label_file, out_file, n):
    f = open(img_file, "rb")
    l = open(label_file, "rb")
    o = open(out_file, "w")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")

    f.close()
    l.close()
    o.close()


def main():
    # train and test data files can be downloaded from: http://yann.lecun.com/exdb/mnist/
    convert("/tmp/mnist/data/train-images-idx3-ubyte", "/tmp/mnist/data/train-labels-idx1-ubyte",
            "/tmp/mnist/data/train.csv", 60000)

    convert("/tmp/mnist/data/t10k-images-idx3-ubyte", "/tmp/mnist/data/t10k-labels-idx1-ubyte",
            "/tmp/mnist/data/test.csv", 10000)


if __name__ == "__main__":
    main()
