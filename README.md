# taking [mxnet](https://mxnet.incubator.apache.org/) for a spin

`doing` is the most efficient way to understand theory behind anything

based on a [clojure mxnet API](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package)
mxterm attempts to bring REPL exploratory feel to, at times, pretty science heavy and complex corners of deep learning.

## Transfer Learning

transfer learning is twofold:

### fold one: "train the beast"

* someone has lots of GPUs
* someone takes a lot of data
* someone creates or chooses an architecture
  - i.e. the shape of neural network: what layers to use, how many of them to use, how layers are connected to each other, etc.
* someone trains this architecture with that data on all those fancy GPUs
* someone saves these learned parameters for this architecure
* someone calls this saved parameters a `model` and lets others to use this model

examples of such architectures:

* image classification: [ResNet](https://arxiv.org/abs/1512.03385)
* natural language processing: [Wikitext 103](https://arxiv.org/abs/1801.06146)

### fold two: "train the beauty"

* you take that "pretrained" model
* you replace its "head"
  - by adding your layer as the last layer of that pretrained model
    (i.e. model was trained to understand English, but you need to train it to write poems: so you are going to train this last layer with poems)
* you train this model with your new data (poems, your classified images, etc.)

hence the new model would be well tuned to your data by relying on all that wisdom from the pretrained model + some your additional training.

## Train a custom image classifier

this is going to be close, but not exactly, to an mxnet "[Fine-tune with Pretrained Models](https://mxnet.incubator.apache.org/versions/master/faq/finetune.html)" example.

instead of traning on [Caltech256](https://authors.library.caltech.edu/7694/) dataset, we'll take a custom dataset of images with two categories: "human" and "no human". This is an example (I use these for my home Raspberry Pi camera to detect package deliveres, strangers, etc..), you can come up with different images and different categories of course.

### Encode images into RecordIO

before plugging these images into mxnet, they need to be converted to RecordIO format which is done by [dev-resources/groot/make-datasets.sh](dev-resources/groot/make-datasets.sh) script that uses mxnet's `im2rec.py` that encodes all the images with their labels into `.rec` files. [Here](https://mxnet.incubator.apache.org/versions/master/architecture/note_data_loading.html) is more on why and how.

### Download a pretrained model

Since there is already a few great options for pretrained models in the space of image recognition we'll train our model on top of "ResNet". We'll take an 18 layer pretrained model:

```bash
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params -P model
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json -P model
```

the reason it is 18 and not 50 or 152 layers, which are also available, is that the idea is to iterate quickly and see what works and what does not. 18 layer model is more shallow and takes a lot less time to train and experiment with. Once the right approach and hyper parameters are figured out, you can download a 152 layer ResNet and train on top of that.

here is what `./data` and `./model` look like now:

```bash
$ tree data
data
└── groot
    ├── groot-train.idx
    ├── groot-train.lst
    ├── groot-train.rec
    ├── groot-valid.idx
    ├── groot-valid.lst
    └── groot-valid.rec
```

```bash
$ tree model
model
├── resnet-18-0000.params
└── resnet-18-symbol.json
```

`groot` is my home camera name: Groot is guarding my home galaxy.

### Learning things

mxnet comes in two flavors: [Gluon API](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html) and [Module API](https://mxnet.incubator.apache.org/api/python/module/module.html). Clojure bindings are currently based on Module API that are basd on Scala mxnet bindings that are based on [JNI bindings](https://github.com/apache/incubator-mxnet/tree/master/scala-package/native). Clojure Gluon API are currently [proposed](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=103089990).

Module API are not complex, but quite different from "traditional" deep learning expectations (i.e. PyTorch, Tensorflow, Keras, etc.) hence take time to wrap one's mind around. mxterm wraps Clojure Module API for its examples simply because _the focus_ is not on any particular API, but rather on _how and why things are done in deep learning_. Besides the source is open and there are several Clojure Module API [examples](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples) to look at.

Let'e rock and roll:

```clojure
(require '[groot.nn :as nn]
         '[groot.infer :as infer]
         '[org.apache.clojure-mxnet.module :as mm])
```

## License

Copyright © 2019 tolitius

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
