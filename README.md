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

`groot` is a name of a little Raspberry Pi home camera that guards an important corner of the galaxy.

### Learning things

mxnet comes in two flavors: [Gluon API](https://mxnet.incubator.apache.org/api/python/gluon/gluon.html) and [Module API](https://mxnet.incubator.apache.org/api/python/module/module.html). Clojure bindings are currently based on Module API that are basd on Scala mxnet bindings that are based on [JNI bindings](https://github.com/apache/incubator-mxnet/tree/master/scala-package/native). Clojure Gluon API are currently [proposed](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=103089990).

Module API are not complex, but quite different from "traditional" deep learning expectations (i.e. PyTorch, Tensorflow, Keras, etc.) hence take time to wrap one's mind around. mxterm wraps Clojure Module API for its examples simply because _the focus_ is not on any particular API, but rather on _how and why things are done in deep learning_. Besides the source is open and there are several Clojure Module API [examples](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples) to look at plus some [great tutorials](https://arthurcaillau.com/blog/).

Let'e rock and roll:

```clojure
(require '[mxterm.nn :as nn]
         '[mxterm.infer :as infer]
```

```clojure
=> (def data-loader (nn/make-data-loader {:train-path "data/groot/groot-train.rec"
                                          :valid-path "data/groot/groot-valid.rec"}))

[21:16:47] src/io/iter_image_recordio_2.cc:172: ImageRecordIOParser2: data/groot/groot-train.rec, use 3 threads for decoding..
[21:16:51] src/io/iter_image_recordio_2.cc:172: ImageRecordIOParser2: data/groot/groot-valid.rec, use 3 threads for decoding..
#'boot.user/data-loader
```

```clojure
=> (def model (nn/make-model {:arch "model/resnet-18"
                              :data-loader data-loader}))

[21:17:42] src/nnvm/legacy_json_util.cc:209: Loading symbol saved by previous version v0.8.0. Attempting to upgrade...
[21:17:42] src/nnvm/legacy_json_util.cc:217: Symbol successfully upgraded!
#'boot.user/model
```

in case we are running on GPUs we can add context when making a model:

```clojure
=> (def model (nn/make-model {:arch "model/resnet-18"
                              :data-loader data-loader
                              :contexts (nn/run-on {:cores :gpu :cnum 1})}))
```

```clojure
=> (nn/fit model data-loader)

Epoch[0] Batch [10]	    Speed: 205.92 samples/sec	Train-accuracy=0.715909
Epoch[0] Batch [20]	    Speed: 224.40 samples/sec	Train-accuracy=0.752976
Epoch[0] Batch [30]	    Speed: 217.39 samples/sec	Train-accuracy=0.770161
Epoch[0] Batch [40]	    Speed: 216.80 samples/sec	Train-accuracy=0.775915
...      ...
Epoch[0] Batch [300]	Speed: 221.30 samples/sec	Train-accuracy=0.836171
Epoch[0] Batch [310]	Speed: 220.69 samples/sec	Train-accuracy=0.837621
Epoch[0] Batch [320]	Speed: 220.39 samples/sec	Train-accuracy=0.838396
Epoch[0] Batch [330]	Speed: 220.69 samples/sec	Train-accuracy=0.839690

Epoch[0] Train-accuracy=0.8417279
Epoch[0] Time cost=25629
Epoch[0] Validation-accuracy=0.58284885
```

`58.2%` validation accuracy is not exactly high, let's train it with some more epochs and with better metrics:

```clojure
=> (require '[org.apache.clojure-mxnet.eval-metric :as em])

=> (def metric (em/comp-metric [(em/accuracy) (em/f1)]))
#'boot.user/metric

=> (nn/fit model data-loader {:epochs 5
                              :params {:eval-metric metric}})

Epoch[0] Batch [10]	    Speed: 211.92 samples/sec	Train-accuracy=0.875000
Epoch[0] Batch [10]	    Speed: 211.92 samples/sec	Train-f1=0.920778
Epoch[0] Batch [300]	Speed: 219.48 samples/sec	Train-accuracy=0.910714
Epoch[0] Batch [300]	Speed: 219.48 samples/sec	Train-f1=0.944446
...
Epoch[4] Batch [320]	Speed: 218.28 samples/sec	Train-accuracy=0.941589
Epoch[4] Batch [320]	Speed: 218.28 samples/sec	Train-f1=0.962487
Epoch[4] Batch [330]	Speed: 217.10 samples/sec	Train-accuracy=0.942032
Epoch[4] Batch [330]	Speed: 217.10 samples/sec	Train-f1=0.962771

Epoch[4] Train-accuracy=0.94266224
Epoch[4] Time cost=24886
Epoch[4] Validation-accuracy=0.9625
```

`96.24%` is a much better validation accuracy.

and since this is an accepted accuracy for `resnet-18` to explore and play with, we can save the model:

```clojure
=> (nn/save-model model "model/groot-18")
INFO  org.apache.mxnet.module.Module: Saved checkpoint to model/shroot-18-0000.params
INFO  org.apache.mxnet.module.Module: Saved optimizer state to model/shroot-18-0000.states
#object[org.apache.mxnet.module.Module 0x599149e8 "org.apache.mxnet.module.Module@599149e8"]
```

## Predicting things

The usual goal of creating and training the model is to use it afterwards to predict things.

Let's load it from where we left off and see if it can recognize humans:

```clojure
=> (def m (nn/load-model "model/groot-18"))
=> m
#object[org.apache.mxnet.module.Module 0x4f09451c "org.apache.mxnet.module.Module@4f09451c"]
```

since the output of this model is a vector of two classes `["human",  "not-human"]`
the model will return a vector of two probabilities the first one will be
a probability that it sees human on the image, the second one a probability it does not.

```clojure
=> (infer/predict m "file:./data/groot/valid/human/04-2018.10-02-19.jpg")
;; [0.9444668 0.05553314]

=> (infer/predict m "file:./data/groot/valid/no-human/07-2018.20-38-06.jpg")
;; [0.1873892 0.8126108]

=> (infer/predict m "file:./data/groot/valid/no-human/01-2019.05-51-47.jpg")
;; [0.004020324 0.99597967]

=> (infer/predict m "file:./data/groot/valid/human/03-2018.11-17-21.jpg")
;; [0.9457831 0.054216918]
```

niice.

## License

Copyright © 2019 tolitius

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
