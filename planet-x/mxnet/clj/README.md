## taking mxnet for a spin

based on from the finetune examples: [python example](https://mxnet.incubator.apache.org/faq/finetune.html) and [clojure example](https://github.com/apache/incubator-mxnet/tree/master/contrib/clojure-package/examples/pre-trained-models)

download resnet (18, 34, 50, 152, etc.) model:

```bash
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json
```

generate record files from the preclassified images:

```bash
python3 $MXNET_HOME/tools/im2rec.py --list --recursive groot-train train/
python3 $MXNET_HOME/tools/im2rec.py --list --recursive groot-valid valid/
python3 $MXNET_HOME/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 groot-train train/
python3 $MXNET_HOME/tools/im2rec.py --resize 256 --quality 90 --num-thread 16 groot-valid valid/
```
