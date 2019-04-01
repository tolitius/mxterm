(ns mxterm.nn
  (:require [clojure.string :as string]
            [org.apache.clojure-mxnet.visualization :as viz]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]))

(defn preload-model

  "load a pretrained model/architecture to use for custom predictions"

  [path]
  (let [model (m/load-checkpoint {:prefix path
                                  :epoch 0})]
    {:model (m/symbol model)
     :arg-params (m/arg-params model)
     :aux-params (m/aux-params model)}))

(defn add-head

  "add a head layer (to learn) to the pretrained model

   model:        the pretrained model
   params:       parameters of the pretrained model
   class-count:  the number of classes to predict
   layer-name:   the layer name to cut the model at"

  [{:keys [model params class-count layer-name]
    :or {layer-name "flatten0"}}]
  (let [all-layers (sym/get-internals model)
        layers (sym/get all-layers (str layer-name "_output"))]
    {:net (as-> layers data
                (sym/fully-connected "fc1" {:data data
                                            :num-hidden class-count})
                (sym/softmax-output "softmax" {:data data}))
     :new-args   (->> params
                      (remove (fn [[k v]] (string/includes? k "fc1")))
                      (into {}))}))

(defn make-model

  "takes a pretrained model and adds a custom (untrained) layer to it.
   returned model is ready to be trained (fit'ted) on a new data
   that is provided with a data loader"

  [{:keys [arch
           data-loader
           class-count
           contexts]
    :or {contexts [(context/cpu 1)]                                 ;; running on cpu by default
         class-count 2}}]                                           ;; binary cross entropy by default
  (let [{:keys [arg-params] :as trained-model} (preload-model arch)
        {:keys [net new-args]} (add-head (assoc trained-model
                                                :class-count class-count))]
    (-> (m/module net {:contexts contexts})
        (m/bind {:data-shapes (mx-io/provide-data-desc (:train data-loader))
                 :label-shapes (mx-io/provide-label-desc (:valid data-loader))})
        (m/init-params {:arg-params new-args
                        :aux-params arg-params
                        :allow-missing true}))))

(defn make-train

  "makes a RecordIO iterator over the set of train images with labels
   the trickies part here is to pack all the images into RecordIO format

   more about the format:  https://mxnet.incubator.apache.org/versions/master/architecture/note_data_loading.html
   script that does it:    dev-resources/groot/make-datasets.sh"

  [{:keys [rec-path batch-size]}]

  (mx-io/image-record-iter
    {:path-imgrec rec-path
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape [3 224 224]
     :shuffle true
     :rand-crop true
     :rand-mirror true}))

(defn make-valid

  "makes a RecordIO iterator over the set of validation images with labels
   the trickies part here is to pack all the images into RecordIO format

   more about the format:  https://mxnet.incubator.apache.org/versions/master/architecture/note_data_loading.html
   script that does it:    dev-resources/groot/make-datasets.sh"

  [{:keys [rec-path batch-size]}]
  (mx-io/image-record-iter
    {:path-imgrec rec-path
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape [3 224 224]
     :rand-crop false
     :rand-mirror false}))

(defn make-data-loader

  "given a path to train and validation data makes a data loader
   that in this case consists out of two mxnet iters: :train and :valid

    => (def data-loader (nn/make-data-loader {:train-path \"data/groot/groot-train.rec\"
                                              :valid-path \"data/groot/groot-valid.rec\"}))"

  [{:keys [train-path
           valid-path
           batch-size]
    :or {batch-size 16}}]

  {:train (make-train {:rec-path train-path
                       :batch-size batch-size})
   :valid (make-valid {:rec-path valid-path
                       :batch-size batch-size})})

(defn fit

  "given the model data loader and fit params trains the model
   mxnet fit is good enough already,
   this function is based on good defaults
   and allows for better REPLing / expoloring:

     => (nn/fit model data-loader)

   or

     => (nn/fit model data-loader {:epochs 1})

   or with some params

     => (def metric (em/comp-metric [(em/accuracy) (em/f1)]))

     => (nn/fit model data-loader {:epochs 1 :params {:eval-metric metric}})"

  ([model data-loader]
   (fit model data-loader {}))
  ([model data-loader {:keys [epochs
                              batch-size
                              params]
                       :or {epochs 1
                            batch-size 16
                            params {}}}]
   (m/fit model
          {:train-data (:train data-loader)
           :eval-data (:valid data-loader)
           :num-epoch epochs
           :fit-params (m/fit-params (merge {:intializer (init/xavier {:rand-type "gaussian"
                                                                       :factor-type "in"
                                                                       :magnitude 2})
                                             :batch-end-callback (callback/speedometer
                                                                   batch-size
                                                                   10)}
                                            params))})))

(defn save-model

  "saves model at a given epoch
   saved model could be loaded later
   for predictions or to continue the training"

  [model path & {:keys [epoch]
                 :or {epoch 0}}]
  (m/save-checkpoint model {:prefix path
                            :epoch epoch
                            :save-opt-states true}) )

(defn load-model

  "loads model that was saved at a given epoch"

  [path & {:keys [epoch]
           :or {epoch 0}}]
  (m/load-checkpoint {:prefix path
                      :epoch epoch}) )

(defn run-on

  "given the kind of cores and their number
   allows to switch between CPU and GPU context
   that models will be run with"

  [{:keys [cores cnum]}]
  (if (= :gpu cores)
    (mapv context/gpu (range cnum))
    (mapv context/cpu (range cnum))))

(defn draw

  "creates a PDF with a prety model architecture

    => (nn/draw {:model m :title \"resnet-18\"})"

  [{:keys [model path data title]
             :or {data [1 3 224 224]
                  path "./"
                  title "groot"}}]
  (let [plot (viz/plot-network
               (m/symbol model)
               {"data" data}
               {:title title
                :node-attrs {:shape "oval"
                             :fixedsize "false"}})]
    (viz/render plot title path)))

;; how to REPL/use it:

(comment

  (require '[mxterm.nn :as nn])

  (def data-loader (nn/make-data-loader {:train-path "data/groot/groot-train.rec"
                                         :valid-path "data/groot/groot-valid.rec"}))

  ;; runnint on CPU (default)
  (def model (nn/make-model {:arch "model/resnet-18"
                             :data-loader data-loader}))

  ;; runnint on GPU
  (def model (nn/make-model {:arch "model/resnet-18"
                             :data-loader data-loader
                             :contexts (nn/run-on {:cores :gpu
                                                   :cnum 1})}))

  (nn/fit model data-loader)

  ;; or with some params
  (def metric (em/comp-metric [(em/accuracy) (em/f1)]))

  (nn/fit model data-loader {:epochs 5 :params {:eval-metric metric}})

  (nn/save-model model "model/groot")

)
