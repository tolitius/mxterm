(ns groot.nn
  (:require [clojure.string :as string]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.symbol :as sym]))

(defn preload-model [path]
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
   layer-name:   the layer name to add"

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

(defn make-model [{:keys [arch
                          data-loader
                          class-count
                          contexts]
                   :or {contexts [(context/cpu 1)]                ;; running on cpu by default
                        class-count 2}}]                          ;; binary cross entropy by default
  (let [{:keys [arg-params] :as trained-model} (preload-model arch)
        {:keys [net new-args]} (add-head (assoc trained-model
                                                :class-count class-count))]
    (-> (m/module net {:contexts contexts})
        (m/bind {:data-shapes (mx-io/provide-data-desc (:train data-loader))
                 :label-shapes (mx-io/provide-label-desc (:valid data-loader))})
        (m/init-params {:arg-params new-args
                        :aux-params arg-params
                        :allow-missing true}))))

(defn make-train [{:keys [rec-path batch-size]}]
  (mx-io/image-record-iter
    {:path-imgrec rec-path
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape [3 224 224]
     :shuffle true
     :rand-crop true
     :rand-mirror true}))

(defn make-valid [{:keys [rec-path batch-size]}]
  (mx-io/image-record-iter
    {:path-imgrec rec-path
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape [3 224 224]
     :rand-crop false
     :rand-mirror false}))

(defn make-data-loader [{:keys [train-path
                                valid-path
                                batch-size]
                         :or {batch-size 16}}]
  {:train (make-train {:rec-path train-path
                       :batch-size batch-size})
   :valid (make-valid {:rec-path valid-path
                       :batch-size batch-size})})

(defn fit [model data-loader {:keys [epochs
                                     batch-size]
                              :or {batch-size 16}}]
  (m/fit model
         {:train-data (:train data-loader)
          :eval-data (:valid data-loader)
          :num-epoch epochs
          :fit-params (m/fit-params {:intializer (init/xavier {:rand-type "gaussian"
                                                               :factor-type "in"
                                                               :magnitude 2})
                                     :batch-end-callback (callback/speedometer
                                                           batch-size
                                                           10)})}))

(defn save-model [model path]
 (m/save-checkpoint model {:prefix path
                           :epoch 0
                           :save-opt-states true}) )

(defn load-model [path]
  (m/load-checkpoint {:prefix path
                      :epoch 0}) )

(defn run-on [{:keys [cores cnum]}]
  (if (= :gpu cores)
    (mapv context/gpu (range cnum))
    (mapv context/cpu (range cnum))))

;; how to use it:

(comment

  (require '[groot.nn :as nn])

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

  (nn/fit model data-loader {:epochs 1})

  (nn/save-model model "model/groot")

)
