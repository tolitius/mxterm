(ns groot.infer
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.shape :as mx-shape]
            [org.apache.clojure-mxnet.symbol :as sym]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

(defn load-image [url & {:keys [show? height width channels]
                         :or {show? true
                              height 224
                              width 224
                              channels 3}}]
  (-> url
      (cvu/mat-from-url)
      (cv/resize! (cv/new-size height width))
      (#(do (if show? (cvu/imshow %)) %))
      (cv/convert-to! cv/CV_8SC3 0.5)
      (cvu/mat->flat-rgb-array)
      (ndarray/array [1 channels height width])))

(defn predict [model img-url & {:keys [show? height width channels]
                                :or {show? false
                                     height 224
                                     width 224
                                     channels 3}}]
  (let [nd-img (load-image img-url :show? show?)]
    (-> model
        (m/bind {:for-training false
                 :data-shapes [{:name "data"
                                :shape [1 channels height width]}]})
        (m/forward {:data [nd-img]})
        (m/outputs)
        (ffirst)
        ndarray/->vec)))


(comment

  (require '[groot.nn :as nn] '[groot.infer :as infer] '[org.apache.clojure-mxnet.module :as mm])

  (def model (nn/load-model "model/groot"))

  (infer/predict model "file:./data/groot/valid/no-human/01-2019.05-51-47.jpg")

  ;; boot.user=> (def m (mm/load-checkpoint {:prefix "model/groot-18" :epoch 6}))

  ;; boot.user=> (infer/predict m "file:./data/groot/valid/human/04-2018.10-02-19.jpg")
  ;; => [0.9444668 0.05553314]

  ;; boot.user=> (infer/predict m "file:./data/groot/valid/no-human/07-2018.20-38-06.jpg")
  ;; => [0.1873892 0.8126108]

  ;; boot.user=> (infer/predict m "file:./data/groot/valid/no-human/01-2019.05-51-47.jpg")
  ;; => [0.004020324 0.99597967]

  ;; boot.user=> (infer/predict m "file:./data/groot/valid/human/03-2018.11-17-21.jpg")
  ;; => [0.9457831 0.054216918]
)
