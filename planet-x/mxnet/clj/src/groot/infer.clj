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
                                :or {show? true
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
        (ffirst))))


(comment

  (require '[groot.nn :as nn] '[groot.infer :as infer])

  (def model (nn/load-model "model/groot"))

  (infer/predict model "file:./data/groot/valid/no-human/01-2019.05-51-47.jpg" :show? false)

  ;; => #object[org.apache.mxnet.NDArray 0x16923d4a "[\n [0.20457779,0.79542214]\n]\n<NDArray (1,2) cpu(0) float32>"]
)
