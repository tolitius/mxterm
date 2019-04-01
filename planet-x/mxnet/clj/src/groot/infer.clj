(ns groot.infer
  (:require [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

(defn load-image [url & {:keys [height width channels]
                         :or {height 224
                              width 224
                              channels 3}}]
  (-> url
      (cvu/mat-from-url)
      (cv/resize! (cv/new-size height width))
      (cv/convert-to! cv/CV_8SC3 0.5)
      (cvu/mat->flat-rgb-array)
      (ndarray/array [1 channels height width])))

(defn predict [model img-url & {:keys [height width channels]
                                :or {height 224
                                     width 224
                                     channels 3}}]
  (let [nd-img (load-image img-url)]
    (-> model
        (m/bind {:for-training false
                 :data-shapes [{:name "data"
                                :shape [1 channels height width]}]})
        (m/forward {:data [nd-img]})
        (m/outputs)
        (ffirst)
        ndarray/->vec)))


;; how to REPL/use it:
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
