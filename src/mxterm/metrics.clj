(ns mxterm.metrics
  (:require [org.apache.clojure-mxnet.util :as util]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.eval-metric :as em])
  (:import [org.apache.mxnet CompositeEvalMetric]))

(defn comp-metric
  "create a metric instance composed out of several metrics"
  [metrics]
  (let [cm (CompositeEvalMetric.)]
    (doseq [m metrics] (.add cm m))
    cm))

(defn get
  "get the values of the metric in as a map of {name value} pairs"
  [metric]
  (apply zipmap (-> (.get metric) util/tuple->vec)))

;; (def rec (atom []))
;; (record-metric rec (eval-metric/accuracy) "accuracy")
(defn record-metric
  "records metrics for every batch in a \"recorder\" (clojure atom)"
  [recorder metric mname]
  (em/custom-metric (fn [label pred]
                      (em/update metric [label] [pred])
                      (let [m (->> metric
                                   get
                                   (apply val))] ;; assumes it's a map, once mxnet #14553 is merged update this
                        (swap! recorder conj m)
                        m))
                    mname))
