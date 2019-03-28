(ns groot.metrics
  (:require [org.apache.clojure-mxnet.util :as util])
  (:import [org.apache.mxnet CompositeEvalMetric]))

(defn comp
  "Create a metric instance composed out of several metrics"
  [metrics]
  (let [cm (CompositeEvalMetric.)]
    (doseq [m metrics] (.add cm m))
    cm))

(defn get
  "Get the values of the metric in as a map of {name value} pairs"
  [metric]
  (apply zipmap (-> (.get metric) util/tuple->vec)))
