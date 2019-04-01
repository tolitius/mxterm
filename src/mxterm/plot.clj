(ns mxterm.plot
  (:require [clojure.edn :as edn]
            [oz.core :as oz]))

(defn fmap-v [m f]
  (into {}
        (for [[k v] m]
          [k (f v)])))

(defn one-minus [xs]
  (map #(- 1 %) xs))

;; {:epochs 5
;;  :metrics {:accuracy [...]
;;            :f1 [...]
;;            :other [...]}}
(defn to-losses [stats]
  (update stats :metrics fmap-v one-minus))

;; this needs work: it's somewhat bumpy on epoch change
(defn to-batch-loss [metrics epochs epoch]
  (let [losses (nth (partition-all (/ (count metrics) epochs)
                                   metrics)
                    epoch)
        loss-count (count losses)
        step (/ 1.0 loss-count)]
    (map vector (take loss-count
                      (iterate #(+ step %) (+ epoch 0.02)))
         losses)))

(defn loss-data [{:keys [epochs metrics]}]
  (for [lname (keys metrics)
        epoch (range epochs)
        [batch loss] (to-batch-loss (metrics lname)
                                    epochs
                                    epoch)]
    {:epoch batch
     :kind lname
     :loss loss}))

(defn plottable-losses [metrics]
  (let [losses (to-losses metrics)]
    {:data {:values (loss-data losses)}
     :encoding {:x {:field "epoch"}
                :y {:field "loss"}
                :color {:field "kind" :type "nominal"}}
     :mark "line"}))

;; would be niice if mxnet model would be
;; wrapped into something that included data loader and train metrics
;; rather than passing a path to scraped metrics after the fact
(defn plot-losses [path]
  (let [metrics (-> path slurp edn/read-string)
        losses (plottable-losses metrics)]
    ;; (oz/start-plot-server!) ;; if not already started
    (oz/view! losses)))


;; how to REPL/use it:
(comment

  (require '[oz.core :as oz] '[mxterm.plot :as plot])
  (oz/start-plot-server!)

  (p/plot-losses "dev-resources/groot/recorded-metrics.edn")

)
