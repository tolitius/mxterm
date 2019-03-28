(def +version+ "0.1.0")

(set-env!
  :source-paths #{"src"}
  :dependencies '[[org.clojure/clojure      "1.9.0"]

                  [org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu "1.5.0-SNAPSHOT"]
                  [org.apache.mxnet.contrib.clojure/clojure-mxnet "1.5.0-SNAPSHOT"]
                  [origami "4.0.0-3"]

                  ;; boot clj
                  [boot/core                "2.8.2"           :scope "provided"]
                  [adzerk/bootlaces         "0.2.0"           :scope "test"]
                  [adzerk/boot-test         "1.2.0"           :scope "test"]
                  [tolitius/boot-check      "0.1.12"          :scope "test"]]

  :repositories #(conj % ["staging" {:url "https://repository.apache.org/content/repositories/staging"
                                     :snapshots true
                                     :update :always}]
                         ["snapshots" {:url "https://repository.apache.org/content/repositories/snapshots"
                                       :snapshots true
                                       :update :always}]
                         ["vendredi" {:url "https://repository.hellonico.info/repository/hellonico/"}]))

(require '[adzerk.bootlaces :refer :all]
         '[tolitius.boot-check :as check]
         '[adzerk.boot-test :as bt])

(bootlaces! +version+)

(defn uber-env []
  (set-env! :source-paths #(conj % "test"))
  (set-env! :resource-paths #(conj % "dev-resources"))
  ; (System/setProperty "conf" "dev-resources/config.edn")
  )

(deftask dev []
  (uber-env)
  (repl))

(deftask test []
  (uber-env)
  (bt/test))

(deftask check-sources []
  (comp
    (check/with-bikeshed)
    (check/with-eastwood)
    (check/with-yagni)
    (check/with-kibit)))

(task-options!
  push {:ensure-branch nil}
  pom {:project     'groot
       :version     +version+
       :description "mxnet touch and feel"
       :url         "https://github.com/tolitius/groot"
       :scm         {:url "https://github.com/tolitius/groot"}
       :license     {"Eclipse Public License"
                     "http://www.eclipse.org/legal/epl-v10.html"}})
