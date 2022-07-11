;; Homework Set #4
;; I worked with Owen Zhang on problems 2, 4, 10, 11, 12.
;; Our reasoning/phrasing of #10 and #12 will likely be similar, although neither of us has seen
;; or heard the answers we put. 
;; For # 11, I suggested a (let ) statement at the beginning, as well as applying (get-corpus )
;; within an equality function instead of outside in a (map ) call, because this kept the object
;; we created in the (let ) statement complete without removing the theta information.


;; I worked with Max Morehead as well, although we contributed very little to eachother's actual
;; code. He helped me with a debug of #9, and we checked answers for the first 6 questions.

;; I worked a little with Harmeena Sandhu, she was having trouble with the map function, so 
;; I explained it's functionality

;; I also worked with Grace Gao (which, I know, is 4 people plus me and is probably not allowed),
;; but none of the other groupmates were available and all I did was tell her which outputs
;; I got differently, not even the values of the outputs. However, she showed me her code and a lot of it is similar




;; Problem (1)

(def vocabulary '(call me ishmael))
(def theta1 (list (/ 1 2 ) (/ 1 4 ) (/ 1 4 )))
(def theta2 (list (/ 1 4 ) (/ 1 2 ) (/ 1 4 )))
(def thetas (list theta1 theta2))
(def theta-prior (list (/ 1 2) (/ 1 2)))

(defn score-categorical [outcome outcomes params]
  (if (empty? params)
    (/ 1 0)
    (if (= outcome (first outcomes))
      (first params)
      (score-categorical outcome (rest outcomes) (rest params)))))

(defn list-foldr [f base lst]
  (if (empty? lst)
    base
    (f (first lst)
       (list-foldr f base (rest lst)))))

(defn log2 [n]
  (/ (Math/log n) (Math/log 2)))


(defn score-BOW-sentence [sen probabilities]
  (list-foldr
   (fn [word rest-score]
     (+ (log2 (score-categorical word vocabulary probabilities))
        rest-score))
   0
   sen))

(defn score-corpus [corpus probabilities]
  (list-foldr
   (fn [sen rst]
     (+ (score-BOW-sentence sen probabilities) rst))
   0
   corpus))


(defn logsumexp [log-vals]
  (let [mx (apply max log-vals)]
    (+ mx
       (log2
        (apply +
               (map (fn [z] (Math/pow 2 z))
                    (map (fn [x] (- x mx)) log-vals)))))))

(def my-corpus '((call me) (call ishmael)))




;; Problem (1)

;; (defn theta-corpus-joint [theta corpus theta-probs] 
;;  (if (empty? corpus)
;;    (log2 (score-categorical theta thetas theta-probs))
;;    (+ (score-BOW-sentence (first corpus) theta)
;;       (theta-corpus-joint theta (rest corpus) theta-probs))))


;; This one works too
(defn theta-corpus-joint [theta corpus theta-probs] 
  (+ (score-corpus corpus theta) 
     (log2 (score-categorical theta thetas theta-probs))))

;;(theta-corpus-joint theta1 my-corpus theta-prior)
;; prints  -7







;; Problem (2)

(defn compute-marginal [corpus theta-probs] 
  (logsumexp (map (fn [x] (theta-corpus-joint x corpus theta-probs)) thetas)))


;; (compute-marginal my-corpus theta-prior)
;; -6.415037499278844







;; Problem (3)
(defn compute-conditional-prob [theta corpus theta-probs] 
  (- (theta-corpus-joint theta corpus theta-probs) 
     (compute-marginal corpus theta-probs)))

;;(compute-conditional-prob theta1 my-corpus theta-prior)
;; -0.5849625007211561
;; (Math/exp -0.5849625007211561) = 0.5571267532719365






;; Problem (4)
(defn compute-conditional-dist [corpus theta-probs] 
  (map (fn[x] (compute-conditional-prob x corpus theta-probs)) thetas))





;; Problem (5)
(compute-conditional-dist my-corpus theta-prior)

(def test (compute-conditional-dist my-corpus theta-prior))

(def distribution (list (Math/pow 2 (first test)) 
                        (Math/pow 2 (first (rest test)))))]


;; distribution = (0.6666666666666667 0.33333333333333337)


(sum distribution)
;; = 1

(logsumexp test)
;; ~ 0

;; When computing these values, we notice that the log of the conditional distribution in a list of negative numbers.
;; We know that the elements of our distribution should sum to 1, but the logged conditional will not do this.
;; Since Log_2(1) = 0, (logsumexp (conditional-probability-vector)) = 0    is the same as  (sum conditional-probability-vector) = 1

;; exponentiating the conditiondal probability vector gives us values ( (2/3),  (1/3) ) for (theta1, theta2) respectively. 
;; This result makes sense because theta1 predicts the word "call" will appear with greater probability. 

;; In my-corpus, "call" appears twice as often as "me" or "Ishmael" exactly as theta1 predicts. 
;; Since we began with an even probability of theta1 or theta2, we expect the conditional probability caluculation 
;; to return a probability of theta1 greater than its non-conditional probability since the condition of the corpus favors theta1. 



;; Problem (6)

(defn compute-posterior-predictive [observed-corpus new-corpus theta-probs]
  (let [conditional-dist (map (fn [x] (Math/pow 2 x)) 
                              (compute-conditional-dist observed-corpus 
                                                        theta-probs))]
    (compute-marginal observed-corpus conditional-dist)))

(compute-posterior-predictive my-corpus my-corpus theta-prior)

;; -6.2630344058337934
;; (Math/pow 2 -6.2630344058337934) = 0.013020833333333336
;; this seems like a reasonable increase given the value upon observing my-corpus is:
;; ~ 0.011





;; Problem (7)
(defn normalize [params]
  (let [sum (apply + params)]
    (map (fn [x] (/ x sum)) params)))

(defn flip [weight]
  (if (< (rand 1) weight)
    true
    false))

(defn sample-categorical [outcomes params]
  (if (flip (first params))
    (first outcomes)
    (sample-categorical (rest outcomes)
                        (normalize (rest params)))))
                                         
                                         
(defn repeat [f n]
  (if (= n 0)
    '()
    (cons (f) (repeat f (- n 1)))))


(defn sample-BOW-sentence [len probabilities]
  (if (= len 0)
    '()
    (cons (sample-categorical vocabulary probabilities)
          (sample-BOW-sentence (- len 1) probabilities))))




;; This function is the answer to Problem (7)

(defn sample-BOW-corpus [theta sent-len corpus-len]
  (let [sentence (fn [] (sample-BOW-sentence sent-len theta))]
    (repeat sentence corpus-len)))

;;(sample-BOW-corpus theta1 2 5)






;; Problem (8)

(defn sample-theta-corpus [sent-len corpus-len theta-probs]
  (let [theta (sample-categorical thetas theta-probs)]
  (list theta (sample-BOW-corpus theta sent-len corpus-len))))

;;(sample-theta-corpus 2 3 theta-prior)





;; Problem (9)

(defn get-theta [theta-corpus]
  (first theta-corpus))

(defn get-corpus [theta-corpus]
  (first (rest theta-corpus)))

(defn sample-thetas-corpora [sample-size sent-len corpus-len theta-probs]
  (repeat (fn [] (sample-theta-corpus sent-len corpus-len theta-probs))sample-size))





(defn estimate-corpus-marginal [corpus 
                                sample-size 
                                sent-len 
                                corpus-len 
                                theta-probs]
  (/ (count (filter (fn [y] (= y corpus)) 
                    (map (fn [x] (get-corpus x))
                         (sample-thetas-corpora sample-size 
                                                sent-len 
                                                corpus-len 
                                                theta-probs)))) 
     sample-size))



;; (estimate-corpus-marginal my-corpus 1000 2 2 theta-prior)






;; Problem (10)
(estimate-corpus-marginal my-corpus 50 2 2 theta-prior) 
;; [0.02, 0, 0, 0, 0.02, 0, 0.04, 0, 0, 0, 0] 

;; In the first set of estimate, the small sample size limits the resolution of the estimate.
;; The sample-size of 50 in the denominator of our marginal estimate means that every time we 
;; observe a matching corpus, we add at least 0.02 to our value. This makes 0.02 the smallest
;; increment with which we can update out marginal estimate.


(estimate-corpus-marginal my-corpus 1000 2 2 theta-prior)
;; [0.011, 0.007, 0.02, 0.012, 0.01, 0.011, 0.018, 0.013, 0.014]

;; Unfortunately, a sample size of 10000 would not run on my computer, but a corpus of 1000 entries
;; highlights the resolution point just as well. Notice now that our answers extend an additional
;; decimal place and show incrementation of 0.001. In general, the specificity of our estimate 
;; is bounded by the term (/ 1 sample-size). Statistically, this centers the confidence interval
;; for the true distribution of (theta1, theta2)  with more precision. 

;; In Problem (2) we found that the log of the estimated marginal to be [-6.415037499278844], which is a probability
;; of [0.01171875]. The average of the marginal estimated with 50 corpora is: 
;; (/ 0.08, 11) = 0.007272727272727273

;; The average of the marginal estimated with 1000 corpora is:
;; (/ 0.116 9) = 0.012888888888888889, a much nearer estimate

;; We know by the central limit theorem that the average of enough estimates will tend to distribute itself normally about the
;; true mean, but because the accuracy of the 1000 sample-size estimate is higher, we can infer that it will converge to the 
;; true marginal of my-corpus faster.

 



;; Problem (11)
(defn get-count [obs observation-list count]
  (if (empty? observation-list)
    count
    (if (= obs (first observation-list))
      (get-count obs (rest observation-list) (+ 1 count))
      (get-count obs (rest observation-list) count))))


(defn get-counts [outcomes observation-list]
  (let [count-obs (fn [obs] (get-count obs observation-list 0))]
    (map count-obs outcomes)))



(defn rejection-sampler [theta 
                         observed-corpus
                         sample-size 
                         sent-len 
                         corpus-len 
                         theta-probs]
  (let [corpus-matches (filter (fn [y] (= (get-corpus y) observed-corpus))
                                    (sample-thetas-corpora sample-size
                                                           sent-len
                                                           corpus-len
                                                           theta-probs))]
    (/ (get-count theta (map (fn [z] (get-theta z)) corpus-matches) 0)
       (count corpus-matches))
    )
  )

(rejection-sampler theta1 my-corpus 5000 2 2 theta-prior)






;; THIS WORKS GIVES ME THETA PROPORTIONS & NICE ##s
;; If the grader sees this, it was kind of funny how much time I spent comparing the below lines to my function 
;; wondering what was wrong, they were almost identical. In the end, it turns out I just hadn't passed in my-corpus
;; as an argument so it was bugging every time  XD

;;(def athing (filter (fn [y] (= (get-corpus y) my-corpus))
;;                    (sample-thetas-corpora 5000 
;;                                           2 
;;                                           2 
;;                                           theta-prior)))

;;(/ (get-count theta1 (map (fn [z] (get-theta z)) athing) 0) 
;;   (count athing))







;; Problem (12)
;;(rejection-sampler theta1 my-corpus 100 2 2 theta-prior)

;; this returns [0, 0.5, ##NaN, 0, 1, ##NaN, 1, 1, ##NaN, 0.66667]

;; (rejection-sampler theta1 my-corpus 8920 2 2 theta-prior)

;; this returns [0.7129, 0.6666, 0.6320, 0.66, 0.7115, 0.5769, 0.6823

;; Even after boosting my sample size to the very limit of what my computer could handle, we still see values off of the probabilty 
;; for [theta1] estimated in problems (5) and (6) (the estimate for 6 is not the one we are looking for, but similar).

;; With the naked eye we can see how much greater the variance of the small sample is compared with the large sample. 
;; Certainly, the large sample shows deviations of about [0.05] or [0.09] from the true value computed in (5), but these jumps
;; are not nearly as large as the [0.33] and [0.1666] jumps we note in the small sample: and that is IF it returns any successes at all

;; Because [my-corpus] appears with (P(my-corpus) < 0.012), in a sample size of 50, there are likely to be more unusable ##NaN 
;; returns. Because of this, when the (rejection-sampler  ) function detects a like value, the return is impacted far more than
;; a detected hit would impact the large sample due to the relative count of matches. In the large sample, even if the density of
;; matches is identical, because the number of prior matches is greater, each additional desired observsation impacts the final
;; result less and less.
