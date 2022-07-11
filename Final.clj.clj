;; FINAL ASSIGNMENT

(def hidden-states '(Start N V))

(def vocabulary '(Call me Ishmael))

(def theta-transition-Start '(0 0.9 0.1))

(def theta-transition-N '(0 0.3 0.7))

(def theta-transition-V '(0 0.8 0.2))

(def theta-transition-dists-1
  (list theta-transition-Start theta-transition-N theta-transition-V))

(def theta-observation-Start '(0.33 0.33 0.33))

(def theta-observation-N '(0.1 0.5 0.4))

(def theta-observation-V '(0.8 0.1 0.1))

(def theta-observation-dists-1
  (list theta-observation-Start
        theta-observation-N
        theta-observation-V))

(defn dist-lookup [state states dists]
  (if (= state (first states))
    (first dists)
    (dist-lookup state (rest states) (rest dists))))

(defn log2 [n]
  (/ (Math/log n) (Math/log 2)))

(defn logsumexp [log-vals]
  (let [mx (apply max log-vals)]
    (+ mx
       (log2
        (apply +
               (map (fn [z] (Math/pow 2 z))
                    (map (fn [x] (- x mx)) log-vals)))))))

(defn logscore-categorical [outcome outcomes params]
  (if (= outcome (first outcomes))
    (log2 (first params))
    (logscore-categorical outcome (rest outcomes) (rest params))))



;; Problem (1)
(defn score-next-state-word [current-hidden 
                             next-hidden 
                             next-observed 
                             theta-transition-dists 
                             theta-observation-dists]
  (+(logscore-categorical next-observed 
                          vocabulary
                          (dist-lookup next-hidden
                                       hidden-states
                                       theta-observation-dists)) 
    (logscore-categorical next-hidden 
                          hidden-states 
                          (dist-lookup current-hidden
                                       hidden-states
                                       theta-transition-dists))))
 

;; Here we take the categorical score to check that the values are reasonable

;;(logscore-categorical 'V
;;                     hidden-states
;;                      (dist-lookup 'N
;;                                   hidden-states
;;                                   theta-transition-dists-1))
;;(Math/pow 2 (logscore-categorical 'V
;;                      hidden-states
;;                      (dist-lookup 'N
;;                                   hidden-states
;;                                   theta-transition-dists-1)))

;; (the sum of all words given {H = N, H+1 = V}) = P(H+1 = V | H = N) = 0.7
;;(Math/pow 2 (logsumexp (list (score-next-state-word 'N
;;                       'V
;;                       'Call
;;                       theta-transition-dists-1
;;                       theta-observation-dists-1)
;;                 (score-next-state-word 'N
;;                       'V
;;                       'me
;;                       theta-transition-dists-1
;;                       theta-observation-dists-1)
;;                 (score-next-state-word 'N
;;                       'V
;;                       'Ishmael
;;                       theta-transition-dists-1
;;                       theta-observation-dists-1))))



;; Problem (2)
(defn compute-next-observation-marginal [current-state
                                         next-observation
                                         theta-transition-dists
                                         theta-observation-dists]
  (logsumexp (list (score-next-state-word current-state
                                          'N
                                          next-observation
                                          theta-transition-dists 
                                          theta-observation-dists)
                   (score-next-state-word current-state
                                          'V
                                          next-observation
                                          theta-transition-dists 
                                          theta-observation-dists))))


;;(compute-next-observation-marginal 'V
;;                                   'Call
;;                                   theta-transition-dists-1
;;                                   theta-observation-dists-1)
;; -2.0588936890535683






;; Problem (3)

(defn score-next-states-words [current-hidden 
                               next-hidden-states 
                               next-words 
                               theta-transition-dists 
                               theta-observation-dists]
  (if (empty? next-words)
    0
    (logsumexp (compute-next-observation-marginal 
                current-hidden
                (first next-words)
                theta-transition-dists 
                theta-observation-dists)
               
               (score-next-states-words 
                (first next-hidden-states)
                (rest next-hidden-states)
                (rest next-words)
                theta-transition-dists
                theta-observation-dists))))


;; Test
;;(score-next-states-words 'Start
;;                         (list N V N N V)
;;                         (list Ishmael me call me call)
;;                         theta-transition-dists-1
;;                         theta-observation-dists-1)



;; Problem (4)
(defn compute-next-words-marginal [current-hidden 
                                   next-words 
                                   theta-transition-dists 
                                   theta-observation-dists]
  (if (empty? (rest next-words))
    (compute-next-observation-marginal
                          current-hidden
                          (first next-words)
                          theta-transition-dists
                          theta-observation-dists)
    (+ (compute-next-observation-marginal
                          current-hidden
                          (first next-words)
                          theta-transition-dists
                          theta-observation-dists)
       (logsumexp (list (compute-next-words-marginal
                         'N
                         (rest next-words)
                         theta-transition-dists 
                         theta-observation-dists)
                        (compute-next-words-marginal
                         'V
                         (rest next-words)
                         theta-transition-dists 
                         theta-observation-dists))))))


;; Test
;;(compute-next-words-marginal 'Start
;;                             (list 'Ishmael 'Ishmael 'me)
;;                             theta-transition-dists-1
;;                             theta-observation-dists-1)



;; Problem (5)

(compute-next-words-marginal 'Start
                             (list 'Call 'Ishmael )
                             theta-transition-dists-1
                             theta-observation-dists-1)
;; -3.4723290837359104


(compute-next-words-marginal 'Start
                             (list 'Ishmael 'Call)
                             theta-transition-dists-1
                             theta-observation-dists-1)
;; -1.7032195825735748

;; These two word pairs have different probabilities because the ordering of hidden states impacts the 
;; ordering of words. 'Call' appears with a high probability when the hidden state is a Verb, but
;; Verbs appear with low probability after 'Start'. The 'Call' at the start of the first combination 
;; means that some low-probability path has been taken. Either the high-probability hidden state Noun 
;; has followed the 'Start', and adopted the low-probability expression 'Call', or the low-probability Verb
;; hidden state has followed the 'Start', and adopted the high-probability expression 'Call'. 

;; Going forward, let us refer to hidden states and their observations as P(H|prior H)-P(W|H). For example,
;; V-Call = low-high, since V after 'Start' is low probability, and 'Call' with V is high probability.
;; N-Call = high-low
;; This format will shorten some explanations

;; CRITICALLY! because the first observation in the first word pait MUST be a combination of one high, one low-probability
;; occurence, the analysis of any following terms will be modified by this high-low combination. The second term
;; 'Ishmael' provides a similar problem since V-Ishmael is high-low following nouns, and low-low following verbs.
;; Likewise, N-Ishmael is low-med follwing a noun, but high-med following a verb. Of the four combinations, 
;; the best crude ranking of probability we have is V-Call N-Ishmael, which is:
;; low-high high-med. The probability of this occurence is the highest, and thus contributes the most to the overall
;; probability value. If we had more following words, the construction that begin with this parse would contribute more to the
;; overall probability than any other constructions with identical tails.

;; In other words, P(V-Call N-Ishmael H3-W3 H4-W4 H5-W5) >= P(?-Call ?-Ishmael H3-W3 H4-W4 H5-W5) for identical values of 
;; H_n and W_n.

;; In the second word pair, 'Ishmael' coming first means a high-high probability combination of N-Ishmael, or a
;; low-low probability combination of V-Ishmael following 'Start'. In terms of the statistics, the analysis will take
;; two paths, one modified by the low-low probability, and another modified by the high-high probability. The second term
;; 'Call' also has a high-high probability following N-Ishmael, while N-Call has a low-low probability.
;; N-Ishmael V-Call is a high-high high-high. Thus given more words in the list, any branch from this term will overpower
;; an identical branch from any other construction of two first-terms. 
;; The fact that possible combinations of these branches are not identical in wordpair one and wordpair two are exactly 
;; why their probabilities are not equal.



;; Problem (6)

;; Not having a 'Start to our input [k] makes copious use of the (reverse)
;; function unfortunately necessary


(defn compute-hidden-prior [k theta-transition-dists]
  (if (empty? (rest k))
    (logscore-categorical (first k) hidden-states theta-transition-Start)
    (+ (logscore-categorical (first (reverse k))
                             hidden-states
                             (dist-lookup (first (rest (reverse k)))
                                          hidden-states
                                          theta-transition-dists))
       (compute-hidden-prior (reverse (rest (reverse k))) 
                             theta-transition-dists))))

    
;; Test
;;(compute-hidden-prior (list 'N 'V 'N 'N) theta-transition-dists-1)



;; Problem (7)
(defn compute-likelihood-of-words [k
                                   words 
                                   theta-observation-dists]
(if (empty? k)
  0
  (+ (logscore-categorical (first words)
                           vocabulary
                           (dist-lookup (first k)
                                        hidden-states
                                        theta-observation-dists)) 
     (compute-likelihood-of-words (rest k)
                                  (rest words)
                                  theta-observation-dists))))

;; Test
;; the below are equal
;;(compute-likelihood-of-words (list 'N 'V 'N 'V) 
;;                             (list 'Ishmael 'Call 'me 'Ishmael)
;;                             theta-observation-dists-1)
;;(log2 (* 0.4 0.8 0.5 0.1))



;; Problem (8)
;; I feel like my solution to this problem is not quite right.

(defn compute-hidden-posterior [k 
                                words 
                                theta-transition-dists 
                                theta-observation-dists]
  (- (+ (compute-likelihood-of-words k
                                     words
                                     theta-observation-dists)
        (compute-hidden-prior k
                              theta-transition-dists))
        (compute-next-words-marginal 'Start
                                     words
                                     theta-transition-dists
                                     theta-observation-dists)))

;; (compute-hidden-posterior (list 'N 'V 'V 'V)
;;                          (list 'me 'Call 'Call 'Call)
;;                          theta-transition-dists-1
;;                          theta-observation-dists-1)



;; Problem (9)

;; for simplicity's sake, and so we aren't ranging from [t] to [t+k], let's assume [t = 0] to start.

;; At the first layer of the function where we calcuate only:
;; P(H1 = h1 | H0 = h0)*P(W1 = w1 | H1 = h1)*P(rest of the words | rest of the hidden states)
;; we have called the function only once because we do not need to sum over the identity of the input hidden state
;; since we already know that. 
;; Every time after the first, we call the function 2 times for each time it has been called previously, so 
;; for k = 2, we will call it 1 + 2 times
;; for k = 3, we will call it 1 + 2 + 4 times

;; in general, the number of times the function (compute-next-words-marginal) is called to determine the marginal 
;; of a k-length list of words is given by the simple sum:

;; Sum(2 ^ (n-1)) for [n] from {1 to k}

;;This means that our equation increases by an order related to O(2^(n-1)) ~ O(2^n)
;; Kind of iterestingly, if we take the sums of these numbers, we realize they will always be off from a power of 2
;; by a single integer each time
;; 1      for k = 1
;; 3      for k = 2
;; 7      for k = 3
;; 15     for k = 4
;; 31     for k = 5
;; 63     for k = 6
;; 127    for k = 7
;; and so on... We see that a better equation for this might be (2^n)-1, which indeed proceeds at order O(2^n)




;; Problem (10)



;; Problem (11)