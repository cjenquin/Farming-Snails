Problem Set #3
;; I worked with: Owen Zhang, Max Morehead, and Grace Gao
;; mostly we talked about the expectations of the problems and clarified some function of (let ) statements
;; and probability calculations, but little exchange of specific 
;; Problem (2) Max pointed out that taking equality of the first element included determining membership in the list, this
;; allowed me to remove an (if ) statement

;; Problem (4) Max suggested using the (repeatedly ) function instead of a (let ) statement or a long (if ) statement.


;; Problem (1)

(def moby-word-tokens '(CALL me Ishmael . Some years ago never mind
how long precisely having little or no money in my purse , and
nothing particular to interest me on shore , I thought I would
sail about a little and see the part of the world . It is
a way I have of driving off the spleen , and regulating the
circulation . Whenever I find myself growing grim about the mouth
whenever it is a damp , drizzly November in my soul whenever I
find myself involuntarily pausing before coffin warehouses , and
bringing up the rear of every funeral I meet and especially
whenever my hypos get such an upper hand of me , that it requires
a strong moral principle to prevent me from deliberately stepping
into the street , and methodically knocking people's hats off
then , I account it high time to get to sea as soon as I can .
This is my substitute for pistol and ball . With a philosophical
flourish Cato throws himself upon his sword I quietly take to the
ship . There is nothing surprising in this . If they but knew it
, almost all men in their degree , some time or other , cherish
very nearly the same feelings toward the ocean with me .))

(defn member-of-list? [w l]
  (if (empty? l)
    false
    (if (= w (first l))
      true
      (member-of-list? w (rest l)))))

(member-of-list? 'the moby-word-tokens)

(defn get-vocabulary [word-tokens vocab])



(defn get-vocabulary [word-tokens vocab]
  (if (empty? word-tokens)
    vocab
    (if (member-of-list? (first word-tokens) 
                         vocab)
      (get-vocabulary (rest word-tokens) 
                      vocab)
      (get-vocabulary (rest word-tokens) 
                      (concat vocab (list (first word-tokens)))))))



;;(get-vocabulary '(the man is the man with a plan) '())
;;(get-vocabulary moby-word-tokens '())




;; Problem (2)

(defn get-count-of-word [w word-tokens count]
  (if (empty? word-tokens)
    count
    (if (= w (first word-tokens))
      (get-count-of-word w (rest word-tokens) (+ 1 count))
      (get-count-of-word w (rest word-tokens) count))))

;;(get-count-of-word 'the '(the the cat cat drank the water) 0)







(defn get-word-counts [vocab word-tokens]
  (let [count-word (fn [w]
                     (get-count-of-word w word-tokens 0))]
    (map count-word vocab)))






;; Problem (3)
;; This is the brain-dead-est way to solve the problem

(def moby-word-frequencies (get-word-counts (get-vocabulary moby-word-tokens
                                                            '()) 
                                            moby-word-tokens))
;;moby-word-frequencies




;; Problem (4)
;; I opted not to use the (repeatedly ) function because I was not certain how to use it with
;; a (let ) statement.

(defn sample-uniform-BOW-sentence [n vocab] 
  (if (= n 0)
    '()
    (cons (sample-categorical vocab (create-uniform-distribution vocab)) 
          (sample-uniform-BOW-sentence (- n 1) 
                                       vocab))
    )
  )


;;(sample-uniform-BOW-sentence 4 '(a b c d))
;;(sample-uniform-BOW-sentence 5 '(the cat in hat))




;; Problem (5)
(defn compute-uniform-BOW-prob [vocab sentence]
  (Math/pow (/ 1 (count vocab)) (count sentence)))

;; this form of function assumes [sentence] is composed of [vocab]
;;(compute-uniform-BOW-prob '(the cat in hat) '(the cat in the hat))
;; return = 0.0009765625 = (1/4)^5



;; Problem (6)
;; Because our BOW distribution is uniform, the probability of observing each element is equal, and thus
;; the only variable term in computing the probability of a sentence is the length of the sentence itself.
;; I have coded my solution to reflect this rather than using the uniform probability distribution, because
;; I know the values of this distribution are 1/(len vocabulary). Indeed, the probability value of all 
;; sentences created this way are equal provided they are the same length. 

(def moby-vocab (get-vocabulary moby-word-tokens '()))

(def x (sample-uniform-BOW-sentence 3 moby-vocab))
;; x = (philosophical sword coffin)

(compute-uniform-BOW-prob moby-vocab '(philosophical sword coffin))
;;  3.72353636163581e-7



(def y (sample-uniform-BOW-sentence 3 moby-vocab))
;; y = (find have interest) 

(compute-uniform-BOW-prob moby-vocab '(find have interest))
;; 3.72353636163581e-7



(def z (sample-uniform-BOW-sentence 3 moby-vocab))
;; z = (the whenever for)

(compute-uniform-BOW-prob moby-vocab '(the whenever for))
;; 3.72353636163581e-7 








(defn sample-BOW-sentence [len vocabulary probabilities]
  (if (= len 0)
    '()
    (cons (sample-categorical vocabulary probabilities)
          (sample-BOW-sentence (- len 1) vocabulary probabilities))))








;; Problem (7)

(def moby-word-probabilities (normalize moby-word-frequencies))

;;moby-word-probabilities



;; Problem (8)

(def moby1 (sample-BOW-sentence 3 moby-vocab moby-word-probabilities))
moby1
;; (pistol people's then)

(def moby2 (sample-BOW-sentence 3 moby-vocab moby-word-probabilities))
moby2
;;(I ball the)

(def moby3 (sample-BOW-sentence 3 moby-vocab moby-word-probabilities))
moby3
;;(throws warehouses the)

(def moby4 (sample-BOW-sentence 3 moby-vocab moby-word-probabilities))
moby4
;;(world sail that)

(def moby5 (sample-BOW-sentence 3 moby-vocab moby-word-probabilities))
moby5
;;(prevent very men)



;; Problem (9)

(defn lookup-probability [w outcomes probs]
  (if (= w (first outcomes))
    (first probs)
    (lookup-probability w (rest outcomes) (rest probs))))

;;(lookup-probability 'cat '(the cat in hat) '(0.1 0.1 0.2 0.7))




(defn product [l]
(apply * l))



;; Problem (10)

(defn compute-BOW-prob [sentence vocabulary probabilities]
  (if (empty? (rest sentence))
    (lookup-probability (first sentence) 
                        vocabulary 
                        probabilities)
    (* (lookup-probability (first sentence)
                                 vocabulary
                                 probabilities) 
             (compute-BOW-prob (rest sentence)
                               vocabulary
                               probabilities))))

;; (compute-BOW-prob '(the cat in the hat) '(the cat in hat comes back) '(0.4 0.2 0.1 0.2 0.05 0.05))



;; Problem (11)


(compute-BOW-prob '(pistol people's then) moby-vocab moby-word-probabilities)
;; 1.1112454483386438e-7


(compute-BOW-prob '(I ball the) moby-vocab moby-word-probabilities)
';; 0.000010001209035047792


(compute-BOW-prob '(throws warehouses the) moby-vocab moby-word-probabilities)
;; 0.0000011112454483386437


(compute-BOW-prob '(world sail that) moby-vocab moby-word-probabilities)
;; 1.1112454483386438e-7
 

(compute-BOW-prob '(prevent very men) moby-vocab moby-word-probabilities)
;; 1.1112454483386438e-7


;; moby1, moby4, and moby5 all have the same probability and are constructed out of words that appear only
;; once in the corpus. Unlike problem 6, the probability of a length 3 sentence is not fixed because we are
;; not using a uniform distribution to assign probability. The three sentences in my trials that have equal
;; probability are proof that using the data from Moby Dick, we can still create BOW sentences with equal
;; probability

;; A simpler way to construct these sentences might be to change only one list item. If we replace an item
;; of any sentence with another item that has equal probability, the probability of the altered sentence
;; is the same as the probability of the original. 



;; To compute the probability of a sentence in a bag of words model, there are a few methods possible.

;; We could know the probability value of each word, in which case we simply multiply the value of each
;; word appearing in our sentence.

;; Alternatively, we could know the frequency of each word in the corpus from which we draw our distribution.
;; In this way, we would multiply our probability by (word_frequency)/(corpus_size) for each word in our
;; sentence, which is the SAME process provided the word probabilities are proportional to the frequency
;; in the corpus. 
