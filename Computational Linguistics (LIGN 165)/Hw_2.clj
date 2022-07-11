;; Problem Set (2)
;; Problem Set #2

;; Problem (1)
(defn sequence-to-power [x n]
  (if (= 0 (- n 1))
    x
    (concat x (sequence-to-power x (- n 1)))))

;;(sequence-to-power (list 'a 'b) 3)



;; Problem (2) ???
(defn in-L? [x] 
  (if (empty? x)
  (= 0 0)
  (if (= 'b (first x))
    (in-L? (rest x))
    (= 1 0))))

;;(in-L? '(b b b b b))




;; Problem (3)
;; We can easily write this procedure using the code of Problem (1)

(defn sequence-to-power [x n]
  (if (= 0 (- n 1))
    x
    (concat x (sequence-to-power x (- n 1)))))


(defn generate-bn-an [k]
  (concat (sequence-to-power (list 'b) k) (sequence-to-power (list 'a) k)))

;;(generate-bn-an 6)




;; Problem (4)
;; Rather than re-invent the wheel, we use the (reverse  ) function as the hint suggests

(defn reverse [l] 
  
  (if (= l '())
  '()
  (concat(reverse (rest l)) (list (first l)))))

;; Now we simply create our removal procedure by calling reverse once to access the last element,
;; and again to re-orient the result.

(defn remove-last-element [l] 
  (reverse (rest (reverse l))))




;; Problem (5)

(defn recognize-bn-an [str]
  (if (= 'b (first str))
      (if (= 'a (first (reverse str)))
        (recognize-bn-an (remove-last-element (rest str)))
        (= 1 0))
      (= 1 0)))



;; Problem (6)
;; My difficulty with this problem is that we cannot simply add L to the front of each A
;; entry using (map (fn [A] (concat L A)) A). Instead, we have to pull the first (and only)
;; entry of L from its list and then (cons ) it to the front of A. This is problematic when 
;; L is longer than one element.

(defn concat-L-A [L A] (map (fn[B] (cons (first L) B)) A))
  
;;(concat-L-A (list 'q) (list (list 'a 'b) (list 'b 'b)))



;; Problem (7)
;; Let A = { empty-string }
;;     B = {'a }

;; CONCAT(A,B) = L = {'a } = CONCAT(B,A)

;; Concatenating language A with any language C results in that language C.


;; Alternatively,
;; Let A = B :  for example A = {'ab, 'ba}, B = {'ab, ba'}
;; CONCAT(A,B) = {'abab, 'baab, 'abba, 'baba }
;; CONCAT(B,A) = {'abab, 'baab, 'abba, 'baba }

;; This will be true for any languages A = B



;; Problem (8) 
;; Want to find languages A and B s.t. CONCAT(A,B) != CONCAT(B,A)
;; Let A = {'ab, 'cd }
;; Let B = {'ba, 'dc }

;; CONCAT(A,B) = {'abba, 'abdc, 'cdba, 'cddc }
;; CONCAT(B,A) = {'baab, 'bacd, 'dcab, 'dccd }
;; No elements of CONCAT(B,A) can be found in CONCAT(A,B)


;; Problem (9)
;; Want to find a language L s.t. L = L^2 --> L = CONCAT(L,L)

;; Let L = {empty-string}
;; Then CONCAT(L,L) = {empty-string} = L = L^2 as required. 
;; If L has even a single non-empty element, L^2 cannot equal L, though L^2 can be a subset
;; of the original language L.



