import FormalConjectures.Util.ForMathlib

set_option maxHeartbeats 700000

open Nat Real Filter EReal Finset
open scoped Classical

namespace KakeyaConstruction


/-!
Lean file for proving a formula for the size of a subset of F_p^3.

Let F_p​ be the finite field with p elements, where p is an odd prime. Let S be the set of squares in F_p​ (including 0).
The set K is constructed as the union of two sets, K1​ and K2​.

K1​={(x,y,z) ∈ F_p^3​∣x^2 + 4y ∈ S and x^2 + 4z ∈ S}.

K2​ is a set defined as the union of K2a​ and K2b​:
K2a​={(0,y,z)∣y^2 − z ∈ S}.
K2b​={(0,0,z)∣z ∈ F_p​}.

K = K1 ∪ K2 = K1 ∪ K2a ∪ K2b

The statement we are proving here is that the size of K is given by the following formula:

If p≡1(mod4):

∣K∣= (2p^3 + 7p^2 + 4p − 5) / 8​

If p≡3(mod4):

∣K∣= (2p^3 + 7p^2 + 2p − 3) / 8
​
These can be combined into a single formula using the quadratic character (Legendre symbol) χ(x)=(px​), where χ(−1)=1 if p≡1(mod4) and χ(−1)=−1 if p≡3(mod4):

∣K∣= (2p^3 + 7p^2 + 3p − 4 + (p − 1)χ(−1)​) / 8
-/




/-!
# Part 1: Definitions as Explicit Functions
-/

-- We define each set as a function that explicitly takes the prime `p`
-- and the `Fact (Nat.Prime p)` instance as arguments.

def K1 (p : ℕ) [Fact (Nat.Prime p)] : Finset (Fin 3 → ZMod p) :=
  univ.filter (fun v =>
    let x := v 0; let y := v 1; let z := v 2
    IsSquare (x^2 + 4*y) ∧ IsSquare (x^2 + 4*z)
  )

def K2a (p : ℕ) [Fact (Nat.Prime p)] : Finset (Fin 3 → ZMod p) :=
  univ.filter (fun v =>
    let x := v 0; let y := v 1; let z := v 2
    x = 0 ∧ IsSquare (y^2 - z)
  )

def K2b (p : ℕ) [Fact (Nat.Prime p)] : Finset (Fin 3 → ZMod p) :=
  univ.filter (fun v =>
    v 0 = 0 ∧ v 1 = 0
  )

-- Now, defining the unions is straightforward: we just apply `p` to each component.
def K2 (p : ℕ) [Fact (Nat.Prime p)] : Finset (Fin 3 → ZMod p) :=
  K2a p ∪ K2b p

def K (p : ℕ) [Fact (Nat.Prime p)] : Finset (Fin 3 → ZMod p) :=
  K1 p ∪ K2 p

/-!
# Part 2: Lemmas and Main Theorem
-/

-- We now use a `variable` block to create a context for our proofs.
variable (p : ℕ) [Fact (Nat.Prime p)]
variable (hp_odd : p > 2)


-- Inside this block, when we write `(K1 p)`, Lean uses the `p` from the context.
lemma card_K1 (hp_odd : p > 2) : (4 * (K1 p).card : ℤ) = p^3 + 2*p^2 + p := by
    have := (Fact.out : (p.Prime)).odd_of_ne_two fun and' =>by simp_all
    rcases this with ⟨x, hx⟩
    subst hx
    delta K1
    rw_mod_cast[Finset.card_filter]
    trans(4)*∑S:ZMod (2*x+1) ×_ × _,ite (IsSquare (S.1^2+4* S.2.1) ∧IsSquare (S.1^2+4* S.2.2)) (1) 0
    · exact (congr_arg _) (Fintype.sum_equiv ⟨ fun and=> (and 0, and (1), and 2), fun and=>![ _,_, _], fun and=>List.ofFn_injective rfl, fun and=>rfl⟩ _ _ (by tauto))
    · trans(4)*∑S:ZMod (2*x+1),(( Finset.univ.image fun and=> (and* and-S^2 :) /4) ×ˢ.image (@ fun and=> (and * and-S ^2 :)/4) (.univ) ).card
      · rw[ Fintype.sum_prod_type, Fintype.sum_congr _ _ fun and=>(Finset.card_filter _ _).symm.trans (congr_arg _ (Finset.ext fun and=>? _))]
        field_simp[IsSquare, sub_eq_iff_eq_add'.trans (comm),show (4 :ZMod (2*x + 1))≠0 from(( (ZMod.isUnit_iff_coprime (2) (2*x + 1)).mpr _).pow 2).ne_zero ∘mod_cast(id), mul_comm (4 :ZMod _),id]
      · push_cast[sq, add_assoc,.>., Finset.card_product, true, Finset.mul_sum]at *
        trans∑S:ZMod (2*x+1), 4*((( Finset.univ.filter fun and=>and.val≤and.val).image fun p=>(p*p-S* S)/4).card^2)
        · push_cast only [le_refl, true, Finset.filter_true_of_mem, implies_true,sq]
        · trans∑S:ZMod (2*x+1), 4*(( Finset.range (x + 1)).image fun p : ℕ=>(p*p-S* S:ZMod (2*x + 1))/4).card^2
          · refine Fintype.sum_congr _ _ fun and=>congr_arg _ ((congr_arg (.^2) (congr_arg _ (Finset.ext fun and' => Finset.mem_image.trans ⟨fun ⟨a, H, _⟩=>? _, by aesop⟩))))
            use Finset.mem_image.2 (if h: a.val≤x then⟨ _, Finset.mem_range_succ_iff.2 h,by simp_arith[*]⟩else⟨(2*x+1-a.val: ℕ),List.mem_range.2 (by valid),by norm_num[a.val_le, *]⟩)
          · rewrite[ Finset.sum_congr rfl fun and x =>(congr_arg _) ((congr_arg) ( ·^2) ( Finset.card_image_of_injOn fun and _ _ _ _ =>? _))]
            · simp_all[show (2*x+1)*(4*(x+1)^2) = (2*x+1)^3+ (2 *((2*x+1)*(2*x + 1))+ (2*x + 1))by ring]
            · field_simp[show (4 :ZMod (2 *by bound + 1))≠0 from((ZMod.isUnit_iff_coprime _ _).2 (Odd.coprime_two_left ⟨_, rfl⟩|>.mul (Odd.coprime_two_left ⟨_, rfl⟩))).ne_zero, mul_self_eq_mul_self_iff]at*
              use (by valid:).elim (by valid ∘ (ZMod.val_cast_of_lt (by valid)).symm.trans.comp (.▸ZMod.val_cast_of_lt (by valid))) (by valid ∘Nat.eq_zero_of_dvd_of_lt.comp (ZMod.natCast_zmod_eq_zero_iff_dvd _ _).1<|mod_cast.▸neg_add_cancel _)

lemma card_K2 (hp_odd : p > 2) : (2 * (K2 p).card : ℤ) = p^2 + 2*p - 1 := by
    -- # Lemma 1: Cardinality of K2a
    -- This set has p choices for y, and for each y, (p+1)/2 choices for z.
    have card_K2a_val : (K2a p).card = (p * (p + 1) / 2: ℕ) := by
      delta K2a
      trans∑S: ZMod (p),∑ A,ite (IsSquare (S^2-A ) ) (1) 0
      · rw [←@@ Finset.sum_product', Finset.card_filter]
        simp_rw [← Finset.sum_filter]
        refine Finset.sum_bij ?_ (?_) ?_ (?_) ?_
        exact (fun a s=>(a 1,a (2)))
        · simp_all only [ implies_true, Finset.mem_filter, Finset.mem_product, true, Finset.mem_univ, and_self_iff]
        · simp_all[<-List.ofFn_injective.eq_iff]
        · refine fun and μ=>⟨![0,_, _], Finset.mem_filter.2 ⟨ Finset.mem_univ _,rfl,(Finset.mem_filter.1 μ).2⟩, rfl⟩
        · exact (↑@@ fun and exists' =>rfl )
      · trans∑S:ZMod p,∑A:ZMod p,ite (IsSquare @A) (1) 0
        · exact Fintype.sum_congr _ _ fun and=> Fintype.sum_equiv (.subLeft _) _ _ fun and=>rfl
        · replace: Finset.univ.filter (IsSquare ·) =.image ((·^2 : ℕ →ZMod p)) (.range ((@(p - 1) / 2 + 1) : ℕ)) := ( Finset.ext) ?_
          · refine(Finset.sum_congr rfl fun and j=>.trans (by rw [← Finset.sum_filter, this]) (Finset.sum_image (mod_cast(?_)))).trans (.trans ( by aesop) (Nat.eq_div_of_mul_eq_right two_ne_zero ?_))
            · linear_combination p*(.div_mul_cancel ((Fact.out:p.Prime).even_sub_one (by valid)).two_dvd+.add_sub_of_le hp_odd.pos)
            · simp_rw [Finset.mem_range_succ_iff,sq_eq_sq_iff_eq_or_eq_neg,ZMod.eq_iff_modEq_nat]
              use fun and _ _ _ h=>h.elim (·.eq_of_abs_lt (max_lt_iff.2 (by valid))) (by_contra fun and=>CharP.cast_eq_zero_iff _ _ _|>.not.2 (by valid ∘p.le_of_dvd (by valid))<|mod_cast.▸neg_add_cancel _)
          · use fun and=> Finset.mem_filter.trans ⟨fun⟨A, B, _⟩=>Finset.mem_image.2 (if R: B.val≤_ then⟨ _, Finset.mem_range_succ_iff.2 R,by simp_all[sq]⟩else⟨(p-B.val: ℕ),?_⟩), by aesop⟩
            use@List.mem_range.2 (by cases (@Fact.out (p.Prime)).eq_two_or_odd with valid),by simp_arith[*, B.val_le,sq]
    -- # Lemma 2: Cardinality of K2b
    -- This set is a line, so it has p points.
    have card_K2b_val : (K2b p).card = p := by
      unfold K2b
      simp_rw [<-Fintype.card_coe, Finset.mem_filter, Finset.mem_univ, true_and]
      exact (ZMod.card p).subst (Fintype.card_congr ⟨fun ⟨a, _⟩=>a 2, fun and=>⟨fun i=>ite (i=0) 0 (ite (i =1) 0 and),rfl, rfl⟩, fun and=>Subtype.eq (List.ofFn_injective (by aesop)), fun and=>by simp_all⟩)
    -- # Lemma 3: Cardinality of the intersection of K2a and K2b
    -- This is {(0,0,z) | IsSquare(-z)}, which has (p+1)/2 points.
    have card_inter_K2a_K2b_val : (K2a p ∩ K2b p).card = ((p + 1) / 2 : ℕ) := by
      refine le_antisymm ?_ ?_
      · cases (isEmpty_or_nonempty (ZMod p))
        · cases‹IsEmpty (ZMod p)›.1 (1)
        · borelize ℝ
          simp_all![KakeyaConstruction.K2a, KakeyaConstruction.K2b]
          trans↑(Finset.univ.image (![0,0,-.^2]:ZMod p → Fin 3 →ZMod p)).card
          · use Finset.card_mono fun and μ=>Finset.mem_image.2 ((Finset.mem_filter.1 (Finset.inter_subset_left μ)).2.2.imp fun and i=>by simp_all[i.symm,←List.ofFn_injective.eq_iff,sq])
          · trans((Finset.range<|((p+1)/2 : ℕ)).image fun x : ℕ=>![0,0,-(x^2:ZMod p)]).card
            · refine Finset.card_mono (Finset.image_subset_iff.2 fun a s=> Finset.mem_image.2 (if M:_ then ⟨a.val,List.mem_range.2 M,by norm_num⟩else⟨(p-a.val : ℕ),List.mem_range.2 ? _,by norm_num[a.val_le]⟩))
              cases (@Fact.out (p.Prime)).eq_two_or_odd with valid
            · norm_num[ Finset.card_image_le.trans]
      · nontriviality ℝ
        delta KakeyaConstruction.K2a KakeyaConstruction.K2b
        trans # {x :ZMod p|IsSquare<|-x}
        · trans(( Finset.range<|((p+1)/2: ℕ)).image (-.^2 : ℕ →ZMod p)).card
          · rw[ Finset.card_image_of_injOn fun and _ _ _ _=>symm ? _, Finset.card_range]
            rw[ Finset.mem_coe, Finset.mem_range,neg_inj,sq_eq_sq_iff_eq_or_eq_neg,ZMod.eq_iff_modEq_nat]at*
            rw [← (by valid:).elim (·.eq_of_abs_lt (max_lt_iff.2 (by valid))) (by_contra fun and=>CharP.cast_eq_zero_iff _ _ _|>.not.2 (by valid ∘p.le_of_dvd (by valid))<|mod_cast.▸neg_add_cancel _),]
          · use Finset.card_mono (Finset.image_subset_iff.2 (by norm_num))
        · exact Finset.card_le_card_of_injOn (![0,0,·]) (by{norm_num}) fun and _ _ _ h =>congr_fun h (2)
    -- # Main Proof: Combine using Inclusion-Exclusion
    have h_pie_K2 : (K2 p).card = ((K2a p).card + (K2b p).card - (K2a p ∩ K2b p).card : ℕ) := by
      rw [K2]
      rw [card_union]
    rw [h_pie_K2, card_K2a_val, card_K2b_val, card_inter_K2a_K2b_val]
    norm_cast
    ring_nf
    exact (congr_arg _ ((Even.two_dvd (by norm_num[parity_simps] :Even (p+p^2))).elim (by cases(Fact.out:p.Prime).eq_two_or_odd with valid))).trans (Int.subNatNat_of_le (by valid)).symm


lemma card_K1_inter_K2 (hp_odd : p > 2) :
  (8 * ((K1 p) ∩ (K2 p)).card : ℤ) = (p:ℤ)^2 + 7*p - (p - 1) * legendreSym p (-1) := by
  -- Let L be shorthand for the Legendre symbol for convenience.
  let L := legendreSym p (-1)
  -- # Lemma 1: The cardinality of the intersection of K1 and K2b.
  -- This set simplifies to {(0,0,z) | IsSquare(z)}.
  have card_inter_K1_K2b :
    (8 * (K1 p ∩ K2b p).card : ℤ) = 4 * (p + 1) := by
    have : Fact p.Prime := ⟨Fact.out⟩
    ring_nf
    norm_cast
    apply Real.zero_lt_one.le.eq_or_lt.elim (by aesop)
    intro _
    simp_rw [K1, K2b]
    trans(Finset.univ.image (![0,0,.^2/(4:ZMod p)])).card*8
    · convert ((congr_arg₂ _) ∘congr_arg _) (Finset.ext fun and=>_) rfl
      constructor
      · use Finset.mem_inter.1 ·|>.elim fun and μ=> Finset.mem_image.2 (( Finset.mem_filter.1 and).2.2.imp fun and i=>? _)
        field_simp[i.symm,sq,←List.ofFn_inj, Finset.mem_filter.1 μ, (by ring: (4:ZMod p)=2^2),show(2 :ZMod p)≠0 from mt (CharP.cast_eq_zero_iff _ _ _).1 _,id]
        exact (mul_div_cancel_left₀ _) ((mul_self_ne_zero.2 ((CharP.cast_eq_zero_iff _ _ _).not.2 (p.not_dvd_of_pos_of_lt (by decide) (by assumption)))))
      · simp_all [show (4 :ZMod p) ≠0000 from fun and=>not_le_of_lt (by assumption) (this.out.dvd_factorial.1 (or_self_iff.mp @(this.out.dvd_mul.mp (((CharP.cast_eq_zero_iff _ _ _).mp and).trans (by decide))))), mul_div_cancel₀ _,eq_comm]
    · trans(( Finset.range ((p+1)/2 : ℕ)).image fun and : ℕ=>![0,0, (and^2/4 :ZMod p)]).card*8
      · congr 2 with
        refine Finset.mem_image.trans ⟨fun⟨A, B, _⟩=> Finset.mem_image.mpr @? _, by aesop⟩
        exact if a: A.val<_ then⟨ _,List.mem_range.2 a,by simp_arith [*]⟩else ⟨ (p-A.val : ℕ),List.mem_range.2 (by cases(this.out.eq_two_or_odd) with valid),by simp_arith [ *, A.val_le]⟩
      · rw[Finset.card_image_of_injOn fun and _ _ _ _=>?_,Finset.card_range]
        · cases this.1.eq_two_or_odd with valid
        · field_simp[sq_eq_sq_iff_eq_or_eq_neg,CharP.cast_eq_zero_iff _ p,p.not_dvd_of_pos_of_lt,<-List.ofFn_injective.eq_iff]at*
          rewrite [div_left_inj' ↑(mt ↑(CharP.cast_eq_zero_iff _ _ _).mp ↑(mt (p.primeFactors_mono) (by simp_arith [ hp_odd.ne.symm, this.elim]))), (@@sq_eq_sq_iff_eq_or_eq_neg),@@ZMod.eq_iff_modEq_nat] at *
          use (by valid:).elim (·.eq_of_abs_lt (max_lt_iff.2 (by valid))) (by_contra fun and=>((CharP.cast_eq_zero_iff _ _ _)).not.2 (by valid ∘p.le_of_dvd (by valid))<|mod_cast.▸neg_add_cancel _)
  -- # Lemma 2: The cardinality of the triple intersection K1 ∩ K2a ∩ K2b.
  -- This set simplifies to {(0,0,z) | IsSquare(z) ∧ IsSquare(-z)}.
  have card_inter_K1_K2a_K2b :
    (8 * (K1 p ∩ K2a p ∩ K2b p).card : ℤ) = 2 * (p + 3 + (p - 1) * L) := by
    nontriviality ℤ
    clear‹_=_›
    delta KakeyaConstruction.K1 KakeyaConstruction.K2a KakeyaConstruction.K2b
    norm_num[legendreSym,← Finset.filter_and, L]
    trans(8)*Nat.card (Finset.image (![0,0,·]) {x :ZMod p|IsSquare x∧IsSquare (-x)})
    · simp_rw [Nat.card_eq_finsetCard,←Fintype.card_coe, Finset.mem_image, Finset.mem_filter,Finset.mem_univ,true_and]
      congr!
      use ? _,?_
      · use@fun⟨⟨ ⟨a, A, e⟩,k, _⟩,p', _⟩=>⟨‹ Fin (3) →_› 2,⟨⟨A/2,?_⟩,by simp_all⟩,List.ofFn_injective <|by simp_all⟩
        norm_num[k,←e,div_mul_div_comm A,show(2 :ZMod p)≠0 from(CharP.cast_eq_zero_iff _ _ _).not.2 (p.not_dvd_of_pos_of_lt _ hp_odd)]
        rw[eq_div_iff fun and=>p.not_dvd_of_pos_of_lt (by decide) (hp_odd) (or_self_iff.1 ((Fact.out:p.Prime).dvd_mul.1.comp ( (CharP.cast_eq_zero_iff _ _ _).1 and).trans (by decide))), mul_comm]
      · use·.elim fun and true => true.2▸by norm_num[IsSquare.mul, true,show IsSquare (4 :ZMod p) from⟨2,_⟩]
    · rw[Nat.card_eq_finsetCard, Finset.card_image_of_injOn]
      · unfold quadraticCharFun
        trans(8)*(( Finset.image (.^2 : ℕ →ZMod p) { a ∈.range (((p-1)/2+1 : ℕ))|IsSquare (-1:ZMod p)})∪{0}).card
        · congr with
          simp_all[sq,Nat.lt_succ]
          split
          · use·.left.elim fun R M=>.inl (if j : R.val ≤_ then⟨ _, Finset.mem_range_succ_iff.mpr j,by bound⟩else⟨(p-R.val : ℕ),List.mem_range.mpr (by cases(@Fact.out (p.Prime)).eq_two_or_odd with valid),?_⟩)
            · use·.elim (·.choose_spec.2▸⟨⟨_, rfl⟩,neg_one_mul (_: ( ZMod p))▸.mul (by assumption) ⟨ _,rfl⟩⟩) (·▸by simp_arith)
            · simp_all[R.val_le]
          · use fun⟨ ⟨a, e⟩,A, B⟩=>.inr (by_contra (by valid ∘ (by use A/a,by field_simp[←e,←B,.]))),by simp_all
        · aesop
          · rw [ Finset.union_eq_left.mpr (by simp_all [Exists.intro 0]), Finset.card_image_of_injOn fun and R M A B =>? _, Finset.card_range]
            · cases inst.1.eq_two_or_odd with valid
            · rw[ Finset.mem_coe, Finset.mem_range_succ_iff,sq_eq_sq_iff_eq_or_eq_neg,ZMod.eq_iff_modEq_nat]at*
              use B.elim (·.eq_of_abs_lt (max_lt_iff.mpr (by valid))) (by_contra fun and =>CharP.cast_eq_zero_iff _ _ _ |>.not.mpr (by valid ∘p.le_of_dvd (by valid)) <|mod_cast ·▸neg_add_cancel _)
          · ring
      · use fun and _ _ A B=>congrFun B (2)
  -- # Lemma 3: The cardinality of the intersection of K1 and K2a.
  -- This is the most complex term, representing |{(0,y,z) | IsSq(y),IsSq(z),IsSq(y²-z)}|.
  -- Its value is derived to make the final calculation correct.
  have card_inter_K1_K2a :
    (8 * (K1 p ∩ K2a p).card : ℤ) = (p:ℤ)^2 + 5*p + 2 + (p - 1) * L := by
    -- # Lemma 3a: The intersection can be described as a filter.
    have h_inter_def : K1 p ∩ K2a p = univ.filter (fun v => v 0 = 0 ∧ IsSquare (v 1) ∧ IsSquare (v 2) ∧ IsSquare (v 1 ^ 2 - v 2)) := by
      ext v
      by_cases hv : v 0 = 0
      · induction π/6+7/8
        field_simp [hv,K1,K2a, and_assoc,sq]
        constructor
        · field_simp[show (4 :ZMod p)=2^2by ring,show(2 :ZMod p)≠0 from hp_odd.not_le.comp (Fact.out:p.Prime).dvd_factorial.1 ∘((CharP.cast_eq_zero_iff _ _ _).1 · |>.trans ( _))]
          field_simp+contextual[show(2 :ZMod p)≠0 from(CharP.cast_eq_zero_iff _ _ _).not.2 (p.not_dvd_of_pos_of_lt _ _)]
          field_simp+contextual[sq,show(2 :ZMod p)≠0 from(CharP.cast_eq_zero_iff _ _ _).not.2 (p.not_dvd_of_pos_of_lt _ _),IsSquare,←div_div]
          exact (fun R L A B I I=> (by·field_simp [← L,Exists.intro (A/2), ←B,show(2 :ZMod p)≠(0) from hp_odd.not_le ∘p.le_of_dvd _ ∘ ((CharP.cast_eq_zero_iff _ _ _) ).mp, false,Exists.intro (R /2)]))
        · exact (.imp ↑(.mul ⟨2,by·bound⟩) (.imp_left ↑(.mul ⟨2,by ·bound⟩)))
      · induction π
        use ?_, ?_
        · simp_all[KakeyaConstruction.K1, KakeyaConstruction.K2a]
        · aesop
    let S : Finset (ZMod p) := Finset.univ.image (·^2)
    have isSquare_iff_mem_S (x : ZMod p) : IsSquare x ↔ x ∈ S := by
      simp [IsSquare, S, Finset.mem_image, Finset.mem_univ, exists_true_left, pow_two, eq_comm]
    let S_star := S.filter (· ≠ 0)
    let CountS (A : ZMod p) := (S.filter (fun z => A - z ∈ S)).card
    have h_set_char : (K1 p ∩ K2a p).card = (∑ y ∈ S, CountS (y^2)) := by
      rw [h_inter_def]
      simp_rw [CountS, isSquare_iff_mem_S]
      convert card_biUnion _ using 2
      ext w
      change _ ↔ w ∈ S.biUnion fun x=>{v ∈ S|x^2 - v ∈ S}.image (![0,x,·])
      · simp_arith[← List.ofFn_inj, and_assoc,eq_comm]
        tauto
      · exact (Finset.card_image_of_injective _ fun v w h ↦ congr_fun h 2).symm
      · exact (fun _ _ _ _ h=> Finset.disjoint_right.2 (Finset.forall_mem_image.2 fun and m=>mt Finset.mem_image.1 (h.comp (congrFun ·.choose_spec.2 (1)))))
    -- # Lemma 3b: Split the sum over S into the zero and non-zero parts.
    have h_sum_split : (∑ y ∈ S, CountS (y ^ 2)) = CountS (0) + ∑ y ∈ S_star, CountS (y ^ 2) := by
      rw [add_comm, ← Finset.sum_filter_add_sum_filter_not S (fun x ↦ x ≠ 0)]
      exact (congr_arg _ (by(((norm_num[ <-isSquare_iff_mem_S, Finset.filter_eq'])))))
    -- # Lemma 3c: The value of CountS(A) for any non-zero square A.
    have h_CountS_sq (A : ZMod p) (hA : A ∈ S_star) :
      (4 * CountS A : ℤ) = p + 4 - L := by
        -- This follows from the standard formula for the number of solutions to x² + y² = A.
        -- We introduce this as a more fundamental lemma for AlphaProof to tackle first.
        have num_sols_is_p_minus_L (A : ZMod p) (hA_ne_zero : A ≠ 0) :
          (univ.filter (fun (v : ZMod p × ZMod p) => v.1^2 + v.2^2 = A)).card = p - legendreSym p (-1) := by
          -- Let χ be the quadratic character. The number of solutions to z²=c is 1 + χ(c).
          let χ := quadraticChar (ZMod p)
          have h_card_sq (c : ZMod p) : (univ.filter (fun x => x^2 = c)).card = 1 + χ c := by
            nontriviality
            rw [@eq_comm ℤ, ← sub_eq_zero]
            by_cases hc : c = 0
            · -- This is the correct block for the `c = 0` case.
              norm_num [hc, Finset.filter_eq']
              exact χ.map_zero
            · -- This is the correct block for the `c ≠ 0` case.
              rw [quadraticChar_apply, quadraticCharFun, add_comm, sub_eq_zero]
              by_cases hputils : IsSquare c
              · have := Fact.mk (Fact.out : p.Prime)
                obtain ⟨a, rfl⟩:=‹IsSquare c›
                norm_num[sq, mul_self_eq_mul_self_iff,eq_neg_iff_add_eq_zero, this.out.odd_of_ne_two (by omega),by assumption, mul_ne_zero_iff.1 hc, Finset.filter_or, Finset.filter_eq']
                norm_num[eq_sub_iff_add_eq.symm, Finset.filter_eq']
              · have : Finset.univ.filter (· ^ 2 = c) = ∅
                · exact Finset.filter_false_of_mem fun x _ h => False.elim <| ‹¬IsSquare c› <| h ▸ IsSquare.sq x
                · norm_num[hc,‹¬_›,this]
          -- # Lemma 1: The number of solutions equals p plus a character sum.
          have card_eq_char_sum :
            ↑((univ.filter (fun (v : ZMod p × ZMod p) => v.1 ^ 2 + v.2 ^ 2 = A)).card) =
            ↑p + ∑ x, χ (A - x ^ 2) := by
            rw [Finset.card_eq_sum_card_fiberwise (fun x _ ↦ Finset.mem_univ x), Nat.cast_sum]
            rw [Fintype.sum_prod_type, ← Finset.sum_fiberwise Finset.univ (fun x : ZMod p => x ^ 2)]
            rw [← Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun x hx => Finset.sum_subset (Finset.subset_univ (Finset.filter (fun z => z ^ 2 = A - y) .univ)) ?_]
            · nontriviality ℤ
              simp_arith +contextual [h_card_sq, Finset.filter_eq', Finset.sum_add_distrib, Finset.sum_filter]
              exact Finset.sum_comm.trans (by norm_num[mul_add,Finset.sum_add_distrib])
            · simp_arith +contextual[← eq_sub_iff_add_eq', Finset.mem_filter.mp hx,Finset.filter_eq']
          -- # Lemma 2: The character sum is equivalent to another form.
          have sum_equiv : (∑ x, χ (A - x ^ 2)) = ∑ u, χ (u * (A - u)) := by
            -- The main goal is to show that `∑_x χ(A - x²) = ∑_u χ(u(A-u))`.
            -- Our strategy is to transform the left-hand side (LHS) step-by-step until it matches the right-hand side (RHS).
            -- # Lemma 1: Re-index the sum
            -- We first change the sum over `x` to a sum over the values `c = A - x^2`.
            -- The general principle is that a sum `∑ₓ f(g(x))` can be rewritten as `∑ᵧ |{x | g(x)=y}| * f(y)`.
            -- Here, `g(x) = A - x^2` and `f = χ`.
            have h_reindexed : (∑ x, χ (A - x ^ 2)) = ∑ c, ↑((univ.filter (fun x => A - x^2 = c)).card) * χ c := by
              -- # Lemma 1: Any value `v` is equal to the sum of `v` over a set containing only that value.
              have h_self_eq_sum (x : ZMod p) : χ (A - x ^ 2) = ∑ c ∈ {A - x ^ 2}, χ c := by
                -- This is true by the `sum_singleton` lemma, which says the sum of a function over
                -- a singleton set `{a}` is just the function evaluated at `a`.
                simp
              simp_rw [h_self_eq_sum, ← Finset.sum_attach {A - _}]
              symm
              conv_rhs => rw [← Finset.sum_fiberwise_of_maps_to (fun _ _ ↦ Finset.mem_univ (A - ‹ZMod p› ^ 2))]
              refine Finset.sum_congr rfl fun y _ => ?_
              conv_rhs => rw [← Finset.sum_congr rfl (fun i hi => by rw [Finset.sum_attach, ← h_self_eq_sum, (Finset.mem_filter.mp hi).2])]
              exact (Finset.sum_const _).symm
            -- # Lemma 2: Compute the size of the fiber sets
            -- For any `c`, the number of solutions `x` to `A - x² = c` is `1 + χ(A-c)`.
            -- This is because `A - x² = c` is the same as `x² = A - c`, and the number of solutions to `x² = y` is given by `1 + χ(y)`.
            have h_card_val (c : ZMod p) : ↑((univ.filter (fun x => A - x ^ 2 = c)).card) = 1 + χ (A - c) := by
              rw [← h_card_sq]
              congr! 2
              exact Finset.filter_congr fun x _ => sub_right_injective.eq_iff' (sub_sub_self _ _)
            -- Now, we can substitute the result from Lemma 2 into the re-indexed sum from Lemma 1.
            rw [h_reindexed]
            simp_rw [h_card_val]
            -- # Main Proof
            -- Let's start by distributing the multiplication on the LHS.
            -- `(1 + χ (A - x)) * χ x` becomes `χ x + χ (A - x) * χ x`.
            have h_lhs_split : (∑ x, (1 + χ (A - x)) * χ x) = (∑ x, χ x) + (∑ x, χ (x * (A - x))) := by
              -- This follows from distributing `χ x` and then using `Finset.sum_add_distrib`.
              conv_rhs => enter [1, 2, x]
              simp_rw [one_add_mul, mul_comm (χ (A - _)), map_mul, sum_add_distrib]
            -- # Lemma 2: The sum of the quadratic character over the whole field is zero.
            have h_sum_chi_is_zero : (∑ x, χ x) = 0 := by
              have h_exists_c : ∃ c, χ c = -1 := by
                have h_exists_nonsquare : ∃ (n : ZMod p), ¬ IsSquare n := by
                  by_cases hF : Function.Injective fun x : ZMod p => x * x
                  · cases p.eq_zero_of_dvd_of_lt ((CharP.cast_eq_zero_iff _ _ _).mp (by(((linear_combination' (norm:= norm_num) (hF (neg_mul_neg 1 1).symm)))))) hp_odd
                  · rw [Finite.injective_iff_surjective] at hF
                    contrapose! hF
                    intro a
                    exact (hF a).imp fun _ => Eq.symm
                obtain ⟨c, hc_not_sq⟩ := h_exists_nonsquare
                use c
                replace hc_not_sq (i : ZMod p) : i ^ 2 ≠ c
                · rintro rfl
                  exact hc_not_sq (IsSquare.sq _)
                · refine add_left_cancel ((h_card_sq c).symm.trans ?_)
                  rw [Finset.filter_false_of_mem fun z _ => hc_not_sq z, card_empty, Nat.cast_zero, add_neg_cancel]
              -- # Main Proof:
              -- The core idea is to find a `c` such that `χ(c) = -1`, and then show that
              -- the sum `S` satisfies `S = -S`, which implies `2*S = 0`, and thus `S=0`.
              -- # Lemma 1: The sum over `χ(v)` is the same as the sum over `χ(c*v)`.
              -- Let's first get the element `c` with the special property from our hypothesis.
              obtain ⟨c, hc⟩ := h_exists_c
              have h_sum_rearranged : (∑ v ∈ univ, χ v) = ∑ v ∈ univ, χ (c * v) := by
                -- This is true because `v ↦ c * v` is a permutation of `ZMod p`.
                -- We can use `Fintype.sum_equiv` with the multiplication equivalence.
                -- This requires showing `c` is non-zero, which we know because `χ(c) = -1` (and `χ(0) = 0`).
                refine (Fintype.sum_bijective _ (mulLeft_bijective₀ c ?_) _ _ fun x ↦ rfl).symm
                intro h
                rw [h, MulChar.map_zero] at hc
                rw [eq_neg_iff_add_eq_zero, zero_add] at hc
                exact one_ne_zero hc
              -- # Main Proof Steps:
              -- Let's start with our rearrangement lemma.
              rw [h_sum_rearranged]
              -- Since χ is multiplicative, `χ(c * v) = χ(c) * χ(v)`. Let's rewrite the sum.
              simp_rw [map_mul]
              -- Now we can factor out the constant `χ(c)` from the sum.
              rw [← Finset.mul_sum]
              -- From our hypothesis, we know `χ(c) = -1`.
              rw [hc]
              conv_lhs => rw [← mul_div_cancel_left₀ (∑ v, χ v) (neg_ne_zero.mpr one_ne_zero)]
              have hzero : ∑ v : ZMod p, χ v = 0
              · -- # Main Proof Steps:
                -- We're at the fun part. Let's start with your rearrangement lemma.
                rw [h_sum_rearranged]
                -- Since χ is a multiplicative character, `χ(c * v) = χ(c) * χ(v)`. Let's apply this inside the sum.
                simp_rw [map_mul]
                -- Now we can pull the constant term `χ(c)` out of the summation.
                rw [← Finset.mul_sum]
                -- From your hypothesis `hc`, we know that `χ(c) = -1`.
                rw [hc]
                -- The goal is now `(∑ v, χ v) = -1 * (∑ v, χ v)`.
                -- An integer `S` that satisfies `S = -S` must be zero. The `eq_neg_self_iff` lemma states this.
                have h_sum_eq_neg : (∑ v, χ v) = -1 * ∑ v, χ v := by
                  calc
                    (∑ v, χ v) = ∑ v, χ (c * v)       := by rw [h_sum_rearranged]
                    _          = (∑ v, χ c * χ v)     := by simp_rw [map_mul]
                    _          = χ c * (∑ v, χ v)     := by rw [Finset.mul_sum]
                    _          = -1 * (∑ v, χ v)      := by rw [hc]
                linarith [h_sum_eq_neg]
              · rw [hzero, mul_zero, Int.zero_ediv]
                exact mul_zero (-1)
            -- # Main Proof
            -- Let's start by splitting the sum on the left-hand side.
            rw [h_lhs_split]
            -- Now, we know the first part of that sum is zero.
            rw [h_sum_chi_is_zero, zero_add]
          -- # Lemma 3: The second character sum evaluates to -χ(-1).
          have eval_char_sum : (∑ u, χ (u * (A - u))) = -χ (-1) := by
            let S := Finset.univ.filter (fun u => u ≠ 0 ∧ u ≠ A)
            have h_sum_S : (∑ u ∈ Finset.univ, χ (u * (A - u))) = ∑ u ∈ S, χ (u * (A - u)) := by
              have eq : ∑ x ∈ univ, χ (x * (A - x)) = ∑ x ∈ S, χ (x * (A - x))
              · rw [← Finset.sum_subset S.subset_univ fun u _ hu => ?_]
                rcases eq_or_ne u 0 with rfl | hu'
                · rw [zero_mul, χ.map_zero]
                · rw [show A - u = 0 by linear_combination -(not_not.mp (hu.comp (Finset.mem_filter.mpr ⟨mem_univ u, hu', ·⟩)))]
                  rw [mul_zero, quadraticChar_zero]
              · exact eq
            rw [h_sum_S]
            have h_term_rewrite (u : ZMod p) (hu : u ∈ S) : χ (u * (A-u)) = χ (A/u - 1) := by
              -- The key idea is to factor `u^2` out of the term on the left-hand side,
              -- and then use the properties of the multiplicative character `χ`.
              -- # Lemma 1: An algebraic identity
              -- We first show that `u * (A - u)` is equal to `u^2 * (A / u - 1)`.
              -- This is a simple identity in a field, valid because `u ≠ 0` (which is guaranteed by `hu : u ∈ S`).
              have h_factor : u * (A - u) = u ^ 2 * (A / u - 1) := by
                rw [mem_filter] at hu
                field_simp [hu]
                ring
              -- We can now substitute this identity into our goal.
              rw [h_factor]
              by_cases hu_zero : u = 0
              · exact ((Finset.mem_filter.mp hu).2.1 hu_zero).elim
              · rw [eq_sub_of_add_eq' (h_card_sq _).symm]
                rw [h_card_sq]
                conv_rhs => rw [← add_sub_cancel_left 1 (χ (A / u - 1)), ← h_card_sq]
                congr 1
                rw [h_card_sq]
                rw [map_mul]
                rw [map_pow, quadraticChar_sq_one hu_zero, one_mul]
            -- First, apply the rewrite from the previous step to simplify the terms within the sum.
            rw [Finset.sum_congr rfl fun u hu => h_term_rewrite u hu]
            -- The goal is now `(∑ u ∈ S, χ (A / u - 1)) = -χ (-1)`.
            -- This calls for a change of variables. Let `v = A / u - 1`.
            -- # Lemma 1: The variable change
            -- We define our new set T, explicitly stating it's a Finset of `ZMod p`.
            let T := (Finset.univ : Finset (ZMod p)).filter (fun v => v ≠ 0 ∧ v ≠ -1)
            have h_sum_T : (∑ u ∈ S, χ (A / u - 1)) = ∑ v ∈ T, χ v := by
              -- The goal is to show `∑ u in S, χ (f u) = ∑ v in T, χ v` where `f u = A/u - 1`.
              -- This can be proven by showing `f` is a bijection from S to T.
              -- We can build an equivalence `e : S ≃ T` and use `Finset.sum_equiv`.
              -- # Lemma 1: Define the function `f` and show it's a bijection between S and T.
              let f : ZMod p → ZMod p := fun u => A / u - 1
              have h_bij : Set.BijOn f S T := by
                constructor
                -- ## 1a. Show f maps S to T (f(S) ⊆ T)
                · intro u hu
                  -- `u ∈ S` means `u ≠ 0` and `u ≠ A`.
                  -- We need to show `f u ≠ 0` and `f u ≠ -1`.
                  -- `f u = 0` implies `A = u`, a contradiction.
                  -- `f u = -1` implies `A = 0`, also a contradiction.
                  simp [T, f]
                  exact (Finset.mem_filter.mp hu).2.symm.imp (sub_ne_zero.mpr ∘ div_ne_one_of_ne ∘ Ne.symm) (And.intro hA_ne_zero)
                -- ## 1b. Show f is injective on S
                · simp [Set.InjOn]
                  constructor
                  · simp_arith![f, hA_ne_zero, div_eq_mul_inv]
                  · intro y hy
                    rw [Finset.mem_coe, Finset.mem_filter] at hy
                    use A / (y + 1)
                    field_simp [hy, S, f, add_eq_zero_iff_eq_neg]
                    exact hA_ne_zero
              choose hgf hgf_spec hfg using h_bij
              exact S.sum_nbij f hgf hgf_spec hfg @fun _ _ ↦ rfl
            rw [h_sum_T]
            -- # Lemma 2: Evaluating the sum over T
            -- The sum of `χ` over all of `ZMod p` is 0. The sum over `T` is this total sum
            -- minus the terms we excluded: `χ(0)` and `χ(-1)`.
            have h_sum_T_val : (∑ v ∈ T, χ v) = -χ (-1) := by
              -- # Lemma 1: The sum of the quadratic character over all elements is zero.
              -- This is a fundamental property of non-trivial characters on a finite field.
              have sum_univ_chi_is_zero : (∑ v ∈ univ, χ v) = 0 := by
                -- This result is available in Mathlib as `MulChar.sum_univ`. You'll need to show
                -- that the quadratic character `χ` is non-trivial, which is true for p > 2.
                -- The main idea is to show the sum `S` satisfies `S = -S`, which implies `2*S = 0`, and thus `S=0`.
                -- To do this, we need to find an element `c` where `χ c = -1`.
                -- # Lemma 1: There exists an element `c` such that `χ c = -1`.
                have h_exists_c : ∃ c, χ c = -1 := by
                  -- This follows the same logic as our last proof. We just need to find a non-square element.
                  -- We can use `ZMod.exists_nonsquare` given `hp_odd`.
                  have h_exists_nonsquare : ∃ (n : ZMod p), ¬ IsSquare n := by
                    by_cases hF : Function.Injective fun x : ZMod p => x * x
                    · cases p.eq_zero_of_dvd_of_lt ((CharP.cast_eq_zero_iff _ _ _).mp (by(((linear_combination' (norm:= norm_num) (hF (neg_mul_neg 1 1).symm)))))) hp_odd
                    · rw [Finite.injective_iff_surjective] at hF
                      contrapose! hF
                      intro a
                      exact (hF a).imp fun _ => Eq.symm
                  obtain ⟨c, hc_not_sq⟩ := h_exists_nonsquare
                  use c
                  replace hc_not_sq (i : ZMod p) : i ^ 2 ≠ c
                  · rintro rfl
                    exact hc_not_sq (IsSquare.sq _)
                  · refine add_left_cancel ((h_card_sq c).symm.trans ?_)
                    rw [Finset.filter_false_of_mem fun z _ => hc_not_sq z, card_empty, Nat.cast_zero, add_neg_cancel]
                -- Let's pick out that element `c` and its property `hc`.
                obtain ⟨c, hc⟩ := h_exists_c
                -- # Lemma 2: The sum over `χ(v)` is the same as the sum over `χ(c*v)`.
                -- This is because `v ↦ c * v` is just a permutation of the elements of the field.
                have h_sum_rearranged : (∑ v ∈ univ, χ v) = ∑ v ∈ univ, χ (c * v) := by
                  -- To prove this, we can use `Fintype.sum_equiv`. We need to provide the equivalence,
                  -- which is multiplication by `c` on the left.
                  refine (Fintype.sum_bijective _ (mulLeft_bijective₀ c ?_) _ _ fun v => rfl).symm
                  rintro rfl
                  cases hc.symm.trans (by rw [quadraticChar_zero])
                -- # Main Proof:
                -- Let's start with our rearrangement identity.
                rw [h_sum_rearranged]
                -- Since χ is multiplicative, `χ (c * v) = χ c * χ v`.
                -- We can rewrite the sum using this fact.
                simp_rw [map_mul]
                -- Now we can factor out the constant `χ c` from the sum.
                rw [← Finset.mul_sum]
                -- We know from Lemma 1 that `χ c = -1`.
                rw [hc]
                -- The goal is now `(∑ v, χ v) = -1 * (∑ v, χ v)`.
                -- An integer `S` that satisfies `S = -S` must be zero.
                -- Let's give our sum a short name to make things easier to read.
                let S := ∑ i : ZMod p, χ i
                -- # Lemma 1: The sum is equal to its own negative.
                have sum_eq_neg_sum : S = -1 * S := by
                  -- We can rewrite the right-hand side using your hypotheses.
                  -- First, substitute χ c for -1.
                  rw [← hc]
                  bound
                  contrapose! h_card_sq
                  by_contra!
                  simp_rw [←sub_eq_iff_eq_add'] at this
                  norm_num[←this, S,←Nat.cast_sum,← Finset.card_eq_sum_card_fiberwise] at h_card_sq
                -- # Main Proof:
                -- Our goal is `-1 * S = 0`.
                -- From our lemma, we know `S = -1 * S`, so let's rewrite the -1 * S in our goal.
                rw [← sum_eq_neg_sum]
                linarith
              -- # Lemma 2: Relate the sum over T to the sum over the whole set.
              -- The sum over `univ` can be split into the sum over `T` and the points `0` and `-1`.
              have h_sum_split : (∑ v ∈ univ, χ v) = (∑ v ∈ T, χ v) + χ 0 + χ (-1) := by
                -- You can prove this by showing `univ` is the disjoint union of `T`, `{0}`, and `{-1}`,
                -- and then using `Finset.sum_disjUnion`.
                rw [add_assoc, ← sub_eq_iff_eq_add]
                simp_arith[T, Finset.filter_and, Finset.filter_ne']
                ring
              -- # Main Proof:
              -- We can now combine these two facts to reach our goal.
              -- Let's start with our lemma about splitting the sum.
              rw [h_sum_split] at sum_univ_chi_is_zero
              -- The character of zero is zero.
              rw [add_assoc] at sum_univ_chi_is_zero
              linear_combination sum_univ_chi_is_zero - χ.map_zero
            -- Applying this final lemma completes the proof.
            rw [h_sum_T_val]
          -- # Main proof: Chain the lemmas together.
          rw [card_eq_char_sum]
          -- Next, we use `sum_equiv` to replace the character sum with an
          -- equivalent one that we know how to evaluate.
          rw [sum_equiv]
          -- Now, we can substitute the final value of that sum using `eval_char_sum`.
          rw [eval_char_sum]
          -- The goal is now `↑p + -χ (-1) = ↑p - legendreSym p (-1)`.
          -- In Mathlib, `legendreSym p (-1)` is equal to `χ (-1)`, so this is true by definition.
          ring_nf
          refine congr_arg ((p : ℤ) - ·) ?_
          norm_cast
        -- Next, we relate the number of solutions (x,y) to CountS A, the number of z's.
        have sols_relate_to_CountS (A : ZMod p) (hA_is_sq : A ∈ S_star) :
          (univ.filter (fun (v : ZMod p × ZMod p) => v.1^2 + v.2^2 = A)).card = 4 * (CountS A) - 4 := by
          -- We partition the solution set N based on whether components are zero.
          let N := univ.filter (fun (v : ZMod p × ZMod p) => v.1^2 + v.2^2 = A)
          let N_x_zero := N.filter (fun v => v.1 = 0)
          let N_y_zero := N.filter (fun v => v.2 = 0)
          let N_xy_ne_zero := N.filter (fun v => v.1 ≠ 0 ∧ v.2 ≠ 0)
          -- The cardinality of N is the sum of the cardinalities of the partitions.
          -- First, we calculate the size of the sets where one component is zero.
          have hA_ne_zero : A ≠ 0 := (mem_filter.mp hA_is_sq).2
          have hA_is_sq' : IsSquare A := (isSquare_iff_mem_S _).mpr (mem_filter.mp hA_is_sq).1
          have card_N_x_zero : N_x_zero.card = 2 := by
            -- This follows because {y | y² = A} has 2 elements if A is a non-zero square.
            obtain ⟨a, ha⟩ := hA_is_sq'
            have : N_x_zero = {(0, a), (0, -a)}
            · ext ⟨x, y⟩
              constructor
              · intro hx
                rw [Finset.mem_filter] at hx
                obtain ⟨x', hx'⟩ := hx
                rw [Finset.mem_insert, Finset.mem_singleton]
                refine (eq_or_eq_neg_of_sq_eq_sq y a ?_).imp (congr_arg₂ _ hx') (congr_arg₂ _ hx')
                rw [mem_filter] at x'
                linear_combination x'.2.trans ha - hx' * (x : ZMod p)
              · field_simp +contextual [N_x_zero, N, or_imp, ha.symm, sq]
            · rw [this]
              apply Finset.card_pair
              intro h
              rw [Prod.mk.inj_iff] at h
              apply hp_odd.ne'
              rw [eq_neg_iff_add_eq_zero, ← two_mul] at h
              exact False.elim (mul_ne_zero ((ZMod.natCast_zmod_eq_zero_iff_dvd 2 p).not.mpr <| Nat.not_dvd_of_pos_of_lt zero_lt_two hp_odd) (left_ne_zero_of_mul (ha ▸ hA_ne_zero)) h.2)
          -- The proof for N_y_zero is symmetric.
          have card_N_y_zero : N_y_zero.card = 2 := by
            -- The proof is symmetric to the one for N_x_zero.
            -- We show N_y_zero is the image of {x | x²=A} under the map x ↦ (x,0).
            have : N_y_zero = (univ.filter (fun x => x^2 = A)).image (fun x => (x, 0)) := by
              ext ⟨x, y⟩
              rw [Finset.mem_image]
              constructor
              · intro hxy
                rw [Finset.mem_filter] at hxy
                refine ⟨x, ?_, Prod.ext rfl hxy.2.symm⟩
                convert hxy.1 using 1
                congr!
                simp_rw [N, mem_filter, mem_univ, true_and, show y = 0 from hxy.2, zero_pow two_ne_zero, add_zero]
              · rintro ⟨x, hxA, hex⟩
                rw [← hex]
                rw [Finset.mem_filter] at hxA ⊢
                exact ⟨Finset.mem_filter.mpr ⟨Finset.mem_univ _, by simp_rw [zero_pow two_ne_zero, add_zero, hxA]⟩, rfl⟩
            rw [← card_N_x_zero, this]
            induction hA_is_sq'
            rw [card_N_x_zero, (by congr : A = _ * _), ← pow_two]
            rw [Finset.card_image_of_injective _ (Prod.mk.inj_right _), card_filter]
            rw [← card_N_x_zero]
            rw [card_N_x_zero]
            rw [← card_N_x_zero]
            rw [card_N_x_zero]
            rw [← card_N_x_zero]
            simp_rw [card_N_x_zero, sq_eq_sq_iff_eq_or_eq_neg, Finset.sum_ite, Finset.sum_const, Nat.smul_one_eq_cast, Finset.filter_or, Finset.filter_eq']
            simp_arith[<- two_mul, left_ne_zero_of_mul (right_ne_zero_of_mul <|Ne.symm (h6 +mul_zero 10 |>.subst hA_ne_zero)), Finset.card_union, eq_neg_iff_add_eq_zero]
            have : ‹ZMod p› ≠ 0
            · rintro rfl
              rw [mul_zero] at *
              exact hA_ne_zero ‹_›
            · rw [Finset.card_eq_zero.mpr, tsub_zero]
              rw [Finset.singleton_inter_of_not_mem]
              rw [Finset.mem_singleton, eq_neg_iff_add_eq_zero, ← two_mul]
              exact mul_ne_zero ((ZMod.natCast_zmod_eq_zero_iff_dvd 2 p).not.mpr <| Nat.not_dvd_of_pos_of_lt zero_lt_two hp_odd) this
          -- The number of solutions is |N| = |N_x_zero| + |N_y_zero| + |N_xy_ne_zero|, since A≠0 implies the first two sets are disjoint from the third.
          -- We relate |N_xy_ne_zero| to CountS A.
          have card_N_xy_ne_zero : N_xy_ne_zero.card = 4 * (CountS A - 2) := by
            -- The number of solutions with x,y≠0 is 4 * |{z ∈ S_star \ {A} | A-z ∈ S_star}|.
            -- This is because each such z=y² gives 4 solutions (±x, ±y).
            -- We then relate this to CountS A.
            -- CountS A = |{z∈S|A-z∈S}| = |{0}|_{A∈S} + |{A}|_{A∈S} + |{z ∈ S_star \ {A} | A-z ∈ S_star}|.
            -- So, |{z ∈ S_star \ {A} | A-z ∈ S_star}| = CountS A - 2.
            -- We show |N_xy_ne_zero| = 4 * T.card and T.card = CountS A - 2, where T = {u ∈ S* | A-u ∈ S*}.
            let T := S_star.filter (fun u => A - u ∈ S_star)
            have card_T_eq_CountS_sub_2 : T.card = CountS A - 2 := by
              let C_set := S.filter (fun z => A - z ∈ S)
              have h_union : C_set = T ∪ {0, A} := by
                ext1 y
                by_cases hy_zero : y = 0
                -- This is the case y = 0. This block was moved to the end of the proof.
                · subst hy_zero
                  have : 0 ∈ S := by
                    rw [← isSquare_iff_mem_S]
                    exact .zero
                  have : A ∈ S := (isSquare_iff_mem_S _).mp hA_is_sq'
                  norm_num [C_set, *]
                -- This is the case y ≠ 0. This block was incorrectly placed in the `y = 0` branch.
                · by_cases hy_A : y = A
                  -- Case y = A (and y ≠ 0)
                  · have : y ∈ S := (isSquare_iff_mem_S y).mp <| hy_A.symm ▸ hA_is_sq'
                    -- The norm_num call simplifies the goal to proving `0 ∈ S`
                    norm_num [← hy_A, this, C_set]
                    -- This `exact` proves the remaining goal.
                    exact Finset.mem_image.mpr ⟨0, Finset.mem_univ 0, zero_pow two_ne_zero⟩
                  -- Case y ≠ A (and y ≠ 0)
                  · norm_num [C_set, T, S_star, sub_ne_zero, Ne.symm hy_A, hy_zero, hy_A]
              have h_disjoint : Disjoint T {0, A} := by simp [T, S_star]
              change #T = #C_set - 2
              rw [h_union, card_union_of_disjoint h_disjoint, card_pair hA_ne_zero.symm, Nat.add_sub_cancel_right]
            have h_cover : N_xy_ne_zero = T.biUnion (fun u => N_xy_ne_zero.filter (fun v => v.2^2 = u)) := by
              ext ⟨x, y⟩
              constructor
              · intro hxy_ne_zero
                rw [Finset.mem_biUnion]
                refine ⟨y ^ 2, ?_, ?_⟩
                · rw [mem_filter] at hxy_ne_zero ⊢
                  have hA2 := hxy_ne_zero.1
                  rw [mem_filter] at hA2
                  simp_arith [← hA2.2, S, S_star, hxy_ne_zero]
                · exact Finset.mem_filter.mpr ⟨hxy_ne_zero, rfl⟩
              · intro hxy
                obtain ⟨u, -, hu⟩ := Finset.mem_biUnion.1 hxy
                exact Finset.filter_subset _ _ hu
            have card_sq_eq_two {c : ZMod p} (hc_nz_sq : IsSquare c ∧ c ≠ 0) :
              (univ.filter (fun x => x^2 = c)).card = 2 := by
              -- First, we unpack the hypothesis `hc_nz_sq`.
              -- This gives us an element `r` such that `c = r^2` and a proof `h_nz_sq'` that `r^2 ≠ 0`.
              obtain ⟨⟨r, rfl⟩, h_nz_sq'⟩ := hc_nz_sq
              -- This small `have` block seems to be leftover from a previous edit.
              -- It correctly proves that `z^2` is definitionally equal to `z*z`, but it isn't used later in the proof.
              -- I've included it here as you asked to use all the original lines.
              have h_sq : (fun z : ZMod p => z ^ 2 = r * r) = fun z : ZMod p => z * z = r * r := by
                ext1 z
                rw [sq]
              -- The main strategy is to `convert` the goal. We state that our set of solutions is equivalent
              -- to the set `{r, -r}`, which has a cardinality of 2.
              -- The `using 2` command creates two sub-goals for us to prove this.
              convert_to #({r, -r} : Finset (ZMod p)) = 2 using 2
              · -- Goal 1: Prove that the set of solutions is indeed `{r, -r}`.
                -- The `norm_num` tactic can prove this automatically by checking set membership for all elements.
                norm_num [sq, mul_self_eq_mul_self_iff, Finset.ext_iff]
              · -- Goal 2: Prove that the cardinality of `{r, -r}` is 2.
                -- This requires proving that `r` and `-r` are distinct elements.
                have : r ≠ -r
                · -- This block proves `r ≠ -r`
                  rw [Ne, eq_neg_iff_add_eq_zero, ← two_mul]
                  refine mul_ne_zero ?_ (left_ne_zero_of_mul h_nz_sq')
                  -- This holds because `p` is an odd prime, so 2 is not a factor of p.
                  exact CharP.cast_eq_zero_iff _ p 2 |>.not.mpr (Nat.not_dvd_of_pos_of_lt Nat.zero_lt_two hp_odd)
                -- Now, use the fact that `r ≠ -r` to show the cardinality is 2.
                rw [card_insert_of_not_mem (not_mem_singleton.2 this), card_singleton]
            have h_card_fiber (u : ZMod p) (hu : u ∈ T) :
              (N_xy_ne_zero.filter (fun v => v.2^2 = u)).card = 4 := by
              -- # Lemma 1: The condition `hu` implies `u` is a non-zero square.
              have u_is_nz_sq : IsSquare u ∧ u ≠ 0 := by
                -- The proof follows from the definitions of T and S_star.
                exact (Finset.mem_filter.mp (Finset.filter_subset _ _ hu)).imp_left (isSquare_iff_mem_S _).mpr
              -- # Lemma 2: The condition `hu` also implies `A - u` is a non-zero square.
              have A_minus_u_is_nz_sq : IsSquare (A - u) ∧ A - u ≠ 0 := by
                -- This also follows directly from `hu : u ∈ T`.
                rw [mem_filter, mem_filter] at hu
                exact (Finset.mem_filter.mp hu.2).imp_left (isSquare_iff_mem_S _).mpr
              -- # Lemma 3: We can rewrite the set we are counting as a Cartesian product.
              have h_set_eq_product :
                (N_xy_ne_zero.filter (fun v => v.2 ^ 2 = u)) =
                (univ.filter (fun x => x ^ 2 = A - u)) ×ˢ (univ.filter (fun y => y ^ 2 = u)) := by
                -- This can be proven by showing that an element is in the left-hand set
                -- if and only if it is in the right-hand set (`ext` tactic).
                -- You'll need to use the definitions of N_xy_ne_zero and the facts from the first two lemmas.
                -- To prove the two sets are equal, we'll show they have the same elements.
                -- The `ext` tactic lets us pick an arbitrary element `⟨x, y⟩` and prove it's in one set iff it's in the other.
                ext ⟨x, y⟩
                -- We can simplify both sides of the `iff` by unfolding the definitions.
                -- `mem_filter` and `mem_product` are the main ones.
                constructor
                · intro h
                  -- h contains our four assumptions.
                  -- We need to show `x^2 = A - u` and `y^2 = u`.
                  -- `y^2 = u` is already one of our assumptions.
                  -- To get `x^2 = A - u`, just substitute `y^2 = u` into `x^2 + y^2 = A`.
                  obtain ⟨h1, h2⟩ := Finset.mem_filter.1 h
                  subst h2
                  rw [mem_product]
                  norm_num[← (mem_filter.1 (Finset.filter_subset _ _ h1)).2]
                -- Now, let's prove the backward direction (←).
                -- We assume the right side and prove the left.
                intro h
                obtain ⟨hx, hy⟩ := Finset.mem_product.mp h
                norm_num [N_xy_ne_zero, N, Finset.mem_filter.1 hx, Finset.mem_filter.1 hy, *, ← sq_pos_iff]
                rw [mem_filter] at hx hy
                constructor
                · rintro rfl
                  exact A_minus_u_is_nz_sq.2 (hx.2.symm.trans <| zero_pow two_ne_zero)
                · intro rfl
                  rw [rfl, zero_pow two_ne_zero] at hy
                  exact u_is_nz_sq.2 hy.2.symm
              -- # Main Proof: Now, we use the lemmas to calculate the cardinality.
              -- We can rewrite our goal using the equality from Lemma 3.
              rw [h_set_eq_product]
              -- The cardinality of a product of finsets is the product of their cardinalities.
              rw [card_product]
              -- Now, we use our existing lemma `card_sq_eq_two` on both parts of the product,
              -- using the facts we proved in the first two lemmas.
              rw [card_sq_eq_two u_is_nz_sq, card_sq_eq_two A_minus_u_is_nz_sq]
              -- The rest is simple arithmetic.
            rw [h_cover, Finset.card_biUnion, Finset.sum_congr rfl h_card_fiber, Finset.sum_const, smul_eq_mul, mul_comm, card_T_eq_CountS_sub_2]
            refine fun v hv w hw huv => Finset.disjoint_filter.mpr fun x hx hx' => ?_
            rwa [hx']
          -- We partition the set of solutions N based on whether any coordinate is zero.
          let N_some_zero := N.filter (fun v => v.1 = 0 ∨ v.2 = 0)
          have h_part : N.card = N_some_zero.card + N_xy_ne_zero.card := by
             have : #N = #N_some_zero + #N_xy_ne_zero
             · rw [← card_union_of_disjoint]
               · congr
                 ext ⟨x, y⟩
                 refine ⟨fun h => ?_, fun h => ?_⟩
                 · rw [← filter_or]
                   refine Finset.mem_filter.mpr ⟨h, ?_⟩
                   tauto
                 · rcases mem_union.1 h with h | h
                   · exact Finset.filter_subset _ _ h
                   · exact Finset.filter_subset _ _ h
               · rw [Finset.disjoint_filter]
                 rintro ⟨v, w⟩ _ hv ⟨hx, hy⟩
                 exact hv.elim hx hy
             · exact this
          rw [h_part, card_N_xy_ne_zero]
          -- Now we show N_some_zero.card = 4, since it's the disjoint union of N_x_zero and N_y_zero (which have no intersection as A≠0).
          have card_N_some_zero : N_some_zero.card = 4 := by
            simp_rw [N_some_zero, filter_or]
            rw [card_union_of_disjoint, card_N_x_zero, card_N_y_zero]
            rw [Finset.disjoint_iff_inter_eq_empty]
            rw [← Finset.filter_and, Finset.filter_eq_empty_iff]
            intro x hx hxA
            rw [mem_filter] at hx
            exact hA_ne_zero (by rw [← hx.2, hxA.1, hxA.2, zero_pow two_ne_zero, zero_add])
          rw [card_N_some_zero, Nat.mul_sub_left_distrib]
          ring_nf
          -- # Lemma 1: Prove the hidden condition
          -- The reason this puzzle is solvable is that your context implies `CountS A ≥ 2`.
          have h_ge_2 : CountS A ≥ 2 := by
            -- How to prove this? A huge hint is the hypothesis you already have:
            -- `card_T_eq_CountS_sub_2 : #T = CountS A - 2`
            -- Since cardinalities (#T) can't be negative, this equality implies `CountS A` must be at least 2.
            -- You can use `Nat.sub_eq_zero_iff_le` or similar lemmas to formalize this.
            -- # Lemma 1: Equate the two formulas for the total number of solutions.
            -- We have one formula from partitioning the solutions (`h_part` and its consequences)
            -- and another from the character sum evaluation (`num_sols_is_p_minus_L`).
            have h_card_eq : ↑p - L = ↑(#N_some_zero) + ↑(#N_xy_ne_zero) := by
              -- We know ↑#N = ↑p - L from `num_sols_is_p_minus_L`.
              -- We also know #N = #N_some_zero + #N_xy_ne_zero from `h_part`.
              -- We can rewrite the first equation using the second.
              rw [← num_sols_is_p_minus_L, h_part]
              -- We need `A ≠ 0` to use this lemma, which we have from `hA_ne_zero`.
              · norm_cast
              exact hA_ne_zero
            -- # Lemma 2: Substitute the known cardinalities to create an equation for `CountS A`.
            -- We know `#N_some_zero = 4` and `#N_xy_ne_zero = 4 * (CountS A - 2)`.
            have h_counts_eq : ↑p - L = 4 + ↑(4 * (CountS A - 2)) := by
              -- Start with the result from Lemma 1.
              rw [h_card_eq]
              -- Substitute the known cardinalities.
              rw [card_N_some_zero, card_N_xy_ne_zero]
              -- We just need to do a little algebra on the coercions.
              simp [Nat.cast_mul]
            -- # Lemma 3: The expression `p - L` is always at least 4 for an odd prime p.
            have h_p_minus_L_ge_4 : ↑p - L ≥ 4 := by
              -- L can only be 1 or -1. We can prove this by cases.
              -- You can use `legendreSym.eq_one_or_neg_one` to handle the cases for L.
              exact (le_add_of_nonneg_right (Nat.cast_nonneg _)).trans h_counts_eq.ge
            classical let M := Finset.image (fun x : ZMod p × ZMod p => (x.1, x.2)) (Finset.univ.filter fun x => x.1 * x.2 = A) -- This used to be rw [Finset.card_image
            rw [mem_filter] at hA_is_sq
            by_contra! h
            apply p.not_dvd_of_pos_of_lt Nat.zero_lt_two hp_odd
            use 1
            rw [mul_one]
            have := Finset.card_le_one_iff_subset_singleton.mp (Nat.le_of_lt_succ h)
            obtain ⟨z, hz⟩ := this
            have hz0 : z ≠ 0
            · rintro rfl
              rw [Finset.subset_singleton_iff] at hz
              rcases hz with hz | hz
              · rw [Finset.eq_empty_iff_forall_not_mem] at hz
                norm_num at hz
                have hA_zero : (0 : ZMod p) ∈ S
                · rw [← isSquare_iff_mem_S]
                  exact .zero
                · exact hz 0 hA_zero (sub_zero A |>.symm ▸ hA_is_sq.1)
              · rw [Finset.eq_singleton_iff_unique_mem] at hz
                have := hz.2 (-A)
                rw [Finset.mem_filter] at this
                rw [sub_neg_eq_add, ← two_mul] at this
                contrapose! hz
                intro
                norm_num[<-isSquare_iff_mem_S]
                exact ⟨A,by simp_arith[hA_is_sq',hA_ne_zero]⟩
            · contrapose! hz0
              refine (mem_singleton.mp (hz ?_)).symm
              rw [Finset.mem_filter, ← isSquare_iff_mem_S]
              rw [sub_zero]
              exact ⟨.zero, hA_is_sq.1⟩
          zify
          have h_le1 : 8 ≤ CountS A * 4 := by
            -- `h_ge_2` is `2 ≤ CountS A`. We can multiply both sides by 4.
            linarith [h_ge_2]
          have h_le2 : 4 ≤ CountS A * 4 := by
            -- This also follows easily from `CountS A ≥ 2`.
            linarith [h_ge_2]
          rw [Nat.cast_sub h_le1, Nat.cast_sub h_le2]
          rw [sub_eq_add_neg, sub_eq_add_neg]
          rw [← add_assoc]
          rw [add_comm 4]
          rw [add_assoc]
          norm_num
        rw [add_sub_right_comm, ← num_sols_is_p_minus_L A (mem_filter.mp hA).right, ((by assumption :) A hA)]
        rw [Int.ofNat_sub]
        · norm_cast
          rw [Int.subNatNat_of_le (Nat.le_mul_of_pos_right 4 _)]
          · rw [Nat.cast_sub]
            exact (sub_add_cancel _ _).symm
            · apply Nat.le_mul_of_pos_right
              rw [Finset.mem_filter] at hA
              apply Finset.card_pos.mpr
              use 0
              rw [Finset.mem_filter]
              exact ⟨Finset.mem_image.mpr ⟨0, Finset.mem_univ _, zero_pow two_ne_zero⟩, sub_zero A |>.symm ▸ hA.1⟩
          · rw [mem_filter] at hA
            apply Finset.card_pos.mpr
            use 0
            rw [Finset.mem_filter]
            rw [← isSquare_iff_mem_S]
            field_simp [hA]
        · cases mem_filter.mp hA
          apply Nat.le_mul_of_pos_right
          apply Finset.card_pos.mpr
          use 0
          norm_num[S,‹A ∈S›]
          apply Finset.mem_image.mp @‹_›|>.imp fun and=> And.right
    -- # Lemma 3d: The value of CountS(0).
    have h_CountS_zero :
      (4 * CountS 0 : ℤ) = p + 3 + (p - 1) * L := by
      -- This counts z where both z and -z are squares.
      apply mul_left_cancel₀ two_ne_zero
      ring_nf
      rw[mul_comm]
      norm_cast at*
      contrapose! card_inter_K1_K2a_K2b
      intro
      simp_rw [Int.subNatNat_of_le hp_odd.pos]at*
      apply‹¬_›
      convert‹_› using 1
      · field_simp [h_inter_def, S,CountS,isSquare_iff_mem_S,List.map]
        apply Finset.card_bij
        show∀ a ∈_,![0,(0),a] ∈_
        · norm_num+contextual[KakeyaConstruction.K2b]
        · exact fun and K V I I=>congrFun I (2)
        · simp_rw [Finset.mem_inter,Finset.mem_filter,Finset.mem_univ,true_and]
          rintro x⟨ ⟨a, ⟨⟩,@c, H, E⟩⟩
          cases(eq_or_ne (x ↑1)) 0
          · use x (↑2),⟨ Finset.mem_image.mpr (by use c,by simp_arith), H,E▸by simp_arith[*]⟩,List.ofFn_injective (by field_simp [ *])
          · cases eq_or_ne (x 02) 0
            · simp_all[KakeyaConstruction.K2b]
            · simp_rw [KakeyaConstruction.K2b]at*
              norm_num[*]at‹(x) ∈_›
      · exact (symm (.trans (by rw [Nat.cast_pred hp_odd.pos]) (by exact show (2 * (p+3+(p-1)*L) = 6+p*2+ _)by ring)))
    -- # Lemma 3e: The sum over S_star is constant.
    have h_sum_S_star_const : (∑ y ∈ S_star, CountS (y ^ 2)) = S_star.card * CountS (1) := by
      rcases π+2
      convert S_star.sum_const _
      linarith[h_CountS_sq (by assumption^2) (by norm_num[S_star, S,le_of_lt hp_odd,Exists.intro ‹ZMod p›,Finset.mem_filter.1 ‹_›]),h_CountS_sq (1) (by norm_num[S_star, S])]
    -- # Lemma 3f: Final calculation.
    have h_final_calc : (8 * (↑(CountS 0) + ↑(S_star.card) * ↑(CountS 1)) : ℤ) = (p:ℤ)^2 + 5*p + 2 + (p-1)*L := by
      have h_S_star_card : (S_star.card : ℤ) = ((p - 1) / 2 : ℤ) := by
        -- This requires showing the number of non-zero quadratic residues is (p-1)/2.
        -- We introduce a lemma for the cardinality as a natural number.
        have card_S_star_nat : S_star.card = (p - 1) / 2 := by
          -- Let D be the set of non-zero elements in ZMod p. Its card is p-1.
          let D := univ.filter (fun (x : ZMod p) => x ≠ 0)
          have h_card_D : D.card = p - 1 := by
            suffices : D = Finset.univ.erase 0
            · rw [this, Finset.card_erase_of_mem (Finset.mem_univ _)]
              simp_rw [Finset.card_univ, ZMod.card]
            · ext
              rw [Finset.mem_filter, Finset.mem_erase, and_comm, ne_comm]
          -- Show the image of D under the squaring map is exactly S_star.
          have h_image_is_S_star : D.image (fun x => x^2) = S_star := by
            ext a; simp [S_star, S, mem_image, mem_filter]
            constructor
            · rintro ⟨x, hx, rfl⟩
              refine ⟨⟨x, rfl⟩, ?_⟩
              exact pow_ne_zero _ (mem_filter.mp hx).2
            · rintro ⟨⟨a, rfl⟩, ha⟩
              rw [sq_eq_zero_iff] at ha
              refine ⟨a, mem_filter.mpr ⟨Finset.mem_univ _, ha⟩, rfl⟩
          have h_card_nonpos_Sal : #S_star ≤ (p - 1)
          · rw [← h_image_is_S_star, ← h_card_D]
            exact card_image_le
          · apply Nat.eq_div_of_mul_eq_right (by norm_num)
            apply Decidable.not_not.mp
            push_neg
            apply (Nat.mul_right_inj (by norm_num : 4 ≠ 0)).mp
            rw [← h_card_D, ← h_image_is_S_star, ← Nat.card_eq_finsetCard]
            rw [Nat.card_eq_finsetCard, h_image_is_S_star]
            rw [← h_image_is_S_star, ← mul_assoc, ← Int.natCast_inj]
            congr 1
            rw [mul_right_comm]
            rw [mul_assoc]
            linear_combination (norm := ring_nf)
            have h0 := Nat.sub_le_sub_right hp_odd.le 1
            rw [← Nat.mul_right_inj (by norm_num : 8 ≠ 0)]
            linear_combination (norm := ring_nf)
            congr 1
            linear_combination (norm := ring_nf)
            congr 1
            linear_combination (norm := ring_nf)
            congr 1
            linear_combination (norm := ring_nf)
            congr 1
            linear_combination (norm := ring_nf)
            congr 1
            linear_combination (norm := ring_nf)
            congr 1
            linear_combination (norm := ring_nf)
            congr 1
            (inhabit Int)
            linear_combination (norm := ring_nf)
            have h := (D.card_eq_sum_card_fiberwise fun x hx => Finset.mem_image_of_mem (· ^ 2) hx).symm
            congr 4
            rw [← h]
            refine symm.comp ( Finset.sum_congr rfl (Finset.forall_mem_image.mpr fun and ha => ? _)).trans ↑( Finset.sum_const _)
            suffices : Finset.filter (fun z : ZMod p => z ^ 2 = and ^ 2) D = {and, -and}
            · rw [this, card_insert_of_not_mem, card_singleton]
              · ring
              · field_simp[←two_mul,eq_neg_iff_add_eq_zero,Finset.mem_filter.1 ha,show(2 :ZMod p)≠0 from mt (CharP.cast_eq_zero_iff _ _ _).1 (p.not_dvd_of_pos_of_lt _ _)]
            · norm_num [D, Finset.ext_iff,( Finset.mem_filter.1 ha).2,sq_eq_sq_iff_eq_or_eq_neg.eq, and_iff_right_of_imp _]
        rw [card_S_star_nat]
        rw [← Nat.cast_pred (Nat.zero_lt_of_lt hp_odd)]
        rw [Int.natCast_div]
        norm_cast
      have h_one : (1 : ZMod p) ∈ S_star
      · norm_num [S_star, S]
      · rw [h_S_star_card]
        rw [show 8 = ((4 : ℕ) : ℤ) * 2 by omega]
        have : 2 ∣ (p : ℤ) - 1
        · omega
        · by_cases H : (p : ℤ) = 2
          · exfalso
            omega
          · ring_nf
            linear_combination(norm := ring_nf) h_CountS_zero * 2 + Int.ediv_mul_cancel this * (CountS 1 * 4) + (p - 1) * h_CountS_sq 1 h_one
    -- Now, prove the main goal using the intermediate lemmas.
    rw [h_set_char, h_sum_split, h_sum_S_star_const]
    rw [Nat.cast_add, Nat.cast_mul]
    exact h_final_calc
  have h_pie : (K1 p ∩ K2 p).card = ((K1 p ∩ K2a p).card + (K1 p ∩ K2b p).card - (K1 p ∩ K2a p ∩ K2b p).card : ℕ) := by
    rw [le_antisymm_iff]; borelize ℂ
    constructor
    · delta KakeyaConstruction.K1 KakeyaConstruction.K2
      simp_arith only[Nat.add_sub_cancel, Finset.inter_union_distrib_left, Finset.card_union_le, Finset.inter_assoc]
      norm_num[Nat.eq_sub_of_add_eq ( Finset.card_union_add_card_inter _ _)]
      exact (tsub_le_iff_right.1) (tsub_le_tsub_left (Finset.card_mono fun and μ=>by simp_all only [ Finset.mem_inter, Finset.mem_filter, and_self]) _)
    · delta KakeyaConstruction.K1 KakeyaConstruction.K2a KakeyaConstruction.K2b KakeyaConstruction.K2
      simp_rw [ Finset.inter_union_distrib_left, Finset.card_union]
      use ((congr_arg _) ((congr_arg _) (Finset.ext ↑(by simp_all)))).ge
  rw [(by assumption :)]; rewrite [Nat.cast_sub]
  · push_cast; linear_combination (card_inter_K1_K2a + card_inter_K1_K2b) - card_inter_K1_K2a_K2b
  · use (Finset.card_le_card Finset.inter_subset_left).trans <| le_self_add

theorem kakeya_set_size  (hp_odd : p > 2) :
  (8 * (K p).card : ℤ) = 2*p^3 + 7*p^2 + 3*p - 4 + (p - 1) * legendreSym p (-1) := by
  let L := legendreSym p (-1)
  have h_PIE : (K p).card = ((K1 p).card + (K2 p).card - ((K1 p) ∩ (K2 p)).card : ℕ) := by
    rw [card]
    rw [K]
    exact card_union _ _
  -- # Lemma to handle integer arithmetic
  have h_calc_int : (8 * (K p).card : ℤ) = (8 * (K1 p).card : ℤ) + (8 * (K2 p).card : ℤ) - (8 * ((K1 p) ∩ (K2 p)).card : ℤ) := by
    rw [h_PIE]
    -- We need to handle potential subtraction on natural numbers carefully before casting.
    -- This requires showing the intersection is smaller than the sum.
    rw [Nat.cast_sub]
    · zify [← mul_add, ← mul_sub]
    · contrapose! h_PIE
      rw [tsub_eq_zero_of_le h_PIE.le]
      rw [add_comm] at h_PIE
      intro h
      rw [add_comm] at h_PIE
      (inhabit ℕ)
      apply and_self_iff.mp
      aesop
      simp_all[KakeyaConstruction.K,KakeyaConstruction.K2,Set.ext_iff,Fact.mk]
  -- Substitute the calculated values for each term.
  rw [h_calc_int]
  rw [card_K1_inter_K2 p hp_odd]
  have h8_K1 : (8 * (K1 p).card : ℤ) = 2 * (p^3 + 2*p^2 + p) := by
    rw [show (8:ℤ) = 2 * 4 by norm_num, mul_assoc]
    rw [card_K1]
    exact hp_odd
  rw [h8_K1]
  have h8_K2 : (8 * (K2 p).card : ℤ) = 4 * (p^2 + 2*p - 1) := by
    rw [show (8:ℤ) = 4 * 2 by norm_num, mul_assoc]
    rw [card_K2]
    exact hp_odd
  rw [h8_K2]
  -- The goal is now a pure algebraic identity. We prove it by breaking it down.
  -- Let RHS := 2*(p^3 + 2*p^2 + p) + 4*(p^2 + 2*p - 1) - (p^2 + 7*p - (p-1)*L)
  -- Let LHS := 2*p^3 + 7*p^2 + 3*p - 4 + (p - 1) * L
  -- Goal: LHS = RHS
  -- # Lemma: Expand the RHS terms
  have h_rhs_expanded :
    2*(p^3 + 2*p^2 + p) + 4*(p^2 + 2*p - 1) - (p^2 + 7*p - (p-1)*L) =
    (2*p^3 + 4*p^2 + 2*p) + (4*p^2 + 8*p - 4) - (p^2 + 7*p - (p-1)*L) := by
    ring
  rw [h_rhs_expanded]
  -- # Lemma: Collect coefficients for each power of p and for L
  have h_coeffs_collected :
    (2*p^3 + 4*p^2 + 2*p) + (4*p^2 + 8*p - 4) - (p^2 + 7*p - (p-1)*L) =
    (2*p^3) + (4*p^2 + 4*p^2 - p^2) + (2*p + 8*p - 7*p) + (-4) + (p-1)*L := by
    ring_nf
  rw [h_coeffs_collected]
  -- # Lemma: Simplify each coefficient group
  have h_p3_simplified : (2*p^3) = 2*p^3 := by
      rfl
  have h_p2_simplified : (4*p^2 + 4*p^2 - p^2 : ℕ) = 7*p^2 := by
      omega
  have h_p1_simplified : (2*p + 8*p - 7*p : ℕ) = 3*p := by
      omega
  ring

#print axioms kakeya_set_size
end KakeyaConstruction
