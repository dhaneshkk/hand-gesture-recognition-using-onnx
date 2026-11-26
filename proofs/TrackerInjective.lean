/-
  Formal model of `utils/tracker.py :: greedy_match`.

  Python: candidate (track, detection) pairs within `max_distance` are sorted by
  distance; a pair is accepted iff neither its track nor its detection has
  already been used. Here the sorted candidate list is `ps`, and acceptance is
  `step`. (Accepted pairs are prepended rather than appended; the set of
  accepted pairs is the same.)

  Proven:
    * `greedy_injective` : no track index and no detection index appears twice
      in the output, i.e. the matching is one-to-one, so `HandTracker.update`
      can never give two detections the same existing track id. (New tracks get
      fresh ids from a strictly increasing counter, so they cannot collide either.)
    * `greedy_sound`     : every accepted pair is one of the candidates, so the
      `max_distance` gate applied when building the candidates still holds.

  Core Lean 4 only (no Mathlib). Check with:  lean proofs/TrackerInjective.lean
-/

namespace Tracker

abbrev Pair := Nat × Nat   -- (track index, detection index)

def conflicts (p q : Pair) : Bool := q.1 == p.1 || q.2 == p.2

def step (acc : List Pair) (p : Pair) : List Pair :=
  if acc.any (conflicts p) then acc else p :: acc

def greedy (ps : List Pair) : List Pair := ps.foldl step []

/-- One-to-one invariant: track and detection projections are duplicate-free. -/
def Inv (acc : List Pair) : Prop :=
  (acc.map Prod.fst).Nodup ∧ (acc.map Prod.snd).Nodup

theorem step_inv (acc : List Pair) (p : Pair) (h : Inv acc) : Inv (step acc p) := by
  unfold step
  split
  · exact h
  · next hany =>
    have hfree : ∀ q ∈ acc, q.1 ≠ p.1 ∧ q.2 ≠ p.2 := by
      intro q hq
      have hq' : conflicts p q = false := by
        cases hc : conflicts p q with
        | false => rfl
        | true => exact absurd (List.any_eq_true.mpr ⟨q, hq, hc⟩) hany
      simp only [conflicts, Bool.or_eq_false_iff, beq_eq_false_iff_ne] at hq'
      exact hq'
    obtain ⟨h1, h2⟩ := h
    refine ⟨?_, ?_⟩
    · simp only [List.map_cons, List.nodup_cons, List.mem_map]
      refine ⟨?_, h1⟩
      rintro ⟨q, hq, hq1⟩
      exact (hfree q hq).1 hq1
    · simp only [List.map_cons, List.nodup_cons, List.mem_map]
      refine ⟨?_, h2⟩
      rintro ⟨q, hq, hq2⟩
      exact (hfree q hq).2 hq2

theorem foldl_inv : ∀ (ps acc : List Pair), Inv acc → Inv (ps.foldl step acc)
  | [], _, h => h
  | p :: ps, acc, h => foldl_inv ps (step acc p) (step_inv acc p h)

theorem greedy_injective (ps : List Pair) : Inv (greedy ps) :=
  foldl_inv ps [] ⟨List.nodup_nil, List.nodup_nil⟩

theorem mem_step {acc : List Pair} {p q : Pair} (h : q ∈ step acc p) : q ∈ acc ∨ q = p := by
  unfold step at h
  split at h
  · exact Or.inl h
  · rcases List.mem_cons.mp h with h' | h'
    · exact Or.inr h'
    · exact Or.inl h'

theorem mem_foldl : ∀ (ps acc : List Pair) (q : Pair),
    q ∈ ps.foldl step acc → q ∈ acc ∨ q ∈ ps
  | [], _, _, h => Or.inl h
  | p :: ps, acc, q, h => by
    rcases mem_foldl ps (step acc p) q h with h' | h'
    · rcases mem_step h' with h'' | h''
      · exact Or.inl h''
      · exact Or.inr (List.mem_cons.mpr (Or.inl h''))
    · exact Or.inr (List.mem_cons.mpr (Or.inr h'))

theorem greedy_sound (ps : List Pair) (q : Pair) (h : q ∈ greedy ps) : q ∈ ps := by
  rcases mem_foldl ps [] q h with h' | h'
  · simp at h'
  · exact h'

end Tracker
