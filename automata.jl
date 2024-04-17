struct State
    name::String
end

struct DeterministicAutomata
    Q::Vector{State}
    A::String
    δ
    q0::State
    F::State
end


A = State("A")
DeterministicAutomata([A], "abc", "", A, A)
