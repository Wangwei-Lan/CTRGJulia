include("../../2DClassical/partition.jl")
using TensorOperations
using KrylovKit
using Arpack
using LinearAlgebra


function ApplyVT(VT::Array{Float64},v::Array{Float64})
    @tensor v[-1,-2] := VT[1,4,-1,2]*VT[3,2,-2,4]*v[1,3]
    return v
end



function FreeEnergyDensity(T::Array{Float64})
    chi = size(T,1)
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(T,y),rand(chi,chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(T,[3,2,1,4]),y),rand(chi,chi),1)
    eigup = reshape(eigvecsup[1],chi^2)
    eigdn = reshape(eigvecsdn[1],chi^2)
    @tensor Z1[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*T[1,4,5,2]*T[3,2,6,4]
    return Z1[1]/LinearAlgebra.dot(eigup,eigdn)
end


function ConstructBaseTensor(Beta::Float64)
    Q = Float64[exp(Beta) exp(-Beta);
                exp(-Beta) exp(Beta)]
    g = zeros(2,2,2,2)
    g[1,1,1,1] = g[2,2,2,2] = 1
    gp = zeros(2,2,2,2)
    gp[1,1,1,1] = -1;gp[2,2,2,2] = 1
    @tensor T[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*g[1,2,3,4]
    return T
end



BetaExact =1/2*log(1+sqrt(2))
Beta = 0.9994*BetaExact
T = ConstructBaseTensor(Beta)


chi = size(T,1)
FEextrapolate = []
Z = FreeEnergyDensity(T)
FE = log(Z)/2
append!(FEextrapolate,[FE])

@tensor TT[-1,-2,-3,-4,-5,-6] := T[-1,-3,-4,1]*T[-2,1,-5,-6]
TT = reshape(TT,4,2,4,2)

Z = FreeEnergyDensity(TT)
FE = log(Z)/4
append!(FEextrapolate,[FE])

@tensor TTT[-1,-2,-3,-4,-5,-6] := TT[-1,-3,-4,1]*T[-2,1,-5,-6]
TTT = reshape(TTT,8,2,8,2)
Z = FreeEnergyDensity(TTT)
FE = log(Z)/6
append!(FEextrapolate,[FE])

@tensor T4[-1,-2,-3,-4,-5,-6] := TTT[-1,-3,-4,1]*T[-2,1,-5,-6]
T4 = reshape(T4,16,2,16,2)
Z = FreeEnergyDensity(T4)
FE = log(Z)/8
append!(FEextrapolate,[FE])


feexact =  ComputeFreeEnergy(Beta)
T = ConstructBaseTensor(Beta)
Ttemp = T
FEextrapolate = Array{Float64}(undef,0)
for j in 2:11
    @tensor Ttemp[-1,-2,-3,-4,-5,-6] := Ttemp[-1,-3,-4,1]*T[-2,1,-5,-6]
    Ttemp = reshape(Ttemp,2^j,2,2^j,2)
    Z = FreeEnergyDensity(Ttemp)
    FE = log(real(Z))/(2*j)
    append!(FEextrapolate,[FE])
end
