include("../../2DClassical/partition.jl")
using TensorOperations
using LinearAlgebra
using Arpack
using KrylovKit
#--- Debug Parameter


#------- function ordered eigen value decomposition
function eigenorder(A::Array{Float64})
    F = eigen(A)
    order = sortperm(F.values,rev=true,by=abs)
    return F.vectors[:,order],F.values[order]
end

function ApplyAT(T::Array{Float64},A::Array{Float64},x::Array{Float64})
    @tensor x[-1,-2] := x[1,2]*T[1,4,-1,3]*A[2,3,-2,4]
    x = x/maximum(x)
    return x
end

x = 1

function HMover(HT::Array{Float64},T::Array{Float64},chimax::Int64)

    chi = size(HT,2);dlink = size(T,1)
    @tensor temp[-1,-2,-3,-4,-5,-6] := HT[6,-2,8,5]*HT[7,-5,9,5]*T[1,-1,6,2]*T[1,-4,7,2]*
                            T[8,-3,3,4]*T[9,-6,3,4]
    #@tensor temp[-1,-2,-3,-4,-5,-6] := HT[6,-2,8,5]*HT[7,5,9,-5]*T[1,-1,6,2]*T[1,2,7,-4]*
    #                        T[8,-3,3,4]*T[9,4,3,-6]
    temp = reshape(temp,chi*dlink^2,chi*dlink^2)
    println(maximum(temp-temp'))
    F = svd((temp+temp')/2)
    append!(SuMatrix,[F.S])
    if chi*dlink^2 > chimax
        Ph = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax)
    else
        Ph = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    end
    return Ph
end

function VMover(VT::Array{Float64},T::Array{Float64},chimax::Int64)
    VT = permutedims(VT,[4,1,2,3])
    T  = permutedims(T,[4,1,2,3])
    Pv = HMover(VT,T,chimax)
    return Pv
end

function  HMover1(T::Array{Float64},CenterLeft::Array{Float64},CenterRight::Array{Float64},chimax::Int64)

    dlink = size(T,2);chi = size(CenterLeft,1)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[8,-2,9]*CenterRight[5,6,9]*CenterLeft[11,-5,10]*CenterRight[7,6,10]*
                T[1,-1,8,2]*T[1,-4,11,2]*T[5,-3,3,4]*T[7,-6,3,4]
    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    println(maximum(temp-temp'))
    F = svd((temp + temp')/2)
    append!(SlMatrix,[F.S])
    if dlink^2*chi > chimax
        Ph = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax)
    else
        Ph = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    end
    return Ph
end

function VMover1(T::Array{Float64},CenterLeft::Array{Float64},CenterRight::Array{Float64},chimax::Int64)
    dlink = size(T,1);chi = size(CenterLeft,1)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[-2,11,10]*CenterRight[6,7,10]*CenterLeft[-5,8,9]*CenterRight[6,5,9]*
                T[-3,1,2,11]*T[-1,7,4,3]*T[-6,1,2,8]*T[-4,5,4,3]
    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    println(maximum(temp-temp'))
    F = svd((temp+temp')/2)
    append!(SuMatrix,[F.S])
    if dlink^2*chi > chimax
        Pv = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax)
    else
        Pv = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    end
    return Pv
end





function  CenterProjector(CenterLeft::Array{Float64},CenterRight::Array{Float64},
                Left::Array{Float64},Right::Array{Float64},chimax::Int64)
    dlink = size(Left,1);chi = size(CenterLeft,3)
    @time @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[3,4,-2]*CenterRight[4,3,-5]*Left[1,2,-3]*
                                Left[6,5,-1]*Right[2,1,-6]*Right[5,6,-4]
    temp = reshape(temp,chi*dlink^4,chi*dlink^4)
    @time F = svd(temp)

    append!(ScMatrix,[F.S])

    if chi*dlink^4 > chimax
        PcL = reshape(F.U[:,1:chimax],dlink^2,chi,dlink^2,chimax)
        PcR = reshape(F.V[:,1:chimax],dlink^2,chi,dlink^2,chimax)
    else
        PcL = reshape(F.U,dlink^2,chi,dlink^2,chi*dlink^4)
        PcR = reshape(F.V,dlink^2,chi,dlink^2,chi*dlink^4)
    end

    return PcL,PcR
end


function CenterProjector1(CenterLeft::Array{Float64},CenterRight::Array{Float64},Up::Array{Float64},
                            Down::Array{Float64},Left::Array{Float64},Right::Array{Float64},chimax::Int64)

    chi = size(CenterLeft,3);Dlink = size(Up,4)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[1,3,-2]*CenterRight[4,5,-5]*Up[7,2,1,-3]*Down[7,-4,4,6]*
                    Left[8,-1,3,2]*Right[8,6,5,-6]

    temp = reshape(temp,chi*Dlink^2,chi*Dlink^2)
    F = svd((temp+temp')/2)
    append!(ScMatrix,[F.S])
    if chi*Dlink^2 > chimax
        PcL = reshape(F.U[:,1:chimax],Dlink,chi,Dlink,chimax)
        PcR = reshape(F.V[:,1:chimax],Dlink,chi,Dlink,chimax)
    else
        PcL = reshape(F.U,Dlink,chi,Dlink,chi*Dlink^2)
        PcR = reshape(F.V,Dlink,chi,Dlink,chi*Dlink^2)
    end

    return PcL,PcR
end




function CorseGraining(T::Array{Float64},NumLayer::Int64,chimax;Sz=Float64[1.0 0.0; 0.0 -1.0])
    HT = T; VT = T
    sizeT = size(T,1)
    #---- Left and Right
    Flr = svd(reshape(T,sizeT^2,sizeT^2))
    TL = Flr.U*Matrix(Diagonal(sqrt.(Flr.S)))
    TR = Flr.V*Matrix(Diagonal(sqrt.(Flr.S)))
    TL = reshape(TL,sizeT,sizeT,sizeT^2)
    TR = reshape(TR,sizeT,sizeT,sizeT^2)

    #---- Up and Right
    Fud = svd(reshape(permutedims(T,[4,1,2,3]),sizeT^2,sizeT^2))
    TU = Fud.U*Matrix(Diagonal(sqrt.(Fud.S)))
    TD = Fud.V*Matrix(Diagonal(sqrt.(Fud.S)))
    TU = reshape(TU,sizeT,sizeT,sizeT^2)
    TD = reshape(TD,sizeT,sizeT,sizeT^2)


    CenterLeft = TL
    CenterRight = TR


    Znorm = Array{Float64}(undef,0)
    CLnorm = Array{Float64}(undef,0)
    CRnorm = Array{Float64}(undef,0)

    HTnorm = Array{Float64}(undef,0)
    VTnorm = Array{Float64}(undef,0)
    NumSite = Array{Int64}(undef,0)
    FEttrg = Array{Float64}(undef,0)

    for j in 1:NumLayer
        println("This is NumLayer $j")
        @time begin
        #---- compute projectors
        #Ph = HMover(HT,T,chimax)
        #Pv = VMover(VT,T,chimax)
        println("Ph")
        Ph = HMover1(VT,CenterLeft,CenterRight,chimax)
        println("Pv")
        Pv = VMover1(HT,CenterLeft,CenterRight,chimax)
        println("Pc")
        #@time PcL,PcR = CenterProjector(CenterLeft,CenterRight,TL,TR,chimax)


        @tensor verticalUp[-1,-2,-3,-4] := VT[3,2,-3,4]*TL[5,4,-4]*TU[2,1,-2]*Pv[5,3,1,-1]
        @tensor verticalDn[-1,-2,-3,-4] := VT[-3,2,3,4]*TR[1,2,-2]*TD[4,5,-4]*Pv[5,3,1,-1]
        @tensor horizontalLeft[-1,-2,-3,-4] := HT[3,2,5,-3]*TL[5,4,-2]*TD[1,3,-4]*Ph[1,2,4,-1]
        @tensor horizontalRight[-1,-2,-3,-4] :=HT[2,-3,4,3]*TR[2,1,-4]*TU[5,4,-2]*Ph[1,3,5,-1]
        @time PcL,PcR = CenterProjector1(CenterLeft,CenterRight,verticalUp,verticalDn,horizontalLeft,horizontalRight,chimax)



        @tensor CenterLeft[-1,-2,-3] := verticalUp[-1,2,1,6]*horizontalLeft[-2,4,3,2]*CenterLeft[1,3,5]*PcL[4,5,6,-3]
        @tensor CenterRight[-1,-2,-3] := verticalDn[-1,2,1,6]*horizontalRight[-2,6,5,4]*CenterRight[1,5,3]*PcR[2,3,4,-3]
        append!(CLnorm,[maximum(CenterLeft)])
        append!(CRnorm,[maximum(CenterRight)])
        CenterLeft = CenterLeft/maximum(CenterLeft)
        CenterRight = CenterRight/maximum(CenterRight)


        println("sizeSz ",size(Sz))
        println("sizeT ",size(T))
        @tensor HTtemp[-1,-2,-3,-4] :=  HT[5,2,3,7]*Sz[-1,4,5,6]*T[3,1,-3,8]*Ph[4,2,1,-2]*Ph[6,7,8,-4]
        @tensor HT[-1,-2,-3,-4] := HT[5,2,3,7]*T[-1,4,5,6]*T[3,1,-3,8]*Ph[4,2,1,-2]*Ph[6,7,8,-4]
        @tensor VT[-1,-2,-3,-4] := VT[3,2,7,4]*T[1,-2,6,2]*T[5,4,8,-4]*Pv[5,3,1,-1]*Pv[8,7,6,-3]
        HTtemp = HTtemp/maximum(HT)
        append!(HTnorm,[maximum(HT)])
        append!(VTnorm,[maximum(VT)])
        HT = HT/maximum(HT)
        VT = VT/maximum(VT)

        @tensor Numerator[] := Sz[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*VT[18,1,4,2]*VT[12,9,18,10]*
                                HTtemp[5,8,15,6]*HT[7,14,16,8]*CenterLeft[4,6,13]*CenterRight[12,14,13]
        @tensor Numerator1[] := Sz[17,3,5,1]*Sz[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*VT[18,1,4,2]*VT[12,9,18,10]*
                                HT[5,8,15,6]*HT[7,14,16,8]*CenterLeft[4,6,13]*CenterRight[12,14,13]
        @tensor Denominator[] := T[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*VT[18,1,4,2]*VT[12,9,18,10]*
                                HT[5,8,15,6]*HT[7,14,16,8]*CenterLeft[4,6,13]*CenterRight[12,14,13]
        energy = Numerator[1]/Denominator[1]
        energy1 = Numerator1[1]/Denominator[1]
        append!(InternalEnergyTree,energy)
        append!(InternalEnergyTree,energy1)


        end
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := CenterLeft[1,2,3]*CenterRight[1,2,3]
        FE = (log(Z[1])+sum(log.(CLnorm))+sum(log.(CRnorm))+2*sum([j-1:-1:0...].*log.(HTnorm))+
                        2*sum([j-1:-1:0...].*log.(VTnorm)))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])

    end

    return CenterLeft,CenterRight,HT,VT,CLnorm,CRnorm,HTnorm,VTnorm,FEttrg,NumSite

end


#-----  Set Parameter
Dlink = 2
#chimax = 10
sx = Float64[0.0 1.0;
            1.0  0.0]
sz = Float64[1.0 0.0;
            0.0  -1.0]


#---- Base Tensor


MagExact = Array{Float64}(undef,0)
MagTree = Array{Float64}(undef,0)
MagTest = Array{Float64}(undef,0)
InternalEnergyTree = Array{Float64}(undef,0)
InternalEnergyExact = Array{Float64}(undef,0)
FreeEnergyExact = Array{Float64}(undef,0)
FreeEnergyTree = Array{Float64}(undef,0)
ScMatrix = []
SuMatrix = []
SlMatrix = []
#for chimax in 10:10:40
#for Beta in 0.44:0.1:0.44
    chimax = 20
    BetaExact =1/2*log(1+sqrt(2))
    Beta = 0.9994*BetaExact
    #Beta = 1/T
    feexact = ComputeFreeEnergy(Beta)
    append!(FreeEnergyExact,[feexact])
    println("This is Beta $Beta")
    #=
    E0=0.0;
    T0= zeros(Dlink,Dlink,Dlink,Dlink)
    T0 = T0 .+ exp(-(0-E0)*Beta)
    T0[1,1,1,1]=T0[2,2,2,2]= exp(-(4-E0)*Beta)
    T0[1,2,1,2]=T0[2,1,2,1]= exp(-(-4-E0)*Beta)
    @tensor Tsx[-1,-2,-3,-4] := T0[-1,-2,1,-4]*sz[1,-3]
    #@tensor Tsz[-1,-2,-3,-4] := T[-1,-2,1,-4]*sz[1,-3]
    =#

    #----- Base Tensor Test from 2D
    Q = Float64[exp(Beta) exp(-Beta);
                exp(-Beta) exp(Beta)]
    g = zeros(Dlink,Dlink,Dlink,Dlink)
    g[1,1,1,1] = g[2,2,2,2] = 1
    gp = zeros(Dlink,Dlink,Dlink,Dlink)
    gp[1,1,1,1] = -1;gp[2,2,2,2] = 1
    @tensor T[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*g[1,2,3,4]
    @tensor Tsx[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*gp[1,2,3,4]

    #for chimax in 10:10:30
    #for NumLayer in 100:100:100
    #chimax = 20
        NumLayer = 200
        Ns = 1+8*sum(1:1:NumLayer)
        CenterLeft,CenterRight,HT,VT,CLnorm,CRnorm,HTnorm,VTnorm,FEttrg,numsite = CorseGraining(T,NumLayer,chimax,Sz=Tsx)
        @tensor Z[] := CenterLeft[1,2,3]*CenterRight[1,2,3]

        FE = (log(Z[1])+sum(log.(CLnorm))+sum(log.(CRnorm))+2*sum([NumLayer-1:-1:0...].*log.(HTnorm))+
                        2*sum([(NumLayer-1):-1:0...].*log.(VTnorm)))/Ns
        append!(FreeEnergyTree,[FE])
    #end
    #=
    @tensor Numerator[] := Tsx[17,3,5,1]*T[19,2,7,3]*Tsx[15,11,17,9]*T[16,10,19,11]*VT[18,1,4,2]*VT[12,9,18,10]*
                            HT[5,8,15,6]*HT[7,14,16,8]*CenterLeft[4,6,13]*CenterRight[12,14,13]
    @tensor Denominator[] := T[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*VT[18,1,4,2]*VT[12,9,18,10]*
                            HT[5,8,15,6]*HT[7,14,16,8]*CenterLeft[4,6,13]*CenterRight[12,14,13]
    energy = Numerator[1]/Denominator[1]
    append!(InternalEnergyTree,energy)
    =#
    internalenergyexact = ComputeInternalEnergy(Beta)
    append!(InternalEnergyExact,internalenergyexact)
    #
#end

#------------------------All below are for testing use

#=
#exact = (1-sinh(2*Beta)^(-4))^(1/8)
#append!(MagExact,[exact])
#
@tensor Numerator[] := Tsx[3,11,1,4]*T[17,10,13,11]*T[2,18,3,7]*T[16,15,17,18]*
                            rhoDenominator[6,5,8,12]*VT[9,4,6,10]*VT[8,7,9,15]*HT[1,14,2,5]*HT[13,12,16,14]
@tensor Denominator[] := T[3,11,1,4]*T[17,10,13,11]*T[2,18,3,7]*T[16,15,17,18]*
                            rhoDenominator[6,5,8,12]*VT[9,4,6,10]*VT[8,7,9,15]*HT[1,14,2,5]*HT[13,12,16,14]
append!(MagTree,[Numerator[1]/Denominator[1]])

@tensor EnergyNumerator[] := Tsx[17,3,4,1]*Tsx[4,9,10,5]*T[24,2,8,3]*T[8,7,14,9]*T[16,23,17,18]*T[22,21,24,23]*
                    VT[20,1,6,2]*VT[6,5,12,7]*VT[19,18,20,21]*HT[10,15,16,11]*HT[14,13,22,15]*rhoDenominator[12,11,19,13]
@tensor EnergyDenominator[] := T[17,3,4,1]*T[4,9,10,5]*T[24,2,8,3]*T[8,7,14,9]*T[16,23,17,18]*T[22,21,24,23]*
                    VT[20,1,6,2]*VT[6,5,12,7]*VT[19,18,20,21]*HT[10,15,16,11]*HT[14,13,22,15]*rhoDenominator[12,11,19,13]

append!(InternalEnergyTree,[EnergyNumerator[1]/EnergyDenominator[1]])

internalenergyexact = ComputeInternalEnergy(Beta)
append!(InternalEnergyExact,[internalenergyexact])

=#
