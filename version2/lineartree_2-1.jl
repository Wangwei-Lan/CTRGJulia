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

function  HMoverL(TU::Array{Float64},TD::Array{Float64},CenterLeft::Array{Float64},
                        CenterRight::Array{Float64},chimax::Int64)
    dlink = size(T,2);chi = size(CenterLeft,1)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[8,-2,9]*CenterRight[5,6,9]*CenterLeft[11,-5,10]*CenterRight[7,6,10]*
                TU[1,-1,8,2]*TU[1,-4,11,2]*TD[5,-3,3,4]*TD[7,-6,3,4]
    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    println(maximum(temp-temp'))
    F = svd((temp + temp')/2)
    append!(SlMatrix,[F.S])
    if dlink^2*chi > chimax
        PhL = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax)
    else
        PhL = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    end
    return PhL
end


function  HMoverR(TU::Array{Float64},TD::Array{Float64},CenterLeft::Array{Float64},
                    CenterRight::Array{Float64},chimax::Int64)
    dlink = size(T,2);chi = size(CenterLeft,1)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[5,6,9]*CenterRight[8,-5,9]*CenterLeft[7,6,10]*CenterRight[11,-2,10]*
                TU[3,4,5,-4]*TD[8,2,1,-6]*TU[3,4,7,-1]*TD[11,2,1,-3]
    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    println(maximum(temp-temp'))
    F = svd((temp + temp')/2)
    append!(SlMatrix,[F.S])
    if dlink^2*chi > chimax
        PhR = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax)
    else
        PhR = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    end
    return PhR
end


function VMoverU(TL::Array{Float64},TR::Array{Float64},CenterLeft::Array{Float64},
                    CenterRight::Array{Float64},chimax::Int64)
    dlink = size(T,1);chi = size(CenterLeft,1)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[-2,11,10]*CenterRight[6,7,10]*CenterLeft[-5,8,9]*CenterRight[6,5,9]*
                TL[-3,1,2,11]*TR[-1,7,4,3]*TL[-6,1,2,8]*TR[-4,5,4,3]
    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    println(maximum(temp-temp'))
    F = svd((temp+temp')/2)
    append!(SuMatrix,[F.S])
    if dlink^2*chi > chimax
        PvU = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax)
    else
        PvU = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    end
    return PvU
end


function VMoverD(TL::Array{Float64},TR::Array{Float64},CenterLeft::Array{Float64},
                    CenterRight::Array{Float64},chimax::Int64)
    dlink = size(T,1);chi = size(CenterLeft,1)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[6,7,10]*CenterRight[-5,11,10]*CenterLeft[6,5,9]*CenterRight[-2,8,9]*
                TL[4,3,-6,7]*TR[2,11,-4,1]*TL[4,3,-3,5]*TR[2,8,-1,1]
    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    println(maximum(temp-temp'))
    F = svd((temp+temp')/2)
    append!(SuMatrix,[F.S])
    if dlink^2*chi > chimax
        PvD = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax)
    else
        PvD = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    end
    return PvD
end





function  CenterProjector(T::Array{Float64},CenterLeft::Array{Float64},CenterRight::Array{Float64},
                HTL::Array{Float64},HTR::Array{Float64},VTU::Array{Float64},VTD::Array{Float64},
                Left::Array{Float64},Right::Array{Float64},chimax::Int64)
    dlink = size(Left,1);chi = size(CenterLeft,3)
    #@tensor temp[-1,-2,-3,-4] := Left[1,2,-1]*Right[2,1,-3]*CenterLeft[3,4,-2]*CenterRight[4,3,-4]

    @tensor temp[-1,-2,-3,-4] := Left[8,7,-1]*Right[7,10,-3]*CenterLeft[16,15,-2]*CenterRight[11,13,-4]*HTL[12,9,8,15]*
                HTR[1,13,6,2]*VTU[2,5,16,1]*VTD[11,10,9,14]*T[3,4,12,5]*T[6,14,4,3]
    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    F = svd(temp)
    if chi*dlink^2 > chimax
        PcL1 = reshape(F.U[:,1:chimax],dlink^2,chi,chimax)
        PcR1 = reshape(F.V[:,1:chimax],dlink^2,chi,chimax)
    else
        PcL1 = reshape(F.U,dlink^2,chi,chi*dlink^2)
        PcR1 = reshape(F.V,dlink^2,chi,chi*dlink^2)
    end


    chi1 = size(PcL1,3)
    #@tensor temp[-1,-2,-3,-4] := Left[2,1,5]*Right[1,2,7]*CenterLeft[4,3,6]*CenterRight[3,4,8]*
    #            PcL1[5,6,-1]*PcR1[7,8,-3]*Left[9,10,-2]*Right[10,9,-4]
    @tensor temp[-1,-2,-3,-4] := Left[2,17,3]*Right[17,6,7]*Left[20,21,-2]*Right[22,20,-4]*HTL[18,16,2,1]*HTR[22,14,13,12]*
                        VTU[12,11,19,21]*VTD[5,6,16,15]*T[9,10,18,11]*T[13,15,10,9]*CenterLeft[19,18,4]*CenterRight[5,14,8]*
                        PcL1[3,4,-1]*PcR1[7,8,-3]
    temp = reshape(temp,dlink^2*chi1,dlink^2*chi1)
    @time F = svd(temp)
    append!(ScMatrix,[F.S])
    if chi1*dlink^2 > chimax
        PcL2 = reshape(F.U[:,1:chimax],chi1,dlink^2,chimax)
        PcR2 = reshape(F.V[:,1:chimax],chi1,dlink^2,chimax)
    else
        PcL2 = reshape(F.U,chi1,dlink^2,chi1*dlink^2)
        PcR2 = reshape(F.V,chi1,dlink^2,chi1*dlink^2)
    end
    return PcL1,PcR1,PcL2,PcR2
end









function CenterProjector1(CenterLeft::Array{Float64},CenterRight::Array{Float64},Up::Array{Float64},
                            Down::Array{Float64},Left::Array{Float64},Right::Array{Float64},chimax::Int64)

    chi = size(CenterLeft,3);Dlink = size(Up,4)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[1,3,-2]*CenterRight[4,5,-5]*Up[8,2,1,-3]*Down[7,-4,4,6]*
                    Left[7,-1,3,2]*Right[8,6,5,-6]

    temp = reshape(temp,chi*Dlink^2,chi*Dlink^2)
    #F = svd((temp+temp')/2)
    F = svd(temp)
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

    HTL = T; VTU = T
    HTR = T; VTD = T
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

    HTLnorm = Array{Float64}(undef,0)
    HTRnorm = Array{Float64}(undef,0)
    VTUnorm = Array{Float64}(undef,0)
    VTDnorm = Array{Float64}(undef,0)


    NumSite = Array{Int64}(undef,0)
    FEttrg = Array{Float64}(undef,0)

    for j in 1:NumLayer
        println("This is NumLayer $j")
        @time begin
            append!(HTLnorm,[maximum(HTL)])
            append!(HTRnorm,[maximum(HTR)])
            append!(VTUnorm,[maximum(VTU)])
            append!(VTDnorm,[maximum(VTD)])
            HTL = HTL/maximum(HTL)
            HTR = HTR/maximum(HTR)
            VTU = VTU/maximum(VTU)
            VTD = VTD/maximum(VTD)
        #---- compute projectors
        #Ph = HMover(HT,T,chimax)
        #Pv = VMover(VT,T,chimax)
        PhL = HMoverL(VTU,VTD,CenterLeft,CenterRight,chimax)
        PhR = HMoverR(VTU,VTD,CenterLeft,CenterRight,chimax)
        PvU = VMoverU(HTL,HTR,CenterLeft,CenterRight,chimax)
        PvD = VMoverD(HTL,HTR,CenterLeft,CenterRight,chimax)

        @tensor verticalUp[-1,-2,-3,-4] := VTU[3,2,-3,4]*TL[5,4,-4]*TU[2,1,-2]*PvU[5,3,1,-1]
        @tensor verticalDn[-1,-2,-3,-4] := VTD[-3,2,3,4]*TR[1,2,-2]*TD[4,5,-4]*PvD[5,3,1,-1]
        @tensor horizontalLeft[-1,-2,-3,-4] := HTL[3,2,5,-3]*TL[5,4,-2]*TD[1,3,-4]*PhL[1,2,4,-1]
        @tensor horizontalRight[-1,-2,-3,-4] :=HTR[2,-3,4,3]*TR[2,1,-4]*TU[5,4,-2]*PhR[1,3,5,-1]
        @time PcL1,PcR1,PcL2,PcR2 = CenterProjector(T,CenterLeft,CenterRight,HTL,HTR,VTU,VTD,TL,TR,chimax)
        #@time PcL,PcR = CenterProjector1(CenterLeft,CenterRight,verticalUp,verticalDn,horizontalLeft,horizontalRight,chimax)


        #@tensor CenterLeft[-1,-2,-3] := verticalUp[-1,2,1,6]*horizontalLeft[-2,4,3,2]*CenterLeft[1,3,5]*PcL[4,5,6,-3]
        #@tensor CenterRight[-1,-2,-3] := verticalDn[-1,2,1,6]*horizontalRight[-2,6,5,4]*CenterRight[1,5,3]*PcR[2,3,4,-3]

        @tensor CenterLeft[-1,-2,-3] := verticalUp[-1,4,5,7]*horizontalLeft[-2,3,2,4]*CenterLeft[5,2,1]*PcL1[3,1,6]*PcL2[6,7,-3]
        @tensor CenterRight[-1,-2,-3] := verticalDn[-1,2,3,5]*horizontalRight[-2,5,4,6]*CenterRight[3,4,1]*PcR1[2,1,7]*PcR2[7,6,-3]
        append!(CLnorm,[maximum(CenterLeft)])
        append!(CRnorm,[maximum(CenterRight)])
        CenterLeft = CenterLeft/maximum(CenterLeft)
        CenterRight = CenterRight/maximum(CenterRight)


        #@tensor HTtemp[-1,-2,-3,-4] :=  HT[5,2,3,7]*Sz[-1,4,5,6]*T[3,1,-3,8]*Ph[4,2,1,-2]*Ph[6,7,8,-4]
        #HTtemp = HTtemp/maximum(HT)
        @tensor HTL[-1,-2,-3,-4] := HTL[5,2,3,7]*T[-1,4,5,6]*T[3,1,-3,8]*PhL[4,2,1,-2]*PhL[6,7,8,-4]
        @tensor HTR[-1,-2,-3,-4] := HTR[5,2,3,7]*T[-1,4,5,6]*T[3,1,-3,8]*PhR[4,2,1,-2]*PhR[6,7,8,-4]
        @tensor VTU[-1,-2,-3,-4] := VTU[3,2,7,4]*T[1,-2,6,2]*T[5,4,8,-4]*PvU[5,3,1,-1]*PvU[8,7,6,-3]
        @tensor VTD[-1,-2,-3,-4] := VTD[3,2,7,4]*T[1,-2,6,2]*T[5,4,8,-4]*PvD[5,3,1,-1]*PvD[8,7,6,-3]


        #=
        #@tensor Numerator[] := Sz[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*VT[18,1,4,2]*VT[12,9,18,10]*
        #                        HTtemp[5,8,15,6]*HT[7,14,16,8]*CenterLeft[4,6,13]*CenterRight[12,14,13]
        #@tensor Numerator1[] := Sz[17,3,5,1]*Sz[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*VT[18,1,4,2]*VT[12,9,18,10]*
        #                        HT[5,8,15,6]*HT[7,14,16,8]*CenterLeft[4,6,13]*CenterRight[12,14,13]
        #@tensor Denominator[] := T[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*VT[18,1,4,2]*VT[12,9,18,10]*
        #                        HT[5,8,15,6]*HT[7,14,16,8]*CenterLeft[4,6,13]*CenterRight[12,14,13]
        #energy = Numerator[1]/Denominator[1]
        #energy1 = Numerator1[1]/Denominator[1]
        #append!(InternalEnergyTree,energy)
        #append!(InternalEnergyTree,energy1)
        =#
        end
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := CenterLeft[1,2,3]*CenterRight[1,2,3]
        FE = (log(Z[1])+sum(log.(CLnorm))+sum(log.(CRnorm))+sum([j:-1:1...].*log.(HTLnorm))+
            sum([j:-1:1...].*log.(HTRnorm))+sum([j:-1:1...].*log.(VTUnorm))+ sum([j:-1:1...].*log.(VTDnorm)))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])

        #end
    end

    return CenterLeft,CenterRight,HTL,VTU,CLnorm,CRnorm,HTLnorm,VTUnorm,FEttrg,NumSite

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
    chimax = 40
    BetaExact =1/2*log(1+sqrt(2))
    Beta = 0.9994*BetaExact
    #Beta = 1/T
    feexact = ComputeFreeEnergy(Beta)
    append!(FreeEnergyExact,[feexact])
    println("This is Beta $Beta")

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

        NumLayer = 600
        Ns = 1+8*sum(1:1:NumLayer)
        CenterLeft,CenterRight,HT,VT,CLnorm,CRnorm,HTnorm,VTnorm,FEttrg,numsite = CorseGraining(T,NumLayer,chimax,Sz=Tsx)


        #@tensor Z[] := CenterLeft[1,2,3]*CenterRight[1,2,3]
        #FE = (log(Z[1])+sum(log.(CLnorm))+sum(log.(CRnorm))+2*sum([NumLayer-1:-1:0...].*log.(HTnorm))+
        #                2*sum([(NumLayer-1):-1:0...].*log.(VTnorm)))/Ns
        #append!(FreeEnergyTree,[FE])



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
