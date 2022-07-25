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


function ApplyVT(VT::Array{Float64},v::Array{Float64})
    @tensor v[-1,-2] = VT[1,4,-1,2]*VT[3,2,-2,4]*v[1,3]
    v = v/maximum(v)
    return v
end



function FreeEnergyDensity(VTU::Array{Float64},VTD::Array{Float64})
    chi = size(VTU,1)
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(VTU,y),rand(chi,chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(VTD,[3,2,1,4]),y),rand(chi,chi),1)

    eigup = reshape(eigvecsup[1],chi^2)
    eigdn = reshape(eigvecsdn[1],chi^2)
    Z = LinearAlgebra.dot(eigup,eigdn)

    @tensor Z1[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*VTU[1,4,5,2]*VTU[3,2,6,4]
    @tensor Z2[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*VTD[1,4,5,2]*VTD[3,2,6,4]
    #println(Z1[1]/dot(eigup,eigdn))
    return Z1[1]/LinearAlgebra.dot(eigup,eigdn)
end



function HMover(HT::Array{Float64},T::Array{Float64},chimax::Int64)

    chi = size(HT,2);dlink = size(T,1)
    @tensor temp[-1,-2,-3,-4,-5,-6] := HT[6,-2,8,5]*HT[7,-5,9,5]*T[1,-1,6,2]*T[1,-4,7,2]*
                            T[8,-3,3,4]*T[9,-6,3,4]
    temp = reshape(temp,chi*dlink^2,chi*dlink^2)
    F = svd((temp+temp')/2)
    chi*dlink^2 > chimax ? Ph = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax) :
            Ph = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    return Ph
end

function VMover(VT::Array{Float64},T::Array{Float64},chimax::Int64)
    VT = permutedims(VT,[4,1,2,3])
    T  = permutedims(T,[4,1,2,3])
    Pv = HMover(VT,T,chimax)
    return Pv
end



function  Mover(RGTensor::Array{Array{Float64}},chimax::Int64;direction=nothing)
    # cl : Center left Tensor
    # cr : Center Right Tensor
    cl = RGTensor[2];cr = RGTensor[3]
    #  t1 : vertical up tensor
    #  t2 : vertical dn tensor
    #  t3 : horizontal left tensor
    #  t4 : horizontal right tensor
    t1 = RGTensor[4];t2 = RGTensor[5];t3 = RGTensor[6];t4 = RGTensor[7]
    chi = size(cl,1);dlink = 2
    if direction == "L"
        @tensor temp[-1,-2,-3,-4,-5,-6] := cl[8,-2,9]*cr[5,6,9]*cl[11,-5,10]*cr[7,6,10]*
                t1[1,-1,8,2]*t1[1,-4,11,2]*t2[5,-3,3,4]*t2[7,-6,3,4]
    elseif direction == "R"
        @tensor temp[-1,-2,-3,-4,-5,-6] := cl[5,6,9]*cr[8,-5,9]*cl[7,6,10]*cr[11,-2,10]*
                    t1[3,4,5,-4]*t2[8,2,1,-6]*t1[3,4,7,-1]*t2[11,2,1,-3]
    elseif direction == "U"
        @tensor temp[-1,-2,-3,-4,-5,-6] := cl[-2,11,10]*cr[6,7,10]*cl[-5,8,9]*cr[6,5,9]*
                    t3[-3,1,2,11]*t4[-1,7,4,3]*t3[-6,1,2,8]*t4[-4,5,4,3]
    elseif direction == "D"
        @tensor temp[-1,-2,-3,-4,-5,-6] := cl[6,7,10]*cr[-5,11,10]*cl[6,5,9]*cr[-2,8,9]*
                    t3[4,3,-6,7]*t4[2,11,-4,1]*t3[4,3,-3,5]*t4[2,8,-1,1]
    end

    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    F = svd( (temp + temp')/2)
    dlink^2*chi > chimax ? Proj = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax) :
                        Proj = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    return Proj
end

function NormalizeTensor(RGTensor,RGnorm)
    for j in 2:7
        append!(RGnorm[j],maximum(RGTensor[j]))
        RGTensor[j] = RGTensor[j]/maximum(RGTensor[j])
    end
end


function CenterProjectorT(RGTensor::Array{Array{Float64}},TL::Array{Float64},
                            TR::Array{Float64},TU::Array{Float64},TD::Array{Float64},chimax::Int64)
    dlink = size(TL,1); chi = size(RGTensor[2],3)
    cl = RGTensor[2]; cr = RGTensor[3];
    t1 = RGTensor[4]; t2 = RGTensor[5]; t3 = RGTensor[6] ; t4 = RGTensor[7]
    @tensor temp[-1,-2,-3,-4,-5,-6] := cl[3,4,-2]*cr[4,3,-5]*TL[1,2,-3]*
                                TL[6,5,-1]*TR[2,1,-6]*TR[5,6,-4]
    #
    #@tensor temp[-1,-2,-3,-4,-5,-6] := cl[1,3,-2]*cr[4,5,-5]*TU[8,2,1,-3]*TD[7,-4,4,6]*
    #                TL[7,-1,3,2]*TR[8,6,5,-6]

    temp = reshape(temp,chi*dlink^4,chi*dlink^4)
    F = svd(temp)
    if chi*dlink^4 > chimax
        PcL = reshape(F.U[:,1:chimax],dlink^2,chi,dlink^2,chimax)
        PcR = reshape(F.V[:,1:chimax],dlink^2,chi,dlink^2,chimax)
    else
        PcL = reshape(F.U,dlink^2,chi,dlink^2,chi*dlink^4)
        PcR = reshape(F.V,dlink^2,chi,dlink^2,chi*dlink^4)
    end
    return PcL,PcR
end



function svdT(T::Array{Float64})
    sizeT = size(T,1)
    F = svd(reshape(T,sizeT^2,sizeT^2))
    T1 = F.U*Matrix(Diagonal(sqrt.(F.S)))
    T2 = F.V*Matrix(Diagonal(sqrt.(F.S)))
    T1 = reshape(T1,sizeT,sizeT,sizeT^2)
    T2 = reshape(T2,sizeT,sizeT,sizeT^2)
    return T1,T2
end


function CorseGraining(T::Array{Float64},NumLayer::Int64,chimax;Sz=Float64[1.0 0.0; 0.0 -1.0])
    println("start corse Graining")

    #---- Diagonal corner tensor
    TL,TR = svdT(T)
    TU,TD = svdT(permutedims(T,[4,1,2,3]))

    #--- Boundary tensor
    # RGTensor[1]   base tensor T
    # RGTensor[2]   Center Tensor Left : cl
    # RGTensor[3]   Center Tensor Right : cr
    # RGTensor[4]   vertical up tensor : vtu
    # RGTensor[5]   vertical dn tensor : vtd
    # RGTensor[6]   horizontal left tensor : htl
    # RGTensor[7]   horizontal right tensor : htr
    RGTensor = Array{Array{Float64}}(undef,7)
    RGTensor[1] = T
    RGTensor[2] = TL;RGTensor[3] = TR
    RGTensor[4:end] = [T for j in 4:7]

    #-- Norms of different tensor
    RGnorm = Array{Array{Float64}}(undef,7)
    RGnorm = [[] for j in 1:7]

    NumSite = Array{Int64}(undef,0)
    FEttrg = Array{Float64}(undef,0)
    FEttrg1 = Array{Float64}(undef,0)
    println("start loop")
    for j in 1:NumLayer
        println("This is NumLayer $j")

        @time begin
        #---- compute projectors
        #PhL = Mover(VTU,VTD,CL,CR,chimax,direction="L")
        #PhR = Mover(VTU,VTD,CL,CR,chimax,direction="R")
        #PvU = Mover(HTL,HTR,CL,CR,chimax,direction="U")
        #PvD = Mover(HTL,HTR,CL,CR,chimax,direction="D")
        PhL = Mover(RGTensor,chimax,direction="L")
        PhR = Mover(RGTensor,chimax,direction="R")
        PvU = Mover(RGTensor,chimax,direction="U")
        PvD = Mover(RGTensor,chimax,direction="D")

        #--- Compute tensors to normailize the center tensor
        @tensor vUp[-1,-2,-3,-4] := RGTensor[4][3,2,-3,4]*TL[5,4,-4]*TU[2,1,-2]*PvU[5,3,1,-1]
        @tensor vDn[-1,-2,-3,-4] := RGTensor[5][-3,2,3,4]*TR[1,2,-2]*TD[4,5,-4]*PvD[5,3,1,-1]
        @tensor hLeft[-1,-2,-3,-4] := RGTensor[6][3,2,5,-3]*TL[5,4,-2]*TD[1,3,-4]*PhL[1,2,4,-1]
        @tensor hRight[-1,-2,-3,-4] := RGTensor[7][2,-3,4,3]*TR[2,1,-4]*TU[5,4,-2]*PhR[1,3,5,-1]

        # Center Projector
        #@time PcL,PcR = CenterProjector(T,CenterLeft,CenterRight,HTL,HTR,VTU,VTD,TL,TR,chimax)
        #@time PcL,PcR = CenterProjector1(CenterLeft,CenterRight,vUp,vUp,hLeft,hRight,chimax)
        @time PcL,PcR = CenterProjectorT(RGTensor,TL,TR,TU,TD,chimax)

        println("start renormalize")
        #---- renormalize center tensor
        @tensor RGTensor[2][-1,-2,-3] := vUp[-1,2,1,6]*hLeft[-2,4,3,2]*RGTensor[2][1,3,5]*PcL[4,5,6,-3]
        @tensor RGTensor[3][-1,-2,-3] := vDn[-1,2,1,6]*hRight[-2,6,5,4]*RGTensor[3][1,5,3]*PcR[2,3,4,-3]
        #------  renormalize boundary tensor
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,2,7,4]*T[1,-2,6,2]*T[5,4,8,-4]*PvU[5,3,1,-1]*PvU[8,7,6,-3]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][3,2,7,4]*T[1,-2,6,2]*T[5,4,8,-4]*PvD[5,3,1,-1]*PvD[8,7,6,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][5,2,3,7]*T[-1,4,5,6]*T[3,1,-3,8]*PhL[4,2,1,-2]*PhL[6,7,8,-4]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][5,2,3,7]*T[-1,4,5,6]*T[3,1,-3,8]*PhR[4,2,1,-2]*PhR[6,7,8,-4]
        #---- Normalize to avoid calculation explotion
        NormalizeTensor(RGTensor,RGnorm)

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
        #
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
        FE = (log(Z[1])+sum(log.(RGnorm[2]))+sum(log.(RGnorm[3]))+sum([j-1:-1:0...].*log.(RGnorm[4]))+
            sum([j-1:-1:0...].*log.(RGnorm[5]))+sum([j-1:-1:0...].*log.(RGnorm[6]))+
            sum([j-1:-1:0...].*log.(RGnorm[7])))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        #

        #=
        Ns = 2*(2*j+1)
        Lambda = FreeEnergyDensity(VTU,VTD)
        FE = (real(Lambda)+sum(log.(VTUnorm))+sum(log.(VTDnorm)))/(2*(2*j+1))
        append!(FEttrg1,[FE])
        append!(NumSite,[Ns])
        =#

    end

    return RGTensor,RGnorm,FEttrg,FEttrg1,NumSite

end




println("start calculation")
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
    chimax = 60
    BetaExact =1/2*log(1+sqrt(2))
    Beta = 0.9994*BetaExact
    #Beta = 1/T
    feexact =  ComputeFreeEnergy(Beta)
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

        NumLayer = 800
        Ns = 1+8*sum(1:1:NumLayer)
        println("before calculation")
        RGTensor,RGnorm,FEttrg,FEttrg1,numsite = CorseGraining(T,NumLayer,chimax,Sz=Tsx)




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
