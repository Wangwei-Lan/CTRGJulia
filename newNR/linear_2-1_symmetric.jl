include("../../2DClassical/partition.jl")
using TensorOperations
using LinearAlgebra
using Arpack
using KrylovKit


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


function  Mover(RGTensor::Array{Array{Float64}},TL::Array{Float64},TR::Array{Float64},chimax::Int64;direction=nothing)
    # cl : Center left Tensor
    # cr : Center Right Tensor
    cl = RGTensor[2];cr = RGTensor[3]
    #  t1 : vertical up tensor
    #  t2 : vertical dn tensor
    #  t3 : horizontal left tensor
    #  t4 : horizontal right tensor
    t1 = RGTensor[4];t2 = RGTensor[5];t3 = RGTensor[6];t4 = RGTensor[7]
    chicl1 = size(cl,1);chicl3 = size(cl,3);dlink = 2

    if direction == "LD"
        @tensor projcenter[-1,-2,-3,-4] := cl[4,3,-2]*cr[5,4,-4]*TL[6,7,-1]*TR[7,8,-3]*t2[1,2,6,3]*t1[5,8,2,1]
        @tensor projhorizontal[-1,-2,-3,-4] := cl[7,10,6]*cr[3,4,6]*cl[7,9,8]*cr[5,4,8]*T[15,-2,16,14]*
                            T[13,-4,16,12]*t2[11,-1,15,10]*t2[11,-3,13,9]*t1[3,14,2,1]*t1[5,12,2,1]
        @tensor projvertical[-1,-2,-3,-4] := cl[4,5,8]*cr[9,7,8]*cl[4,3,6]*cr[10,7,6]*T[12,16,-4,13]*T[14,16,-2,15]*
                            t2[2,1,12,5]*t2[2,1,14,3]*t1[9,13,-3,11]*t1[10,15,-1,11]
    elseif  direction == "RU"
        @tensor projcenter[-1,-2,-3,-4] := cl[3,4,-2]*cr[4,5,-4]*TL[6,7,-1]*TR[8,6,-3]*t1[2,1,3,7]*t2[8,5,1,2]
        @tensor projvertical[-1,-2,-3,-4] := cl[9,7,6]*cr[4,5,6]*cl[11,7,8]*cr[4,3,8]*t1[-1,10,9,12]*
                            t2[13,5,2,1]*t1[-3,10,11,14]*t2[15,3,2,1]*T[-2,12,13,16]*T[-4,14,15,16]
        @tensor projhorizontal[-1,-2,-3,-4] := cl[3,4,8]*cr[7,11,8]*cl[5,4,6]*cr[7,9,6]*t1[1,2,3,14]*t2[15,11,10,-3]*
                            t1[1,2,5,13]*t2[12,9,10,-1]*T[16,14,15,-4]*T[16,13,12,-2]
    end
    projcenter = reshape(projcenter,dlink^2*chicl3,dlink^2*chicl3)
    println(maximum(projcenter-projcenter'))
    F = svd(projcenter)
    if dlink^2*chicl3 > chimax
        PcL = reshape(F.U[:,1:chimax],dlink^2,chicl3,chimax)
        PcR = reshape(F.V[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(F.U,dlink^2,chicl3,dlink^2*chicl3)
        PcR = reshape(F.V,dlink^2,chicl3,dlink^2*chicl3)
    end

    projhorizontal = reshape(projhorizontal,chicl1*dlink,chicl1*dlink)
    F = svd((projhorizontal+projhorizontal')/2)
    dlink*chicl1 > chimax ? Ph = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                        Ph = reshape(F.U,chicl1,dlink,dlink*chicl1)


    projvertical = reshape(projvertical,chicl1*dlink,chicl1*dlink)
    F = svd((projvertical+projvertical')/2)
    dlink*chicl1 > chimax ? Pv = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                            Pv= reshape(F.U,chicl1,dlink,dlink*chicl1)

    return PcL,PcR,Ph,Pv

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
        @time begin
        PcL,PcR,Ph,Pv = Mover(RGTensor,TL,TR,chimax,direction="LD")
        Pv = Ph
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][4,6,8]*PcL[7,8,-3]*Ph[1,2,-2]*Pv[4,5,-1]*
                                        RGTensor[5][5,1,3,6]*TL[3,2,7]
        #@tensor RGTensor[3][-1,-2,-3] := RGTensor[3][5,4,8]*PcR[7,8,-3]*Ph[4,6,-2]*Pv[1,2,-1]*
        #                                RGTensor[4][5,3,1,6]*TR[2,3,7]
        RGTensor[3] = copy(RGTensor[2])
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][-1,4,1,2]*T[1,5,-3,3]*Ph[4,5,-2]*Ph[2,3,-4]
        #@tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,1,5,-4]*T[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        RGTensor[4] = permutedims(RGTensor[5],[4,1,2,3])
        end
        NormalizeTensor(RGTensor,RGnorm)

        @time begin
        PcL,PcR,Ph,Pv = Mover(RGTensor,TL,TR,chimax,direction="RU")
        Pv = Ph
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][6,4,7]*TL[3,2,8]*RGTensor[4][1,5,6,2]*
                                    Ph[4,5,-2]*Pv[1,3,-1]*PcL[8,7,-3]
        #@tensor RGTensor[3][-1,-2,-3] := RGTensor[3][4,5,7]*TR[2,3,8]*RGTensor[5][2,5,6,1]*
        #                            Ph[1,3,-2]*Pv[4,6,-1]*PcR[8,7,-3]
        RGTensor[3] = copy(RGTensor[2])
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][1,5,-3,3]*T[-1,4,1,2]*Ph[5,4,-2]*Ph[3,2,-4]
        RGTensor[4] = permutedims(RGTensor[5],[4,1,2,3])
        #@tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][2,-2,4,1]*T[3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        end
        NormalizeTensor(RGTensor,RGnorm)
        #---- Normalize to avoid calculation explotion
        end
        #
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
        println("Z ",Z[1])
        FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
                sum([j-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
        append!(FEttrg,[FE])
        #
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
    chimax = 20
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
        NumLayer = 600
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
