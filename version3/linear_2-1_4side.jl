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



function FreeEnergyDensity(RGTensor::Array{Array{Float64}})
    chi = size(RGTensor[4],1)
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1)
    #println(eigvalup)
    #println(eigvaldn)
    eigup = reshape(eigvecsup[1],chi^2)
    eigdn = reshape(eigvecsdn[1],chi^2)
    Z = LinearAlgebra.dot(eigup,eigdn)

    @tensor Z1[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*RGTensor[4][1,4,5,2]*RGTensor[4][3,2,6,4]
    #@tensor Z2[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*VTD[1,4,5,2]*VTD[3,2,6,4]
    #println(Z1[1]/dot(eigup,eigdn))
    return Z1[1]/LinearAlgebra.dot(eigup,eigdn)
end

function InternalEnergyTreeRG(RGTensor::Array{Array{Float64}})

    chi = size(RGTensor[4],1)
    cl = RGTensor[2] ; cr = RGTensor[3]; t = RGTensor[1]
    t1 = RGTensor[4];t2 = RGTensor[5] ; t1e = RGTensor[8]
    #eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1,:LM,maxiter=300)
    #eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1,:LM,maxiter=300)


    #@tensor Numerator[] := real(eigvecsup)[1][1,3]*real(eigvecsdn)[1][6,5]*RGTensor[4][1,4,6,2]*RGTensor[8][3,2,5,4]
    #@tensor Denominator[] := real(eigvecsup[1])[1,3]*real(eigvecsdn)[1][6,5]*RGTensor[4][1,4,6,2]*RGTensor[4][3,2,5,4]
    #@tensor Numerator[] := cl*cr*t1*t1*t2*t2*t
    #@tensor Denominator[] := cl*cr*t1e*t1*t2*t2*t
    @tensor Numerator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1e[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]
    @tensor Denominator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]

    return Numerator[1]/Denominator[1]



end

x= 1





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
        @tensor projcenter[-1,-2,-3,-4] := cl[4,3,-2]*cr[5,4,-4]*TL[6,7,-1]*TR[7,8,-3]*t2[1,2,6,3]*t3[5,8,2,1]
        @tensor projhl[-1,-2,-3,-4] := cl[7,10,6]*cr[3,4,6]*cl[7,9,8]*cr[5,4,8]*T[15,-2,16,14]*
                            T[13,-4,16,12]*t2[11,-1,15,10]*t2[11,-3,13,9]*t3[3,14,2,1]*t3[5,12,2,1]
        @tensor projvd[-1,-2,-3,-4] := cl[4,5,8]*cr[9,7,8]*cl[4,3,6]*cr[10,7,6]*T[12,16,-4,13]*T[14,16,-2,15]*
                            t2[2,1,12,5]*t2[2,1,14,3]*t3[9,13,-3,11]*t3[10,15,-1,11]
        @tensor projhr[-1,-2,-3,-4] := cl[3,4,7]*cr[8,-1,7]*t3[8,2,1,-2]*cl[3,4,5]*cr[6,-3,5]*t3[6,2,1,-4]
        @tensor projvu[-1,-2,-3,-4] := cl[-1,6,5]*cr[4,3,5]*cl[-3,7,8]*cr[4,3,8]*t2[-2,1,2,6]*t2[-4,1,2,7]
    elseif  direction == "RU"
        @tensor projcenter[-1,-2,-3,-4] := cl[3,4,-2]*cr[4,5,-4]*TL[6,7,-1]*TR[8,6,-3]*t1[2,1,3,7]*t4[8,5,1,2]
        @tensor projvu[-1,-2,-3,-4] := cl[9,7,6]*cr[4,5,6]*cl[11,7,8]*cr[4,3,8]*t1[-1,10,9,12]*
                            t4[13,5,2,1]*t1[-3,10,11,14]*t4[15,3,2,1]*T[-2,12,13,16]*T[-4,14,15,16]
        @tensor projhr[-1,-2,-3,-4] := cl[3,4,8]*cr[7,11,8]*cl[5,4,6]*cr[7,9,6]*t1[1,2,3,14]*t4[15,11,10,-3]*
                            t1[1,2,5,13]*t4[12,9,10,-1]*T[16,14,15,-4]*T[16,13,12,-2]
        @tensor projhl[-1,-2,-3,-4] := cl[8,-1,7]*cr[3,4,7]*cl[6,-3,5]*cr[3,4,5]*t1[1,-2,8,2]*t1[1,-4,6,2]
        @tensor projvd[-1,-2,-3,-4] := cl[4,3,5]*cr[-3,6,5]*cl[4,3,7]*cr[-1,8,7]*t4[1,6,-4,2]*t4[1,8,-2,2]

    end
    projcenter = reshape(projcenter,dlink^2*chicl3,dlink^2*chicl3)
    #println(maximum(projcenter-projcenter'))
    F = svd(projcenter)
    #println(F.S)
    if dlink^2*chicl3 > chimax
        PcL = reshape(F.U[:,1:chimax],dlink^2,chicl3,chimax)
        PcR = reshape(F.V[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(F.U,dlink^2,chicl3,dlink^2*chicl3)
        PcR = reshape(F.V,dlink^2,chicl3,dlink^2*chicl3)
    end

    projhl = reshape(projhl,chicl1*dlink,chicl1*dlink)
    #println(maximum(projh-projh'))
    F = svd((projhl+projhl')/2)
    dlink*chicl1 > chimax ? PhL = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                        PhL = reshape(F.U,chicl1,dlink,dlink*chicl1)

    x = 1

    projhr = reshape(projhr,chicl1*dlink,chicl1*dlink)
    #println(maximum(projh-projh'))
    F = svd((projhr+projhr')/2)
    dlink*chicl1 > chimax ? PhR = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                        PhR = reshape(F.U,chicl1,dlink,dlink*chicl1)
    x = 1
    projvu = reshape(projvu,chicl1*dlink,chicl1*dlink)
    #println(maximum(projv-projv'))
    F = svd((projvu+projvu')/2)
    dlink*chicl1 > chimax ? PvU = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                            PvU= reshape(F.U,chicl1,dlink,dlink*chicl1)
    x = 1
    projvd = reshape(projvd,chicl1*dlink,chicl1*dlink)
    #println(maximum(projv-projv'))
    F = svd((projvd+projvd')/2)
    dlink*chicl1 > chimax ? PvD = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                            PvD= reshape(F.U,chicl1,dlink,dlink*chicl1)

    return PcL,PcR,PhL,PhR,PvU,PvD

end

function NormalizeTensor(RGTensor,RGnorm)
    for j in 2:7
        append!(RGnorm[j],maximum(RGTensor[j]))
        if j == 4
            RGTensor[8] = RGTensor[8]/maximum(RGTensor[j])
        end
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
    T = reshape(permutedims(T,[1,2,4,3]),sizeT^2,sizeT^2)
    F = svd((T+T')/2)
    T1 = F.U*Matrix(Diagonal(sqrt.(F.S)))
    T2 = F.V*Matrix(Diagonal(sqrt.(F.S)))
    T1 = reshape(T1,sizeT,sizeT,sizeT^2)
    #T2 = permutedims(reshape(T2,sizeT,sizeT,sizeT^2),[2,1,3])
    T2 = reshape(T2,sizeT,sizeT,sizeT^2)
    return T1,T2
end


function CorseGraining(T::Array{Float64},NumLayer::Int64,chimax;Tsx=Float64[1.0 0.0; 0.0 -1.0])
    println("start corse Graining")

    #---- Diagonal corner tensor
    TL,TR = svdT(T)
    #TU,TD = svdT(permutedims(T,[4,1,2,3]))
    TL = TU; TD = TR
    #--- Boundary tensor
    # RGTensor[1]   base tensor T
    # RGTensor[2]   Center Tensor Left : cl
    # RGTensor[3]   Center Tensor Right : cr
    # RGTensor[4]   vertical up tensor : vtu
    # RGTensor[5]   horizontal left tensor : htl
    # RGTensor[6]   vertical dn tensor : vtd
    # RGTensor[7]   horizontal right tensor : htr
    RGTensor = Array{Array{Float64}}(undef,8)
    RGTensor[1] = T
    RGTensor[2] = TL;RGTensor[3] = TR
    RGTensor[4:end] = [T for j in 4:7]
    RGTensor[8] = Tsx
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
        PcL,PcR,PhL,PhR,PvU,PvD = Mover(RGTensor,TL,TR,chimax,direction="LD")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][4,6,8]*PcL[7,8,-3]*PhL[1,2,-2]*PvU[4,5,-1]*
                                        RGTensor[5][5,1,3,6]*TL[3,2,7]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][5,4,8]*PcR[7,8,-3]*PhR[4,6,-2]*PvD[1,2,-1]*
                                        RGTensor[6][5,3,1,6]*TR[2,3,7]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][-1,4,1,2]*T[1,5,-3,3]*PhL[4,5,-2]*PhL[2,3,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,1,5,-4]*T[2,-2,4,1]*PvU[3,2,-1]*PvU[5,4,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][3,1,5,-4]*T[2,-2,4,1]*PvD[3,2,-1]*PvD[5,4,-3]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][-1,4,1,2]*T[1,5,-3,3]*PhR[4,5,-2]*PhR[2,3,-4]
        if j == 1
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*Tsx[2,-2,4,1]*PvU[3,2,-1]*PvU[5,4,-3]
        else
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*T[2,-2,4,1]*PvU[3,2,-1]*PvU[5,4,-3]
        end
        end
        NormalizeTensor(RGTensor,RGnorm)

        @time begin
        PcL,PcR,PhL,PhR,PvU,PvD = Mover(RGTensor,TL,TR,chimax,direction="RU")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][6,4,7]*TL[3,2,8]*RGTensor[4][1,5,6,2]*
                                    PhL[4,5,-2]*PvU[1,3,-1]*PcL[8,7,-3]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][4,5,7]*TR[2,3,8]*RGTensor[7][2,5,6,1]*
                                    PhR[1,3,-2]*PvD[4,6,-1]*PcR[8,7,-3]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][1,5,-3,3]*T[-1,4,1,2]*PhL[5,4,-2]*PhL[3,2,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][2,-2,4,1]*T[3,1,5,-4]*PvU[2,3,-1]*PvU[4,5,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][2,-2,4,1]*T[3,1,5,-4]*PvD[2,3,-1]*PvD[4,5,-3]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][1,5,-3,3]*T[-1,4,1,2]*PhR[5,4,-2]*PhR[3,2,-4]
        @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][2,-2,4,1]*T[3,1,5,-4]*PvU[2,3,-1]*PvU[4,5,-3]
        end
        NormalizeTensor(RGTensor,RGnorm)
        #---- Normalize to avoid calculation explotion
        end
        #
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
        #println("Z ",Z[1])
        #=
        FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
                sum([j-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        =#
        #
        #=
        Ns = 2*(2*j+1)
        Lambda = FreeEnergyDensity(RGTensor[4],RGTensor[4])
        FE = (real(Lambda)+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*j+1))
        append!(FEttrg,[FE])
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
    chimax = 20
    BetaExact =1/2*log(1+sqrt(2))
    Beta = 0.3#0.9994*BetaExact
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

    @tensor TT[-1,-2,-3,-4,-5,-6,-7,-8] := T[-1,-3,2,1]*T[-2,1,4,-7]*T[2,-4,-5,3]*T[4,3,-6,-8]
    TT = reshape(TT,4,4,4,4)
    @tensor TTh[-1,-2,-3,-4,-5,-6] := T[-1,-2,1,-5]*T[1,-3,-4,-6]
    @tensor TTv[-1,-2,-3,-4,-5,-6] := T[-1,-3,-4,1]*T[-2,1,-5,-6]
    TTh = reshape(TTh,2,4,2,4)
    TTv = reshape(TTv,4,2,4,2)
    #TODO try to contract 4 T at first step and then do RG steps

    #for chimax in 10:10:30
    #for NumLayer in 100:100:100
    #chimax = 20
        NumLayer = 200
        Ns = 1+8*sum(1:1:NumLayer)
        println("before calculation")

    println("start corse Graining")

    #---- Diagonal corner tensor
    TL,TR = svdT(T)
    TU,TD = svdT(permutedims(T,[4,1,2,3]))
    #TU = TL; TD = TR

    TTL,TTR = svdT(TT)
    #TU,TD = svdT(permutedims(T,[4,1,2,3]))
    TTU = TTL; TTD = TTR
    #--- Boundary tensor
    # RGTensor[1]   base tensor T
    # RGTensor[2]   Center Tensor Left : cl
    # RGTensor[3]   Center Tensor Right : cr
    # RGTensor[4]   vertical up tensor : vtu
    # RGTensor[5]   vertical dn tensor : vtd
    # RGTensor[6]   horizontal left tensor : htl
    # RGTensor[7]   horizontal right tensor : htr
    RGTensor = Array{Array{Float64}}(undef,8)
    RGTensor[1] = T
    RGTensor[2] = TL;RGTensor[3] = TR
    RGTensor[4] = T
    RGTensor[5] = T
    RGTensor[6:7] = [T for j in 6:7]
    RGTensor[8] = Tsx
    #-- Norms of different tensor
    RGnorm = Array{Array{Float64}}(undef,7)
    RGnorm = [[] for j in 1:7]

    NumSite = Array{Int64}(undef,0)
    FEttrg = Array{Float64}(undef,0)
    FEttrg1 = Array{Float64}(undef,0)
    PcL = [];PcR = []
    println("start loop")
    for j in 1:NumLayer
        println("This is NumLayer $j")

        @time begin
        #---- compute projectors
        @time begin
        PcL,PcR,PhL,PhR,PvU,PvD = Mover(RGTensor,TL,TR,chimax,direction="LD")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][4,6,8]*PcL[7,8,-3]*PhL[1,2,-2]*PvU[4,5,-1]*
                                        RGTensor[5][5,1,3,6]*TL[3,2,7]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][5,4,8]*PcR[7,8,-3]*PhR[4,6,-2]*PvD[1,2,-1]*
                                        RGTensor[6][5,3,1,6]*TR[2,3,7]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][-1,4,1,2]*T[1,5,-3,3]*PhL[4,5,-2]*PhL[2,3,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,1,5,-4]*T[2,-2,4,1]*PvU[3,2,-1]*PvU[5,4,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][3,1,5,-4]*T[2,-2,4,1]*PvD[3,2,-1]*PvD[5,4,-3]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][-1,4,1,2]*T[1,5,-3,3]*PhR[4,5,-2]*PhR[2,3,-4]
        if j == 1
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*Tsx[2,-2,4,1]*PvU[3,2,-1]*PvU[5,4,-3]
        else
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*T[2,-2,4,1]*PvU[3,2,-1]*PvU[5,4,-3]
        end
        end
        NormalizeTensor(RGTensor,RGnorm)

        @time begin
        PcL,PcR,PhL,PhR,PvU,PvD = Mover(RGTensor,TL,TR,chimax,direction="RU")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][6,4,7]*TL[3,2,8]*RGTensor[4][1,5,6,2]*
                                    PhL[4,5,-2]*PvU[1,3,-1]*PcL[8,7,-3]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][4,5,7]*TR[2,3,8]*RGTensor[7][2,5,6,1]*
                                    PhR[1,3,-2]*PvD[4,6,-1]*PcR[8,7,-3]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][1,5,-3,3]*T[-1,4,1,2]*PhL[5,4,-2]*PhL[3,2,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][2,-2,4,1]*T[3,1,5,-4]*PvU[2,3,-1]*PvU[4,5,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][2,-2,4,1]*T[3,1,5,-4]*PvD[2,3,-1]*PvD[4,5,-3]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][1,5,-3,3]*T[-1,4,1,2]*PhR[5,4,-2]*PhR[3,2,-4]
        @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][2,-2,4,1]*T[3,1,5,-4]*PvU[2,3,-1]*PvU[4,5,-3]
        end
        NormalizeTensor(RGTensor,RGnorm)
        #---- Normalize to avoid calculation explotion
        end

        #
        internalenergytree = InternalEnergyTreeRG(RGTensor)
        append!(InternalEnergyTree,[internalenergytree])
        Ns = 1+8*sum(1:1:j)
        #Ns = 4*(j+1)^2
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
        println("Z ",Z[1])
        #
        FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
                sum([j-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(RGnorm[6][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[6][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[7][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[7][2:2:end])))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        #
        #
        #=
        Ns = 2*(2*j+1)
        Lambda = FreeEnergyDensity(RGTensor[4],RGTensor[4])
        FE = (real(Lambda)+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*j+1))
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        =#

    end












        #RGTensor,RGnorm,FEttrg,FEttrg1,numsite = CorseGraining(T,NumLayer,chimax,Sz=Tsx)


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








#=
FEtest = []
for j in 1:400
    Lambda = FreeEnergyDensity(RGTensor[4],RGTensor[4])
    FE = (real(Lambda)+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*NumLayer+1))
    append!(FEtest,[FE])
end
=#
    #
#end
