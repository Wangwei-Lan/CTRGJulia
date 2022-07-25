include("../../2DClassical/partition.jl")
using TensorOperations
using LinearAlgebra
using Arpack
using KrylovKit



# TODO :: Mixed Gauge or Mixed Isometry for Center Projector
# TODO :: Internal Energy
# TODO :: MPS, try to make it better for periodic and non periodic cases!!!


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
    @tensor v[-1,-2] := VT[1,4,-1,2]*VT[3,2,-2,4]*v[1,3]
    return v
end

function ApplyTensor(CL::Array{Float64},CR::Array{Float64})

end



function FreeEnergyDensity(RGTensor::Array{Array{Float64}})
    chi = size(RGTensor[4],1)
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1)
    eigup = reshape(eigvecsup[1],chi^2)
    eigdn = reshape(eigvecsdn[1],chi^2)
    Z = LinearAlgebra.dot(eigup,eigdn)
    @tensor Z1[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*RGTensor[4][1,4,5,2]*RGTensor[4][3,2,6,4]

    return Z1[1]/LinearAlgebra.dot(eigup,eigdn)
end



function InternalEnergyTreeRG(RGTensor::Array{Array{Float64}})
    chi = size(RGTensor[4],1)
    cl = RGTensor[2] ; cr = RGTensor[3]; t = RGTensor[1]
    t1 = RGTensor[4];t2 = RGTensor[5] ; t1e = RGTensor[8]
    #
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1,:LM,maxiter=300)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1,:LM,maxiter=300)
    @tensor Numerator1[] := real(eigvecsup)[1][1,3]*real(eigvecsdn)[1][6,5]*RGTensor[4][1,4,6,2]*RGTensor[8][3,2,5,4]
    @tensor Denominator1[] := real(eigvecsup[1])[1,3]*real(eigvecsdn)[1][6,5]*RGTensor[4][1,4,6,2]*RGTensor[4][3,2,5,4]
    #
    #
    @tensor Numerator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1e[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]
    @tensor Denominator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]
    #
    return Numerator[1]/Denominator[1],Numerator1[1]/Denominator1[1]
end





function  Mover(RGTensor::Array{Array{Float64}},TL::Array{Float64},TR::Array{Float64},chimax::Int64;direction=nothing)
    # cl : Center left Tensor
    # cr : Center Right Tensor
    T = RGTensor[1]
    cl = RGTensor[2];cr = RGTensor[3]
    #  t1 : vertical up tensor
    #  t2 : vertical dn tensor
    #  t3 : horizontal left tensor
    #  t4 : horizontal right tensor
    t1 = RGTensor[4];t2 = RGTensor[5];t3 = RGTensor[6];t4 = RGTensor[7]
    chicl1 = size(cl,1);chicl3 = size(cl,3);dlink = 2

    if direction == "LD"
        @tensor projcenter[-1,-2,-3,-4] := cl[4,3,-2]*cr[5,4,-4]*TL[6,7,-1]*TR[7,8,-3]*t2[1,2,6,3]*t1[5,8,2,1]
        @tensor projcenter1[-1,-2,-3,-4] := cl[4,3,-2]*cl[4,5,-4]*TL[6,7,-1]*TL[8,7,-3]*t2[1,2,6,3]*t2[1,2,8,5]
        @tensor projcenter2[-1,-2,-3,-4] := cr[5,4,-4]*cr[3,4,-2]*TR[7,8,-3]*TR[7,6,-1]*t1[5,8,2,1]*t1[3,6,2,1]
        @tensor projhorizontal[-1,-2,-3,-4] := cl[7,10,6]*cr[3,4,6]*cl[7,9,8]*cr[5,4,8]*T[15,-2,16,14]*
                            T[13,-4,16,12]*t2[11,-1,15,10]*t2[11,-3,13,9]*t1[3,14,2,1]*t1[5,12,2,1]
        @tensor projvertical[-1,-2,-3,-4] := cl[4,5,8]*cr[9,7,8]*cl[4,3,6]*cr[10,7,6]*T[12,16,-4,13]*T[14,16,-2,15]*
                            t2[2,1,12,5]*t2[2,1,14,3]*t1[9,13,-3,11]*t1[10,15,-1,11]
    elseif  direction == "RU"
        @tensor projcenter[-1,-2,-3,-4] := cl[3,4,-2]*cr[4,5,-4]*TL[6,7,-1]*TR[8,6,-3]*t1[2,1,3,7]*t2[8,5,1,2]
        @tensor projcenter1[-1,-2,-3,-4] := cl[3,4,-2]*cl[5,4,-4]*TL[6,7,-1]*TL[6,8,-3]*t1[2,1,3,7]*t1[2,1,5,8]
        @tensor projcenter2[-1,-2,-3,-4] := cr[4,5,-4]*cr[4,3,-2]*TR[8,6,-3]*TR[7,6,-1]*t2[8,5,1,2]*t2[7,3,1,2]
        @tensor projvertical[-1,-2,-3,-4] := cl[9,7,6]*cr[4,5,6]*cl[11,7,8]*cr[4,3,8]*t1[-1,10,9,12]*
                            t2[13,5,2,1]*t1[-3,10,11,14]*t2[15,3,2,1]*T[-2,12,13,16]*T[-4,14,15,16]
        @tensor projhorizontal[-1,-2,-3,-4] := cl[3,4,8]*cr[7,11,8]*cl[5,4,6]*cr[7,9,6]*t1[1,2,3,14]*t2[15,11,10,-3]*
                            t1[1,2,5,13]*t2[12,9,10,-1]*T[16,14,15,-4]*T[16,13,12,-2]
    end
    #
    projcenter = reshape(projcenter,dlink^2*chicl3,dlink^2*chicl3)
    F = svd((projcenter+projcenter')/2)
    if dlink^2*chicl3 > chimax
        PcL = reshape(F.U[:,1:chimax],dlink^2,chicl3,chimax)
        PcR = reshape(F.V[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(F.U,dlink^2,chicl3,dlink^2*chicl3)
        PcR = reshape(F.V,dlink^2,chicl3,dlink^2*chicl3)
    end
    #
    #
    projcenter1 = reshape(projcenter1,dlink^2*chicl3,dlink^2*chicl3)
    F1 = svd((projcenter1+projcenter1')/2)
    if dlink^2*chicl3 > chimax
        PcL = reshape(F1.U[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(F1.U,dlink^2,chicl3,dlink^2*chicl3)
    end
    projcenter2 = reshape(projcenter2,dlink^2*chicl3,dlink^2*chicl3)
    F2 = svd((projcenter2+projcenter2')/2)
    if dlink^2*chicl3 > chimax
        PcR = reshape(F2.U[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcR = reshape(F2.U,dlink^2,chicl3,dlink^2*chicl3)
    end

    @tensor mix[-1,-2] := PcL[1,2,-1]*PcR[1,2,-2]
    F3 = svd(mix)
    @tensor PcL[-1,-2,-3] := PcL[-1,-2,1]*F3.U[1,2]*sqrt(inv(Matrix(Diagonal(F3.S))))[2,-3]
    @tensor PcR[-1,-2,-3] := sqrt(inv(Matrix(Diagonal(F3.S))))[-3,1]*F3.V[2,1]*PcR[-1,-2,2]
    #

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
        if j == 4
            RGTensor[8] = RGTensor[8]/maximum(RGTensor[j])
        end
        RGTensor[j] = RGTensor[j]/maximum(RGTensor[j])
    end
end



function svdT(T::Array{Float64})
    sizeT = size(T,1)
    T = reshape(permutedims(T,[1,2,4,3]),sizeT^2,sizeT^2)
    F = svd((T+T')/2)
    T1 = F.U*Matrix(Diagonal(sqrt.(F.S)))
    T2 = F.V*Matrix(Diagonal(sqrt.(F.S)))
    T1 = reshape(T1,sizeT,sizeT,sizeT^2)
    T2 = permutedims(reshape(T2,sizeT,sizeT,sizeT^2),[2,1,3])
    #T2 = reshape(T2,sizeT,sizeT,sizeT^2)
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
    # RGTensor[5]   vertical dn tensor : vtd
    # RGTensor[6]   horizontal left tensor : htl
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
        PcL,PcR,Ph,Pv = Mover(RGTensor,TL,TR,chimax,direction="LD")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][4,6,8]*PcL[7,8,-3]*Ph[1,2,-2]*Pv[4,5,-1]*
                                        RGTensor[5][5,1,3,6]*TL[3,2,7]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][5,4,8]*PcR[7,8,-3]*Ph[4,6,-2]*Pv[1,2,-1]*
                                        RGTensor[4][5,3,1,6]*TR[2,3,7]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][-1,4,1,2]*T[1,5,-3,3]*Ph[4,5,-2]*Ph[2,3,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,1,5,-4]*T[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        if j == 1
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*Tsx[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        else
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*T[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        end
        end
        NormalizeTensor(RGTensor,RGnorm)

        @time begin
        PcL,PcR,Ph,Pv = Mover(RGTensor,TL,TR,chimax,direction="RU")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][6,4,7]*TL[3,2,8]*RGTensor[4][1,5,6,2]*
                                    Ph[4,5,-2]*Pv[1,3,-1]*PcL[8,7,-3]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][4,5,7]*TR[2,3,8]*RGTensor[5][2,5,6,1]*
                                    Ph[1,3,-2]*Pv[4,6,-1]*PcR[8,7,-3]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][1,5,-3,3]*T[-1,4,1,2]*Ph[5,4,-2]*Ph[3,2,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][2,-2,4,1]*T[3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][2,-2,4,1]*T[3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        end
        NormalizeTensor(RGTensor,RGnorm)
        #---- Normalize to avoid calculation explotion
        end
        #
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
        #println("Z ",Z[1])
        FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
                sum([j-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
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
InternalEnergyTree1 = Array{Float64}(undef,0)
InternalEnergyExact = Array{Float64}(undef,0)
FreeEnergyExact = Array{Float64}(undef,0)
FreeEnergyTree = Array{Float64}(undef,0)
PhMatrix = []
PvMatrix = []
ScMatrix = []
SuMatrix = []
SlMatrix = []


#for chimax in 10:10:100
#for Beta in 0.30:0.02:0.55
#for temperature in 2.30:0.01:2.30
    temperature = 2.25
    chimax = 30
    BetaExact =1/2*log(1+sqrt(2))
    #Beta = 0.9994*BetaExact
    Beta = 1/temperature
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
    @tensor TTvsx[-1,-2,-3,-4,-5,-6] := Tsx[-1,-3,-4,1]*Tsx[-2,1,-5,-6]
    TTh = reshape(TTh,2,4,2,4)
    TTv = reshape(TTv,4,2,4,2)
    TTvsx = reshape(TTvsx,4,2,4,2)
    #TODO try to contract 4 T at first step and then do RG steps

    #for chimax in 10:10:30
    #for NumLayer in 100:100:100
    #chimax = 20
    NumLayer = 1000
    Ns = 1+8*sum(1:1:NumLayer)
    println("before calculation")

    println("start corse Graining")
    #---- Diagonal corner tensor
    TL,TR = svdT(T)
    TU,TD = svdT(permutedims(T,[4,1,2,3]))
    #TU = TL; TD = TR

    TTL,TTR = svdT(TT)
    #TTU,TTD = svdT(permutedims(TT,[4,1,2,3]))
    #TTU = TTL; TTD = TTR
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
    #=
    RGTensor[2] = TTL;RGTensor[3] = TTR
    RGTensor[4] = TTv
    RGTensor[5] = TTh
    RGTensor[8] = TTvsx
    =#
    #
    RGTensor[2] = TL;RGTensor[3] = TR
    RGTensor[4] = T
    RGTensor[5] = T
    RGTensor[8] = Tsx
    #
    RGTensor[6:7] = [T for j in 6:7]
    #-- Norms of different tensor
    RGnorm = Array{Array{Float64}}(undef,7)
    RGnorm = [[] for j in 1:7]

    NumSite = Array{Int64}(undef,0)
    FEttrg = Array{Float64}(undef,0)
    FEttrg1 = Array{Float64}(undef,0)
    PcLMatrix = [];PcRMatrix = []
    PhMatrix = Array{Array{Float64}}(undef,0);
    PvMatrix = Array{Array{Float64}}(undef,0)
    println("start loop")
    for j in 1:NumLayer
        println("This is NumLayer $j")

        @time begin
        #---- compute projectors
        #@time begin
        PcL,PcR,Ph,Pv = Mover(RGTensor,TL,TR,chimax,direction="LD")
        append!(PhMatrix,[Ph])
        append!(PvMatrix,[Pv])
        #@tensor test[-1,-2] := PcL[1,2,-1]*PcR[1,2,-2]
        #println(norm(test-Matrix(1.0I,size(test))))
        append!(PcLMatrix,[PcL])
        append!(PcRMatrix,[PcR])
        #PcR = PcL
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][4,6,8]*PcL[7,8,-3]*Ph[1,2,-2]*Pv[4,5,-1]*
                                        RGTensor[5][5,1,3,6]*TL[3,2,7]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][5,4,8]*PcR[7,8,-3]*Ph[4,6,-2]*Pv[1,2,-1]*
                                        RGTensor[4][5,3,1,6]*TR[2,3,7]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][-1,4,1,2]*T[1,5,-3,3]*Ph[4,5,-2]*Ph[2,3,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,1,5,-4]*T[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        #
        if j == 1
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*Tsx[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        else
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*T[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        end
        #
        #end
        NormalizeTensor(RGTensor,RGnorm)

        #@time begin
        PcL,PcR,Ph,Pv = Mover(RGTensor,TL,TR,chimax,direction="RU")
        append!(PhMatrix,[Ph])
        append!(PvMatrix,[Pv])
        #PcR = PcL
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][6,4,7]*TL[3,2,8]*RGTensor[4][1,5,6,2]*
                                    Ph[4,5,-2]*Pv[1,3,-1]*PcL[8,7,-3]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][4,5,7]*TR[2,3,8]*RGTensor[5][2,5,6,1]*
                                    Ph[1,3,-2]*Pv[4,6,-1]*PcR[8,7,-3]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][1,5,-3,3]*T[-1,4,1,2]*Ph[5,4,-2]*Ph[3,2,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][2,-2,4,1]*T[3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][2,-2,4,1]*T[3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        #end
        NormalizeTensor(RGTensor,RGnorm)
        #---- Normalize to avoid calculation explotion
        end
        #

        #
        #Ns = (2*(j+1))^2
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
        FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
                sum([j-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        #
        #
        #=
        Ns = 2*(2*j+1)
        Lambda = FreeEnergyDensity(RGTensor[4],RGTensor[4])
        FE = (log(real(Lambda))+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*j+1))
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        =#

    end
    #internalenergytree,internalenergytree1 = InternalEnergyTreeRG(RGTensor)
    #
    @tensor Numerator[] := Tsx[17,3,5,1]*T[19,2,7,3]*Tsx[15,11,17,9]*T[16,10,19,11]*RGTensor[4][18,1,4,2]*
                        RGTensor[4][12,9,18,10]*RGTensor[5][5,8,15,6]*RGTensor[5][7,14,16,8]*
                        RGTensor[2][4,6,13]*RGTensor[3][12,14,13]
    @tensor Denominator[] := T[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*RGTensor[4][18,1,4,2]*
                        RGTensor[4][12,9,18,10]*RGTensor[5][5,8,15,6]*RGTensor[5][7,14,16,8]*
                        RGTensor[2][4,6,13]*RGTensor[3][12,14,13]
    internalenergytree = Numerator[1]/Denominator[1]
    #
    append!(InternalEnergyTree,[internalenergytree])
    internalenergyexact = ComputeInternalEnergy(Beta)
    append!(InternalEnergyExact,internalenergyexact)
    #append!(InternalEnergyTree1,[internalenergytree1])

    #Ns = (2*(NumLayer+1))^2
    #=
    Ns = 1+8*sum(1:1:NumLayer)
    @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
    FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
    append!(FreeEnergyTree,[FE])
    append!(NumSite,[Ns])
    =#


#end




        #RGTensor,RGnorm,FEttrg,FEttrg1,numsite = CorseGraining(T,NumLayer,chimax,Sz=Tsx)
        #@tensor Z[] := CenterLeft[1,2,3]*CenterRight[1,2,3]
        #FE = (log(Z[1])+sum(log.(CLnorm))+sum(log.(CRnorm))+2*sum([NumLayer-1:-1:0...].*log.(HTnorm))+
        #                2*sum([(NumLayer-1):-1:0...].*log.(VTnorm)))/Ns
        #append!(FreeEnergyTree,[FE])
    #end
    #=

    =#
    #Lambda = FreeEnergyDensity(RGTensor)
    #FE = (log(real(Lambda))+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*NumLayer+2))
    #append!(FreeEnergyTree,[FE])
#end



#=
#-----------------------  Below is calculation of internal energy vs sites

function ApplyT(T::Array{Float64},v::Array{Float64})
    @tensor v[-1] := T[1,2,-1,2]*v[1]
    return v
end

function ComputeIE(Proj::Array{Array{Float64}},RGTensor::Array{Array{Float64}},
                        Tsx::Array{Float64},Hamiltonian::Array{Float64},site::Int64)
    #eigvalup,eigvecsup = eigsolve(y->ApplyT(RGTensor[4],y),rand(size(RGTensor[4],1)))
    #eigvaldn,eigvecsdn = eigsolve(y->ApplyT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(size(RGTensor[4],1)))
    sitetot = size(Proj,1) + 1
    T = RGTensor[1]
    Proj = reverse(Proj)
    #=
    @tensor Env[-1,-2] := real(eigvecsdn[1])[2]*real(eigvecsup[1])[1]*Proj[1][-1,3,1]*Proj[1][-2,3,2]
    @tensor Denominator[] := real(eigvecsdn[1])[1]*real(eigvecsup[1])[1]
    Denominator = Denominator[1]/maximum(Env)
    Env = Env/maximum(Env)
    for j in 2:2*site-1
        @tensor Env[-1,-2] := Env[1,2]*Proj[j][-1,3,1]*Proj[j][-2,3,2]
        Denominator = Denominator/maximum(Env)
        Env = Env/maximum(Env)
    end
    @tensor Numerator[] := Env[1,2]*Proj[2*site][5,3,1]*Proj[2*site+1][10,7,5]*Proj[2*site+2][12,11,10]*
                         Proj[2*site][6,4,2]*Proj[2*site+1][8,7,6]*Proj[2*site+2][12,9,8]*Hamiltonian[3,11,4,9]
    internalenergy = Numerator[1]/Denominator
    =#
    #@tensor Denominator[-1,-2,-3,-4] := real(eigvecsup[1])[1]*real(eigvecsdn[1])[6]*Proj[1][2,3,1]*Proj[1][7,9,6]*
    #                    Proj[2][-1,4,2]*Proj[2][-3,8,7]*T[4,5,8,-2]*T[3,-4,9,5]
    @tensor Denominator[-1,-2,-3,-4] := Proj[1][4,2,1]*Proj[1][8,3,1]*Proj[2][-1,5,4]*Proj[2][-3,7,8]*T[5,6,7,-2]*T[2,-4,3,6]
    Denominator = Denominator/maximum(Denominator)

    for j in 3:2*site-1
        #println("j is $j")
        if j%2 == 1
            @tensor Denominator[-1,-2,-3,-4] := Denominator[1,-2,4,2]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,-4,5,2]
        else
            @tensor Denominator[-1,-2,-3,-4] := Denominator[1,2,4,-4]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,2,5,-2]
        end
        Denominator = Denominator/maximum(Denominator)
    end

    @tensor Numerator[-1,-2,-3,-4] := Denominator[1,3,4,8]*Proj[2*site][6,2,1]*Proj[2*site][9,5,4]*Proj[2*site+1][11,7,6]*
                        Proj[2*site+1][14,10,9]*Proj[2*site+2][-1,13,11]*Proj[2*site+2][-3,15,14]*Tsx[2,3,5,12]*Tsx[13,12,15,-2]*
                        T[7,-4,10,8]
    @tensor Denominator[-1,-2,-3,-4] := Denominator[1,3,4,8]*Proj[2*site][6,2,1]*Proj[2*site][9,5,4]*Proj[2*site+1][11,7,6]*
                        Proj[2*site+1][14,10,9]*Proj[2*site+2][-1,13,11]*Proj[2*site+2][-3,15,14]*T[2,3,5,12]*T[13,12,15,-2]*
                        T[7,-4,10,8]
    Numerator = Numerator/maximum(Denominator)
    Denominator = Denominator/maximum(Denominator)
    #println("Whoaw j is ",2*site)
    #println("j is ",2*site+1)
    #println("j is ",2*site+2)

    for j in 2*site+3:size(Proj,1)-1
        #println("j is $j")
        if j %2 == 1
            @tensor Numerator[-1,-2,-3,-4] := Numerator[1,-2,4,2]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,-4,5,2]
            @tensor Denominator[-1,-2,-3,-4] := Denominator[1,-2,4,2]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,-4,5,2]
        else
            @tensor Numerator[-1,-2,-3,-4] := Numerator[1,2,4,-4]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,2,5,-2]
            @tensor Denominator[-1,-2,-3,-4] := Denominator[1,2,4,-4]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,2,5,-2]
        end
        Numerator = Numerator/maximum(Denominator)
        Denominator = Denominator/maximum(Denominator)
    end
    @tensor Numerator[] := Numerator[1,2,7,4]*Proj[end][5,3,1]*Proj[end][9,8,7]*T[3,2,8,6]*T[5,6,9,4]
    @tensor Denominator[] := Denominator[1,2,7,4]*Proj[end][5,3,1]*Proj[end][9,8,7]*T[3,2,8,6]*T[5,6,9,4]

    internalenergy = Numerator[1]/Denominator[1]
    return internalenergy

end

@tensor Hamiltonian[-1,-2,-3,-4] := sx[-1,-3]*sx[-2,-4]
InternalEnergyTree = []
for j in 2:199
    #println("This is loop $j")
    append!(InternalEnergyTree,[ComputeIE(PvMatrix,RGTensor,Tsx,Hamiltonian,j)])
end
test = (-2*InternalEnergyTree.-InternalEnergyExact)./InternalEnergyExact
=#
