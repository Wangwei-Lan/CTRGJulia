include("../../2DClassical/partition.jl")
using TensorOperations
using LinearAlgebra
using Arpack
using KrylovKit


# TODO : When Do svd, try to differentiate F.U and F.V do not use the same one!!!

#-----------------------------------------------------------------------------
#
#
#                         tool  functions
#
#
#-----------------------------------------------------------------------------
#------- function ordered eigen value decomposition
function eigenorder(A::Array{Float64})
    F = eigen(A)
    order = sortperm(F.values,rev=true,by=abs)
    return F.vectors[:,order],F.values[order]
end

function ApplyVT(VT::Array{Float64},v::Array{Float64})
    @tensor v[-1,-2] = VT[1,4,-1,2]*VT[3,2,-2,4]*v[1,3]
    v = v/maximum(v)
    return v
end



"""
    svdT(T::Array{Float64})
    svd of base Tensor

                            -1
                            |
                            |
                      -2————T————-4
                            |
                            |
                            -3
"""
function svdT(T::Array{Float64})
    sizeT = size(T,1)
    T = reshape(permutedims(T,[1,2,4,3]),sizeT^2,sizeT^2)
    F = svd(T)
    T1 = F.U*Matrix(Diagonal(sqrt.(F.S)))
    T2 = F.V*Matrix(Diagonal(sqrt.(F.S)))
    T1 = reshape(T1,sizeT,sizeT,sizeT^2)
    T2 = permutedims(reshape(T2,sizeT,sizeT,sizeT^2),[2,1,3])
    #T2 = reshape(T2,sizeT,sizeT,sizeT^2)
    return T1,T2
end





#-------------------------------------------------------------------------------
#
#
#                    functions to obtain physical properties
#
#
#-------------------------------------------------------------------------------
#------TODO: need modify for better calculation results
function FreeEnergyDensity(RGTensor::Array{Array{Float64}})
    chi = size(RGTensor[4],1)
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1)
    eigup = reshape(eigvecsup[1],chi^2)
    eigdn = reshape(eigvecsdn[1],chi^2)
    Z = LinearAlgebra.dot(eigup,eigdn)

    @tensor Z[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*RGTensor[4][1,4,5,2]*RGTensor[4][3,2,6,4]
    return Z[1]/LinearAlgebra.dot(eigup,eigdn)
end

#--- TODO: need modify for 4 site
function InternalEnergyTreeRG(RGTensor::Array{Array{Float64}})
    chi = size(RGTensor[4],1)
    cl = RGTensor[2] ; cr = RGTensor[3]; t = RGTensor[1]
    t1 = RGTensor[4];t2 = RGTensor[5] ; t1e = RGTensor[8]
    #eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1,:LM,maxiter=300)
    #eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1,:LM,maxiter=300)
    @tensor Numerator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1e[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]
    @tensor Denominator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]

    return Numerator[1]/Denominator[1]
end





#-------------------------------------------------------------------------------
#
#
#
#
#
#-------------------------------------------------------------------------------
function  Mover(RGTensor::Array{Array{Float64}},TL::Array{Float64},TR::Array{Float64},chimax::Int64;direction=nothing)
    # cl : Center left Tensor
    # cr : Center Right Tensor
    cl = RGTensor[2];cr = RGTensor[3]
    #  t1 : vertical up tensor
    #  t2 : horizontal left tensor
    #  t3 : vertical dn tensor
    #  t4 : horizontal right tensor
    t1 = RGTensor[4];t2 = RGTensor[5];t3 = RGTensor[6];t4 = RGTensor[7]
    chicl1 = size(cl,1);chicl3 = size(cl,3);dlink = 2

    #----- Environment for obtaining projectors
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
    #projcenter = projcenter/maximum(projcenter)
    #projhl = projhl/maximum(projhl)
    #projhr = projhr/maximum(projhr)
    #projvd = projvd/maximum(projvd)
    #projvu = projvu/maximum(projvu)
    #---- svd to get projectors
    projcenter = reshape(projcenter,dlink^2*chicl3,dlink^2*chicl3)
    F = svd(projcenter)
    #eigvalues,eigvectors = eigenorder(projcenter)
    #println(eigvalues)
    if dlink^2*chicl3 > chimax
        PcL = reshape(F.U[:,1:chimax],dlink^2,chicl3,chimax)
        PcR = reshape(F.V[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(F.U,dlink^2,chicl3,dlink^2*chicl3)
        PcR = reshape(F.V,dlink^2,chicl3,dlink^2*chicl3)
    end



    projhl = reshape(projhl,chicl1*dlink,chicl1*dlink)
    F = svd((projhl+projhl')/2)
    if dlink*chicl1 > chimax
        PhL1 = reshape(F.U[:,1:chimax],chicl1,dlink,chimax)
        PhL2 = reshape(F.V[:,1:chimax],chicl1,dlink,chimax)
    else
        PhL1 = reshape(F.U,chicl1,dlink,dlink*chicl1)
        PhL2 = reshape(F.V,chicl1,dlink,dlink*chicl1)
    end
    #println(maximum(F.U-F.V))
    #println(maximum(abs.(F.U)-abs.(F.V)))
    #dlink*chicl1 > chimax ? PhL = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
    #                    PhL = reshape(F.U,chicl1,dlink,dlink*chicl1)
    #eigevectors,eigevalues = eigenorder((projhl+projhl')/2)
    #dlink*chicl1 > chimax ? PhL = reshape(eigevectors[:,1:chimax],chicl1,dlink,chimax) :
    #                    PhL = reshape(eigevectors,chicl1,dlink,dlink*chicl1)


    projhr = reshape(projhr,chicl1*dlink,chicl1*dlink)
    F = svd((projhr+projhr')/2)
    if dlink*chicl1 > chimax
        PhR1 = reshape(F.U[:,1:chimax],chicl1,dlink,chimax)
        PhR2 = reshape(F.V[:,1:chimax],chicl1,dlink,chimax)
    else
        PhR1 = reshape(F.U,chicl1,dlink,dlink*chicl1)
        PhR2 = reshape(F.V,chicl1,dlink,dlink*chicl1)
    end
    #println(maximum(F.U-F.V))
    #println(maximum(abs.(F.U)-abs.(F.V)))
    #dlink*chicl1 > chimax ? PhR = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
    #                    PhR = reshape(F.U,chicl1,dlink,dlink*chicl1)
    #eigevectors,eigevalues = eigenorder((projhr+projhr')/2)
    #dlink*chicl1 > chimax ? PhR = reshape(eigevectors[:,1:chimax],chicl1,dlink,chimax) :
    #                    PhR = reshape(eigevectors,chicl1,dlink,dlink*chicl1)


    projvu = reshape(projvu,chicl1*dlink,chicl1*dlink)
    F = svd((projvu+projvu')/2)
    if dlink*chicl1 > chimax
        PvU1 = reshape(F.U[:,1:chimax],chicl1,dlink,chimax)
        PvU2 = reshape(F.V[:,1:chimax],chicl1,dlink,chimax)
    else
       PvU1 = reshape(F.U,chicl1,dlink,dlink*chicl1)
       PvU2 = reshape(F.V,chicl1,dlink,dlink*chicl1)
    end
    #println(maximum(F.U-F.V))
    #println(maximum(abs.(F.U)-abs.(F.V)))
    #dlink*chicl1 > chimax ? PvU = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
    #                        PvU= reshape(F.U,chicl1,dlink,dlink*chicl1)
    #eigevectors,eigevalues = eigenorder((projvu+projvu')/2)
    #dlink*chicl1 > chimax ? PvU = reshape(eigevectors[:,1:chimax],chicl1,dlink,chimax) :
    #                        PvU= reshape(eigevectors,chicl1,dlink,dlink*chicl1)

    projvd = reshape(projvd,chicl1*dlink,chicl1*dlink)
    F = svd((projvd+projvd')/2)
    if dlink*chicl1 > chimax
        PvD1 = reshape(F.U[:,1:chimax],chicl1,dlink,chimax)
        PvD2 = reshape(F.V[:,1:chimax],chicl1,dlink,chimax)
    else
        PvD1 = reshape(F.U,chicl1,dlink,dlink*chicl1)
        PvD2 = reshape(F.V,chicl1,dlink,dlink*chicl1)
    end
    #println(maximum(abs.(F.U)-abs.(F.V)))
    #dlink*chicl1 > chimax ? PvD = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
    #                        PvD= reshape(F.U,chicl1,dlink,dlink*chicl1)
    #eigevectors,eigevalues = eigenorder((projvd+projvd')/2)
    #dlink*chicl1 > chimax ? PvD = reshape(eigevectors[:,1:chimax],chicl1,dlink,chimax) :
    #                        PvD= reshape(eigevectors,chicl1,dlink,dlink*chicl1)

    return PcL,PcR,PhL1,PhL2,PhR1,PhR2,PvU1,PvU2,PvD1,PvD2
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




function CorseGraining(T::Array{Float64},NumLayer::Int64,chimax;Tsx=Float64[1.0 0.0; 0.0 -1.0])
    println("start corse Graining")

    #---- Diagonal corner tensor
    TL,TR = svdT(T)
    #TU,TD = svdT(permutedims(T,[4,1,2,3]))
    TU = TL; TD = TR
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
        PcL,PcR,PhL1,PhL2,PhR1,PhR2,PvU1,PvU2,PvD1,PvD2 = Mover(RGTensor,TL,TR,chimax,direction="LD")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][4,6,8]*PcL[7,8,-3]*PhL1[1,2,-2]*PvU1[4,5,-1]*
                                        RGTensor[5][5,1,3,6]*TL[3,2,7]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][5,4,8]*PcR[7,8,-3]*PhR1[4,6,-2]*PvD1[1,2,-1]*
                                        RGTensor[6][5,3,1,6]*TR[2,3,7]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][-1,4,1,2]*T[1,5,-3,3]*PhL1[4,5,-2]*PhL1[2,3,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,1,5,-4]*T[2,-2,4,1]*PvU1[3,2,-1]*PvU1[5,4,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][3,1,5,-4]*T[2,-2,4,1]*PvD1[3,2,-1]*PvD1[5,4,-3]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][-1,4,1,2]*T[1,5,-3,3]*PhR1[4,5,-2]*PhR1[2,3,-4]
        if j == 1
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*Tsx[2,-2,4,1]*PvU1[3,2,-1]*PvU1[5,4,-3]
        else
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*T[2,-2,4,1]*PvU1[3,2,-1]*PvU1[5,4,-3]
        end
        end
        NormalizeTensor(RGTensor,RGnorm)

        @time begin
        PcL,PcR,PhL1,PhL2,PhR1,PhR2,PvU1,PvU2,PvD1,PvD2 = Mover(RGTensor,TL,TR,chimax,direction="RU")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][6,4,7]*TL[3,2,8]*RGTensor[4][1,5,6,2]*
                                    PhL1[4,5,-2]*PvU1[1,3,-1]*PcL[8,7,-3]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][4,5,7]*TR[2,3,8]*RGTensor[7][2,5,6,1]*
                                    PhR1[1,3,-2]*PvD1[4,6,-1]*PcR[8,7,-3]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][1,5,-3,3]*T[-1,4,1,2]*PhL1[5,4,-2]*PhL1[3,2,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][2,-2,4,1]*T[3,1,5,-4]*PvU1[2,3,-1]*PvU1[4,5,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][2,-2,4,1]*T[3,1,5,-4]*PvD1[2,3,-1]*PvD1[4,5,-3]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][1,5,-3,3]*T[-1,4,1,2]*PhR1[5,4,-2]*PhR1[3,2,-4]
        @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][2,-2,4,1]*T[3,1,5,-4]*PvU1[2,3,-1]*PvU1[4,5,-3]
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

#----- Fixed paramenters
sx = Float64[0.0 1.0;
            1.0  0.0]
sz = Float64[1.0 0.0;
            0.0  -1.0]

#-----  Set Parameter
Dlink = 2


#-----    Store physical results
InternalEnergyTree = Array{Float64}(undef,0)
InternalEnergyExact = Array{Float64}(undef,0)
FreeEnergyTree = Array{Float64}(undef,0)
FreeEnergyExact = Array{Float64}(undef,0)
feexact = 0.0
internalenergyexact = 0.0
NumSite = Array{Int64}(undef,0)
FEttrg = Array{Float64}(undef,0)
PcL = [];PcR = []

#-------- Start Calculation

#for chimax in 10:10:40
#for Beta in 0.44:0.1:0.44



    #----- change parameter
    chimax = 10
    BetaExact =1/2*log(1+sqrt(2))
    Beta = 0.3#0.9994*BetaExact
    println("This is Beta $Beta")

    #-- exact results
    global feexact =  ComputeFreeEnergy(Beta)
    global internalenergyexact = ComputeInternalEnergy(Beta)

    #----- Base Tensor Test from 2D
    Q = Float64[exp(Beta) exp(-Beta);
                exp(-Beta) exp(Beta)]
    g = zeros(2,2,2,2)
    g[1,1,1,1] = g[2,2,2,2] = 1
    gp = zeros(2,2,2,2)
    gp[1,1,1,1] = -1;gp[2,2,2,2] = 1
    @tensor T[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*g[1,2,3,4]
    @tensor Tsx[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*gp[1,2,3,4]


    #---- NumSite and NumLayer
    NumLayer = 200
    Ns = 1+8*sum(1:1:NumLayer)



    #---- Diagonal corner tensor
    TL,TR = svdT(T)
    TU,TD = svdT(permutedims(T,[4,1,2,3]))
    #TU = TL; TD = TR

    #--- Boundary tensor
    # RGTensor[1]   base tensor T
    # RGTensor[2]   Center Tensor Left : cl
    # RGTensor[3]   Center Tensor Right : cr
    # RGTensor[4]   vertical up tensor : vtu
    # RGTensor[5]   horizontal left tensor : htl
    # RGTensor[6]   vertical dn tensor : vtd
    # RGTensor[7]   horizontal right tensor : htr
    # RGTensor[8]   Used for internal energy calculation
    RGTensor = Array{Array{Float64}}(undef,8)
    RGTensor[1] = T
    RGTensor[2] = TL;RGTensor[3] = TR
    RGTensor[4:7] = [T for j in 4:7]
    RGTensor[8] = Tsx

    RGnorm = Array{Array{Float64}}(undef,7) # norm for different tensors
    RGnorm = [[] for j in 1:7]



    println("start corse Graining")
    for j in 1:NumLayer
        println("This is NumLayer $j")

        @time begin
        #---- compute projectors
        @time begin
        PcL,PcR,PhL1,PhL2,PhR1,PhR2,PvU1,PvU2,PvD1,PvD2 = Mover(RGTensor,TL,TR,chimax,direction="LD")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][4,6,8]*PcL[7,8,-3]*PhL1[1,2,-2]*PvU1[4,5,-1]*
                                        RGTensor[5][5,1,3,6]*TL[3,2,7]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][5,4,8]*PcR[7,8,-3]*PhR1[4,6,-2]*PvD1[1,2,-1]*
                                        RGTensor[6][5,3,1,6]*TR[2,3,7]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][-1,4,1,2]*T[1,5,-3,3]*PhL1[4,5,-2]*PhL1[2,3,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,1,5,-4]*T[2,-2,4,1]*PvU1[3,2,-1]*PvU1[5,4,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][3,1,5,-4]*T[2,-2,4,1]*PvD1[3,2,-1]*PvD1[5,4,-3]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][-1,4,1,2]*T[1,5,-3,3]*PhR1[4,5,-2]*PhR1[2,3,-4]
        if j == 1
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*Tsx[2,-2,4,1]*PvU1[3,2,-1]*PvU1[5,4,-3]
        else
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*T[2,-2,4,1]*PvU1[3,2,-1]*PvU1[5,4,-3]
        end
        end
        NormalizeTensor(RGTensor,RGnorm)

        @time begin
        PcL,PcR,PhL1,PhL2,PhR1,PhR2,PvU1,PvU2,PvD1,PvD2 = Mover(RGTensor,TL,TR,chimax,direction="RU")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][6,4,7]*TL[3,2,8]*RGTensor[4][1,5,6,2]*
                                    PhL1[4,5,-2]*PvU1[1,3,-1]*PcL[8,7,-3]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][4,5,7]*TR[2,3,8]*RGTensor[7][2,5,6,1]*
                                    PhR1[1,3,-2]*PvD1[4,6,-1]*PcR[8,7,-3]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][1,5,-3,3]*T[-1,4,1,2]*PhL1[5,4,-2]*PhL1[3,2,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][2,-2,4,1]*T[3,1,5,-4]*PvU1[2,3,-1]*PvU1[4,5,-3]
        @tensor RGTensor[6][-1,-2,-3,-4] := RGTensor[6][2,-2,4,1]*T[3,1,5,-4]*PvD1[2,3,-1]*PvD1[4,5,-3]
        @tensor RGTensor[7][-1,-2,-3,-4] := RGTensor[7][1,5,-3,3]*T[-1,4,1,2]*PhR1[5,4,-2]*PhR1[3,2,-4]
        @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][2,-2,4,1]*T[3,1,5,-4]*PvU1[2,3,-1]*PvU1[4,5,-3]
        end
        NormalizeTensor(RGTensor,RGnorm)
        #---- Normalize to avoid calculation explotion
        end

        # Compute internal energy of tree
        internalenergytree = InternalEnergyTreeRG(RGTensor)
        append!(InternalEnergyTree,[internalenergytree])

        #----- Compute Free Energy
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]

        println("Z ",Z[1])

        FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
                sum([j-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(RGnorm[6][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[6][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[7][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[7][2:2:end])))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])

    end


#end
