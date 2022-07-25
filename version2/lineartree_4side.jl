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



# cl : CenterLeft
# cr : CenterRight
# t1 : tensor up or left
# t2 : tensor dn or right
function  Mover(t1::Array{Float64},t2::Array{Float64},cl::Array{Float64},
                        cr::Array{Float64},chimax::Int64;direction = nothing)
    dlink = minimum(size(t1));chi = size(cl,1)

    if direction == "L"
        @tensor temp[-1,-2,-3,-4,-5,-6] := cl[8,-2,9]*cr[5,6,9]*cl[11,-5,10]*cr[7,6,10]*
                t1[1,-1,8,2]*t1[1,-4,11,2]*t2[5,-3,3,4]*t2[7,-6,3,4]
    elseif direction == "R"
        @tensor temp[-1,-2,-3,-4,-5,-6] := cl[5,6,9]*cr[8,-5,9]*cl[7,6,10]*cr[11,-2,10]*
                    t1[3,4,5,-4]*t2[8,2,1,-6]*t1[3,4,7,-1]*t2[11,2,1,-3]
    elseif direction == "U"
        @tensor temp[-1,-2,-3,-4,-5,-6] := cl[-2,11,10]*cr[6,7,10]*cl[-5,8,9]*cr[6,5,9]*
                    t1[-3,1,2,11]*t2[-1,7,4,3]*t1[-6,1,2,8]*t2[-4,5,4,3]
    elseif direction == "D"
        @tensor temp[-1,-2,-3,-4,-5,-6] := cl[6,7,10]*cr[-5,11,10]*cl[6,5,9]*cr[-2,8,9]*
                    t1[4,3,-6,7]*t2[2,11,-4,1]*t1[4,3,-3,5]*t2[2,8,-1,1]
    end
    temp = reshape(temp,dlink^2*chi,dlink^2*chi)
    F = svd( (temp + temp')/2)
    dlink^2*chi > chimax ? Proj = reshape(F.U[:,1:chimax],dlink,chi,dlink,chimax) :
                        Proj = reshape(F.U,dlink,chi,dlink,dlink^2*chi)
    return Proj
end



function  CenterProjector(T::Array{Float64},CenterLeft::Array{Float64},CenterRight::Array{Float64},
                HTL::Array{Float64},HTR::Array{Float64},VTU::Array{Float64},VTD::Array{Float64},
                Left::Array{Float64},Right::Array{Float64},chimax::Int64)
    dlink = size(Left,1);chi = size(CenterLeft,3)
    @tensor temp[-1,-2,-3,-4,-5,-6] := CenterLeft[3,4,-2]*CenterRight[4,3,-5]*Left[1,2,-3]*
                                Left[6,5,-1]*Right[2,1,-6]*Right[5,6,-4]
    #@tensor temp[-1,-2,-3,-4,-5,-6] :=
    #@tensor temp[-1,-2,-3,-4,-5,-6] := T[10,9,14,11]*T[12,15,9,10]*HTL[14,3,2,18]*HTR[7,13,12,8]*VTU[8,11,17,6]*
    #                VTD[16,4,3,15]*Left[5,6,-3]*Right[7,5,-6]*Left[2,1,-1]*Right[1,4,-4]*CenterLeft[17,18,-2]*CenterRight[16,13,-5]
    temp = reshape(temp,chi*dlink^4,chi*dlink^4)
    F = svd(temp)
    #append!(ScMatrix,[F.S])
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
    # Horizontal and Vertical Boundary Tensor
    HTL = T; VTU = T
    HTR = T; VTD = T

    # Corner Tensor
    TL,TR = svdT(T)
    TU,TD = svdT(permutedims(T,[4,1,2,3]))

    # Center Tensor
    CenterLeft = TL
    CenterRight = TR


    #-- Stores norms of tensors
    CLnorm = Array{Float64}(undef,0)
    CRnorm = Array{Float64}(undef,0)
    HTLnorm = Array{Float64}(undef,0)
    HTRnorm = Array{Float64}(undef,0)
    VTUnorm = Array{Float64}(undef,0)
    VTDnorm = Array{Float64}(undef,0)


    NumSite = Array{Int64}(undef,0)
    FEttrg = Array{Float64}(undef,0)
    FEttrg1 = Array{Float64}(undef,0)
    println("start loop")
    for j in 1:NumLayer
        println("This is NumLayer $j")

        @time begin
        #---- compute projectors
        PhL = Mover(VTU,VTD,CenterLeft,CenterRight,chimax,direction="L")
        PhR = Mover(VTU,VTD,CenterLeft,CenterRight,chimax,direction="R")
        PvU = Mover(HTL,HTR,CenterLeft,CenterRight,chimax,direction="U")
        PvD = Mover(HTL,HTR,CenterLeft,CenterRight,chimax,direction="D")

        #--- Compute tensors to normailize the center tensor
        @tensor vUp[-1,-2,-3,-4] := VTU[3,2,-3,4]*TL[5,4,-4]*TU[2,1,-2]*PvU[5,3,1,-1]
        @tensor vUp[-1,-2,-3,-4] := VTD[-3,2,3,4]*TR[1,2,-2]*TD[4,5,-4]*PvD[5,3,1,-1]
        @tensor hLeft[-1,-2,-3,-4] := HTL[3,2,5,-3]*TL[5,4,-2]*TD[1,3,-4]*PhL[1,2,4,-1]
        @tensor hRight[-1,-2,-3,-4] := HTR[2,-3,4,3]*TR[2,1,-4]*TU[5,4,-2]*PhR[1,3,5,-1]

        # Center Projector
        @time PcL,PcR = CenterProjector(T,CenterLeft,CenterRight,HTL,HTR,VTU,VTD,TL,TR,chimax)
        #@time PcL,PcR = CenterProjector1(CenterLeft,CenterRight,vUp,vUp,hLeft,hRight,chimax)

        #---- renormalize center tensor
        @tensor CenterLeft[-1,-2,-3] := vUp[-1,2,1,6]*hLeft[-2,4,3,2]*CenterLeft[1,3,5]*PcL[4,5,6,-3]
        @tensor CenterRight[-1,-2,-3] := vUp[-1,2,1,6]*hRight[-2,6,5,4]*CenterRight[1,5,3]*PcR[2,3,4,-3]

        append!(CLnorm,[maximum(CenterLeft)])
        append!(CRnorm,[maximum(CenterRight)])
        CenterLeft = CenterLeft/maximum(CenterLeft)
        CenterRight = CenterRight/maximum(CenterRight)


        #------  renormalize boundary tensor
        @tensor HTL[-1,-2,-3,-4] := HTL[5,2,3,7]*T[-1,4,5,6]*T[3,1,-3,8]*PhL[4,2,1,-2]*PhL[6,7,8,-4]
        @tensor HTR[-1,-2,-3,-4] := HTR[5,2,3,7]*T[-1,4,5,6]*T[3,1,-3,8]*PhR[4,2,1,-2]*PhR[6,7,8,-4]
        @tensor VTU[-1,-2,-3,-4] := VTU[3,2,7,4]*T[1,-2,6,2]*T[5,4,8,-4]*PvU[5,3,1,-1]*PvU[8,7,6,-3]
        @tensor VTD[-1,-2,-3,-4] := VTD[3,2,7,4]*T[1,-2,6,2]*T[5,4,8,-4]*PvD[5,3,1,-1]*PvD[8,7,6,-3]
        append!(HTLnorm,[maximum(HTL)])
        append!(HTRnorm,[maximum(HTR)])
        append!(VTUnorm,[maximum(VTU)])
        append!(VTDnorm,[maximum(VTD)])
        HTL = HTL/maximum(HTL)
        HTR = HTR/maximum(HTR)
        VTU = VTU/maximum(VTU)
        VTD = VTD/maximum(VTD)

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
        @tensor Z[] := CenterLeft[1,2,3]*CenterRight[1,2,3]
        FE = (log(Z[1])+sum(log.(CLnorm))+sum(log.(CRnorm))+sum([j-1:-1:0...].*log.(HTLnorm))+
            sum([j-1:-1:0...].*log.(HTRnorm))+sum([j-1:-1:0...].*log.(VTUnorm))+ sum([j-1:-1:0...].*log.(VTDnorm)))/Ns
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

    return CenterLeft,CenterRight,HTL,HTR,VTU,VTD,
                        CLnorm,CRnorm,HTLnorm,HTRnorm,VTUnorm,VTDnorm,FEttrg,FEttrg1,NumSite

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
    chimax = 30
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
        CenterLeft,CenterRight,HTL,HTR,VTU,VTD,CLnorm,CRnorm,HTLnorm,HTRnorm,VTUnorm,VTDnorm,
                                FEttrg,FEttrg1,numsite = CorseGraining(T,NumLayer,chimax,Sz=Tsx)




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
