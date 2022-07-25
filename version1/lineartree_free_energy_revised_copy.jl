include("../2DClassical/partition.jl")
using TensorOperations
using LinearAlgebra
using Arpack

#--- Debug Parameter


#------- function ordered eigen value decomposition
function eigenorder(A::Array{Float64})
    F = eigen(A)
    order = sortperm(F.values,rev=true,by=abs)
    return F.vectors[:,order],F.values[order]
end

function HMover(T::Array{Float64},A::Array{Float64},chimax;Asz=nothing)
    #------  construct Projector based on higher order svd (more details should refer to arxiv.1201.1144)
    ##
    chiAup = size(A,1); chiAright = size(A,4)
    chiTup = size(T,1); chiTleft = size(T,2)
    @tensor TEMP[-1,-2,-3,-4] := T[-1,1,2,5]*T[-3,1,2,6]*A[-2,5,3,4]*A[-4,6,3,4]
    #
    #
    #@tensor TEMP[-1,-2,-3,-4] :=  T[7,1,2,5]*T[-1,8,7,10]*T[9,1,2,6]*T[-3,8,9,14]*
    #                        A[11,5,3,4]*A[-2,10,11,13]*A[12,6,3,4]*A[-4,14,12,13]
    #
    #
    TEMP = reshape(TEMP,chiAup*chiTup,chiAup*chiTup)
    eigvectors,eigvalues = eigenorder((TEMP+TEMP')/2)
    ProjectorUp = eigvectors
    #F = svd((TEMP+TEMP')/2)
    #ProjectorUp = F.U
    #----- Projector
    if size(ProjectorUp,2) > chimax
        ProjectorUp = reshape(ProjectorUp[:,1:chimax],chiTup,chiAup,chimax)
        #eigtrun = norm(eigvectors)/norm(eigvectors[1:chimax])
    else
        ProjectorUp = reshape(ProjectorUp,chiTup,chiAup,chiTup*chiAup)
        #eigtrun = 1.0
    end
    #----- Update A tensor
    if Asz == nothing
        @tensor A[-1,-2,-3,-4] := T[2,-2,4,3]*A[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
        return A,ProjectorUp
    else
        #  Asz is Numerator
        if size(Asz,1) == size(ProjectorUp,1)                                # This if deal with Left and Right mover
            @tensor Asz[-1,-2,-3,-4] := Asz[2,-2,4,3]*A[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
        else
            @tensor Asz[-1,-2,-3,-4] := T[2,-2,4,3]*Asz[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
        end
        # update A, which is Denominator , should rename to denominator
        @tensor A[-1,-2,-3,-4] := T[2,-2,4,3]*A[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
        return A,Asz,ProjectorUp
    end
end

function VMover(T::Array{Float64},A::Array{Float64},chimax::Int64;Asz = nothing)
    A = permutedims(A,[2,3,4,1])
    T = permutedims(T,[2,3,4,1])
    if Asz == nothing
        A,PV = HMover(T,A,chimax)
        return permutedims(A,[4,1,2,3]),PV
    else
        Asz = permutedims(Asz,[2,3,4,1])
        A,Asz,PV = HMover(T,A,chimax,Asz=Asz)
        return permutedims(A,[4,1,2,3]),permutedims(Asz,[4,1,2,3]),PV
    end
end


function CorseGraining(T::Array{Float64},NumLayer::Int64,chimax;Sz=Float64[1.0 0.0; 0.0 -1.0])
    HT = T; VT = T
    CenterDenominator = T
    @tensor CenterNumerator[-1,-2,-3,-4] := T[-1,-2,1,-4]*Sz[1,-3]  # Used to compute Expectation value!
    Znorm = Array{Float64}(undef,0)
    HTnorm = Array{Float64}(undef,0)
    VTnorm = Array{Float64}(undef,0)
    NumSite = Array{Int64}(undef,0)
    FEttrg = Array{Float64}(undef,0)
    for j in 1:NumLayer
        @time begin
        j%1000 == 0 ? println("This is Loop $j") : ()
        htnorm = maximum(HT);append!(HTnorm,[htnorm])
        HT = HT/maximum(HT)
        #------ Left Mover
        #CenterDenominator,CenterNumerator,PL = HMover(HT,CenterDenominator,chimax,Asz=CenterNumerator)
        CenterDenominator,PL = HMover(HT,CenterDenominator,chimax)
        @tensor VT[-1,-2,-3,-4] := VT[1,3,5,-4]*T[2,-2,4,3]*PL[2,1,-1]*PL[4,5,-3]

        #------ Right Mover
        #CenterDenominator,CenterNumerator,PR = HMover(CenterDenominator,HT,chimax,Asz=CenterNumerator)
        CenterDenominator,PR = HMover(CenterDenominator,HT,chimax)

        @tensor VT[-1,-2,-3,-4] := VT[2,-2,4,3]*T[1,3,5,-4]*PR[2,1,-1]*PR[4,5,-3]

        vtnorm = maximum(VT);append!(VTnorm,[vtnorm])
        VT = VT/maximum(VT)

        #------ Up Mover
        #CenterDenominator,CenterNumerator,PD = VMover(VT,CenterDenominator,chimax,Asz=CenterNumerator)
        CenterDenominator,PD = VMover(VT,CenterDenominator,chimax)

        @tensor HT[-1,-2,-3,-4] := HT[-1,1,3,4]*T[3,2,-3,5]*PD[2,1,-2]*PD[5,4,-4]

        #------ Down Mover
        #CenterDenominator,CenterNumerator,PU = VMover(CenterDenominator,VT,chimax,Asz=CenterNumerator)
        CenterDenominator,PU = VMover(CenterDenominator,VT,chimax)

        @tensor HT[-1,-2,-3,-4] := HT[3,2,-3,5]*T[-1,1,3,4]*PU[2,1,-2]*PU[5,4,-4]
        #append!(Znorm,[maximum(CenterDenominator)*htnorm^2*vtnorm^2])
        # test
        append!(Znorm,[maximum(CenterDenominator)])

        CenterNumerator = CenterNumerator/maximum(CenterDenominator)
        CenterDenominator = CenterDenominator/maximum(CenterDenominator)

        Ns = 1+8*sum(1:1:j)
        Z = tr(reshape(CenterDenominator,size(CenterDenominator,1)^2,size(CenterDenominator,1)^2))
        @tensor Z[] := CenterDenominator[1,2,1,2]
        FE = (log(Z[1])+sum(log.(Znorm))+2*sum([j:-1:1...].*log.(HTnorm))+
                        2*sum([j:-1:1...].*log.(VTnorm)))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        end
    end

    return CenterDenominator,CenterNumerator,HT,VT,Znorm,HTnorm,VTnorm,FEttrg,NumSite

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


#for Beta in 0.3:0.05:0.6
    BetaExact =1/2*log(1+sqrt(2))
    Beta = 0.3#0.9994*BetaExact
    feexact = ComputeFreeEnergy(Beta)
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
    chimax = 40
        NumLayer = 50
        Ns = 1+8*sum(1:1:NumLayer)
        rhoDenominator,rhoNumerator,HT,VT,Znorm,HTnorm,VTnorm,FEttrg,NumSite = CorseGraining(T,NumLayer,chimax,Sz=sx)
        @tensor Z[] := rhoDenominator[1,2,1,2]
        #FE = (log(Z[1])+sum(log.(Znorm))+2*sum([1:1:NumLayer...].*log.(HTnorm))+
        #                2*sum([1:1:NumLayer...].*log.(VTnorm)))/Ns
        global FE = (log(Z[1])+sum(log.(Znorm))+2*sum([NumLayer:-1:1...].*log.(HTnorm))+
                        2*sum([NumLayer:-1:1...].*log.(VTnorm)))/Ns
        append!(FreeEnergyTree,[FE])
    #end
    #end


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
