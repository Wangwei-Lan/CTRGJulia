include("../../2DClassical/partition.jl")
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

function HMover(T::Array{Float64},A::Array{Float64},chimax;Asz=nothing,Tsz=nothing)
    println("HMover")
    #------  construct Projector based on higher order svd (more details should refer to arxiv.1201.1144)
    @tensor TEMP[-1,-2,-3,-4,-5,-6] := T[-1,-3,-4,1]*A[-2,1,-5,-6]
    chiAup = size(A,1); chiAright = size(A,4)
    chiTup = size(T,1); chiTleft = size(T,2)
    TEMP = reshape(TEMP,chiAup*chiTup,chiAup*chiTup*chiTleft*chiAright)
    TEMP = TEMP*TEMP'
    eigvectors,eigvalues = eigenorder(TEMP)
    ProjectorUp = eigvectors
    #----- Projector
    if size(ProjectorUp,2) > chimax
        ProjectorUp = reshape(ProjectorUp[:,1:chimax],chiTup,chiAup,chimax)
    else
        ProjectorUp = reshape(ProjectorUp,chiTup,chiAup,chiTup*chiAup)
    end
    #----- Update A tensor
    println("type of Asz $(typeof(Asz))")
    println("type of Tsz $(typeof(Tsz))")

    if Asz == nothing || Tsz == nothing
        @tensor A[-1,-2,-3,-4] := T[2,-2,4,3]*A[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
        println("HMover finished")
        return A,nothing,ProjectorUp
    else
        # update A, which is Denominator , should rename to denominator
        #if Tsz == nothing
        #    @tensor Asz[-1,-2,-3,-4] := T[2,-2,4,3]*Asz[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
        #    @tensor A[-1,-2,-3,-4] := T[2,-2,4,3]*A[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
        #else
            @tensor Asz[-1,-2,-3,-4] := Tsz[2,-2,4,3]*Asz[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
            @tensor A[-1,-2,-3,-4] := T[2,-2,4,3]*A[1,3,5,-4]*ProjectorUp[2,1,-1]*ProjectorUp[4,5,-3]
        #end
        println("Hmover finished")
        return A,Asz,ProjectorUp
    end
end

function VMover(T::Array{Float64},A::Array{Float64},chimax::Int64;Asz = nothing,Tsz=nothing)
    A = permutedims(A,[2,3,4,1])
    T = permutedims(T,[2,3,4,1])
    if Asz == nothing || Tsz == nothing
        A,asz,PV = HMover(T,A,chimax)
        return permutedims(A,[4,1,2,3]),nothing,PV
    else
        Asz = permutedims(Asz,[2,3,4,1])
        Tsz != nothing ? Tsz = permutedims(Tsz,[2,3,4,1]) : ()
        A,Asz,PV = HMover(T,A,chimax,Asz=Asz,Tsz=Tsz)
        return permutedims(A,[4,1,2,3]),permutedims(Asz,[4,1,2,3]),PV
    end
end


function CorseGraining(T::Array{Float64},NumLayer::Int64;Sz=Float64[1.0 0.0; 0.0 -1.0],Tsx=nothing,layer=-1)
    HT = T; VT = T;VTsz = nothing;HTsz = nothing
    CenterDenominator = T
    CenterNumerator = nothing
    #@tensor Tsx[-1,-2,-3,-4] := T[-1,-2,1,-4]*Sz[1,-3]  # Used to compute Expectation value!
    Znorm = Array{Float64}(undef,0)
    HTnorm = Array{Float64}(undef,0)
    VTnorm = Array{Float64}(undef,0)
    EnergyLayer = Array{Float64}(undef,0)

    for j in 1:NumLayer
        println("This is loop $j")
        j%1000 == 0 ? println("This is Loop $j") : ()

        htnorm = maximum(HT);append!(HTnorm,[htnorm])
        HT = HT/maximum(HT)
        #------ Left Mover
        println("-------------------------------------left--------------------------------------------")
        CenterDenominator,CenterNumerator,PL = HMover(HT,CenterDenominator,chimax,Tsz=HT,Asz=CenterNumerator)
        #------ Right Mover
        println("-------------------------------------right--------------------------------------------")

        CenterDenominator,CenterNumerator,PR = HMover(CenterDenominator,HT,chimax,Tsz=CenterNumerator,Asz=HT)


        j == layer ? (VTsz = VT;HTsz = HT) : ()

        @tensor VT[-1,-2,-3,-4] := VT[1,3,5,-4]*T[2,-2,4,3]*PL[2,1,-1]*PL[4,5,-3]
        @tensor VT[-1,-2,-3,-4] := VT[2,-2,4,3]*T[1,3,5,-4]*PR[2,1,-1]*PR[4,5,-3]

        if j == layer || j == layer+1
            @tensor VTsz[-1,-2,-3,-4] := VTsz[1,3,5,-4]*Tsx[2,-2,4,3]*PL[2,1,-1]*PL[4,5,-3]
            @tensor VTsz[-1,-2,-3,-4] := VTsz[2,-2,4,3]*T[1,3,5,-4]*PR[2,1,-1]*PR[4,5,-3]
            #VTsz = VTsz
            vtnorm = 1.0 ;append!(VTnorm,[vtnorm])
        else
            vtnorm = maximum(VT);append!(VTnorm,[vtnorm])
            VT = VT/maximum(VT)
        end

        #------ Up Mover
        println("typeof VTsz $(typeof(VTsz))")
        j == layer+1 ? CenterNumerator = CenterDenominator : ()
        println("-------------------------------------up--------------------------------------------")

        if j == layer+1
            CenterDenominator,CenterNumerator,PD = VMover(VT,CenterDenominator,chimax,Tsz=VTsz,Asz=CenterNumerator)
        else
            CenterDenominator,CenterNumerator,PD = VMover(VT,CenterDenominator,chimax,Tsz=VT,Asz=CenterNumerator)
        end
        #------ Down Mover
        println("-------------------------------------down--------------------------------------------")

        CenterDenominator,CenterNumerator,PU = VMover(CenterDenominator,VT,chimax,Tsz=CenterNumerator,Asz=VT)

        @tensor HT[-1,-2,-3,-4] := HT[-1,1,3,4]*T[3,2,-3,5]*PD[2,1,-2]*PD[5,4,-4]
        @tensor HT[-1,-2,-3,-4] := HT[3,2,-3,5]*T[-1,1,3,4]*PU[2,1,-2]*PU[5,4,-4]

        append!(Znorm,[maximum(CenterDenominator)*htnorm^2*vtnorm^2])
        CenterNumerator== nothing ?  () : (CenterNumerator=CenterNumerator/maximum(CenterDenominator))
        CenterDenominator = CenterDenominator/maximum(CenterDenominator)
    end


    return CenterDenominator,CenterNumerator,HT,VT,Znorm,HTnorm,VTnorm,EnergyLayer
end
#-----  Set Parameter
Dlink = 2
chimax = 10
sx = Float64[0.0 1.0;
            1.0  0.0]
sz = Float64[1.0 0.0;
            0.0  -1.0]


#---- Base Tensor


#T= 1.0;Beta =  1/T #log(1+sqrt(2))/2
MagExact = Array{Float64}(undef,0)
MagTree = Array{Float64}(undef,0)
MagTest = Array{Float64}(undef,0)
InternalEnergyTree = Array{Float64}(undef,0)
InternalEnergyExact = Array{Float64}(undef,0)
FreeEnergyExact = Array{Float64}(undef,0)
FreeEnergyTree = Array{Float64}(undef,0)
EnergyTreeLayer = []
for Beta in 0.3:0.02:0.3

    #Temperature = 1.0
    #Beta = 1/Temperature
    println("This is Beta $Beta")
    #=
    E0=0.0;
    T= zeros(Dlink,Dlink,Dlink,Dlink)
    T = T .+ exp(-(0-E0)*Beta)
    T[1,1,1,1]=T[2,2,2,2]= exp(-(4-E0)*Beta)
    T[1,2,1,2]=T[2,1,2,1]= exp(-(-4-E0)*Beta)
    @tensor Tsx[-1,-2,-3,-4] := T[-1,-2,1,-4]*sx[1,-3]
    @tensor Tsz[-1,-2,-3,-4] := T[-1,-2,1,-4]*sz[1,-3]
    =#

    #----- Base Tensor Test from 2D
    #
    Q = Float64[exp(Beta) exp(-Beta);
                exp(-Beta) exp(Beta)]
    g = zeros(Dlink,Dlink,Dlink,Dlink)
    g[1,1,1,1] = g[2,2,2,2] = 1
    gp = zeros(Dlink,Dlink,Dlink,Dlink)
    gp[1,1,1,1] = -1;gp[2,2,2,2] = 1
    @tensor T[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*g[1,2,3,4]
    @tensor Tsx[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*gp[1,2,3,4]
    for l in 1:1:20
        NumLayer = 100
        Ns = 1+8*sum(1:1:NumLayer)
        rhoDenominator,rhoNumerator,HT,VT,Znorm,HTnorm,VTnorm,EnergyLayer = CorseGraining(T,NumLayer,Sz=sx,Tsx=Tsx,layer=l)
        #println(EnergyLayer)
        append!(EnergyTreeLayer,[EnergyLayer])
        @tensor EnergyNumerator[] := T[17,3,4,1]*T[4,9,10,5]*T[24,2,8,3]*T[8,7,14,9]*T[16,23,17,18]*T[22,21,24,23]*
                            VT[20,1,6,2]*VT[6,5,12,7]*VT[19,18,20,21]*HT[10,15,16,11]*HT[14,13,22,15]*rhoNumerator[12,11,19,13]
        @tensor EnergyDenominator[] := T[17,3,4,1]*T[4,9,10,5]*T[24,2,8,3]*T[8,7,14,9]*T[16,23,17,18]*T[22,21,24,23]*
                            VT[20,1,6,2]*VT[6,5,12,7]*VT[19,18,20,21]*HT[10,15,16,11]*HT[14,13,22,15]*rhoDenominator[12,11,19,13]

        append!(InternalEnergyTree,[EnergyNumerator[1]/EnergyDenominator[1]])
        #Z = tr(reshape(rhoDenominator,size(rhoDenominator,1)^2,size(rhoDenominator,1)^2))
        #FE = (log(Z)+sum(log.(Znorm))+2*sum([1:1:NumLayer...].*log.(HTnorm))+
        #                2*sum([1:1:NumLayer...].*log.(VTnorm)))/Ns
        #feexact = ComputeFreeEnergy(Beta)
        #append!(FreeEnergyExact,[feexact])
        #append!(FreeEnergyTree,[FE])
    end
    #exact = (1-sinh(2*Beta)^(-4))^(1/8)
    #append!(MagExact,[exact])
    #=
    @tensor Numerator[] := Tsx[3,11,1,4]*T[17,10,13,11]*T[2,18,3,7]*T[16,15,17,18]*
                                rhoDenominator[6,5,8,12]*VT[9,4,6,10]*VT[8,7,9,15]*HT[1,14,2,5]*HT[13,12,16,14]
    @tensor Denominator[] := T[3,11,1,4]*T[17,10,13,11]*T[2,18,3,7]*T[16,15,17,18]*
                                rhoDenominator[6,5,8,12]*VT[9,4,6,10]*VT[8,7,9,15]*HT[1,14,2,5]*HT[13,12,16,14]
    append!(MagTree,[Numerator[1]/Denominator[1]])

    @tensor EnergyNumerator[] := T[17,3,4,1]*T[4,9,10,5]*T[24,2,8,3]*T[8,7,14,9]*T[16,23,17,18]*T[22,21,24,23]*
                        VT[20,1,6,2]*VT[6,5,12,7]*VT[19,18,20,21]*HT[10,15,16,11]*HT[14,13,22,15]*rhoNumerator[12,11,19,13]
    @tensor EnergyDenominator[] := T[17,3,4,1]*T[4,9,10,5]*T[24,2,8,3]*T[8,7,14,9]*T[16,23,17,18]*T[22,21,24,23]*
                        VT[20,1,6,2]*VT[6,5,12,7]*VT[19,18,20,21]*HT[10,15,16,11]*HT[14,13,22,15]*rhoDenominator[12,11,19,13]

    append!(InternalEnergyTree,[EnergyNumerator[1]/EnergyDenominator[1]])

    @tensor EnergyNumerator[] := Tsx[17,3,4,1]*Tsx[4,9,10,5]*T[24,2,8,3]*T[8,7,14,9]*T[16,23,17,18]*T[22,21,24,23]*
                        VT[20,1,6,2]*VT[6,5,12,7]*VT[19,18,20,21]*HT[10,15,16,11]*HT[14,13,22,15]*rhoDenominator[12,11,19,13]
    @tensor EnergyDenominator[] := T[17,3,4,1]*T[4,9,10,5]*T[24,2,8,3]*T[8,7,14,9]*T[16,23,17,18]*T[22,21,24,23]*
                        VT[20,1,6,2]*VT[6,5,12,7]*VT[19,18,20,21]*HT[10,15,16,11]*HT[14,13,22,15]*rhoDenominator[12,11,19,13]

    append!(InternalEnergyTree,[EnergyNumerator[1]/EnergyDenominator[1]])
    =#
    #
    internalenergyexact = ComputeInternalEnergy(Beta)
    append!(InternalEnergyExact,[internalenergyexact])



end
