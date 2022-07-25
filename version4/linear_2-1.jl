include("../../../VERSION1/2DClassical/partition.jl")
include("./all.jl")


function ConstructBaseTensorSingle(Beta) :: Tuple{Array{Float64},Array{Float64}}
    sx = Float64[0.0 1.0;
            1.0  0.0]
    sz = Float64[1.0 0.0;
            0.0  -1.0]
    #  Base Tensor is defined according to below relation

    #           √Q
    #           |
    #       √Q——g——√Q
    #           |
    #           √Q
    Q = Float64[exp(Beta) exp(-Beta);
                exp(-Beta) exp(Beta)]
    g = zeros(2,2,2,2)
    g[1,1,1,1] = g[2,2,2,2] = 1
    gp =  zeros(2,2,2,2)
    gp[1,1,1,1] = -1;gp[2,2,2,2] = 1
    @tensor T[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*g[1,2,3,4]
    @tensor Tsx[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*gp[1,2,3,4]
    return T,Tsx
end


function ConstructBaseTensorDouble(Beta::Float64)
    sx = [0.0 1.0;1.0 0.0]
    sz = [1.0 0.0; 0.0 -1.0]
    E0 = 0.0
    T = zeros(2,2,2,2)
    T = T .+ exp(-(0-E0)*Beta)
    T[1,1,1,1]=T[2,2,2,2]= exp(-(-4-E0)*Beta)
    T[1,2,1,2]=T[2,1,2,1]= exp(-(4-E0)*Beta)
    @tensor Tsx[-1,-2,-3,-4] := T[-1,-2,1,-4]*sz[1,-3]
    return T,Tsx
end




# ?
# * Important
# !
# @param myParam for this

#
FreeEnergyExact = Array{Float64}(undef,0)
FreeEnergyTree = Array{Float64}(undef,0)
FreeEnergy = Array{Array{Float64}}(undef,0)
InternalEnergy = Array{Array{Float64}}(undef,0)
InternalEnergyExact = Array{Float64}(undef,0)
NumSite = Array{Float64}(undef,0)

ParameterTest = Array{Parameter1}(undef,0)
IEtest = []
FEtest = []
PcLtest = []
PcRtest = []


#chimax = 128

#for temperature in 2.28:0.01:2.28
for chimax in [8,16,32,64,96,128,160,192,256,320,384,448]#10:10:30
        println("   chimax  $chimax   ")
        jldopen("./calculation_$chimax.jld", "w") do file
        g = g_create(file,"Parameter")
        TemperatureExact = 2/log(1+sqrt(2))
        #temperature = 0.9994TemperatureExact
        temperature = TemperatureExact
        parameter = Parameter1(
                chimax,
                1/temperature,
                63)
        feexact = ComputeFreeEnergy(parameter.Beta)
        append!(FreeEnergyExact,[feexact])
        #internalenergyexact = ComputeInternalEnergy(1/temperature)
        #append!(InternalEnergyExact,[internalenergyexact])
        @time NewRG(parameter,method="NewRG")

        #Ns = 1+8*sum(1:1:parameter.step)
        #
        NumLayer = parameter.step
        RGTensor = parameter.RGTensor;
        RGnorm = parameter.RGnorm
        #
        Ns = 2*(2*NumLayer+2)
        #FE = FreeEnergyDensityCenterSingle(parameter.RGTensor,parameter.RGnorm,NumLayer)
        FE = FreeEnergyDensityCenterDouble(parameter.RGTensor,parameter.RGnorm,NumLayer)
        append!(FreeEnergyTree,[FE*TemperatureExact])
        append!(NumSite,[Ns])
        #
        #=
        Ns = 2*(2*(parameter.step+1))^2
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
        FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
                sum([NumLayer-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
                sum([NumLayer:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([NumLayer-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([NumLayer:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
        append!(FreeEnergyTree,[FE*TemperatureExact])
        =#
        #global FreeEnergy=parameter.FEttrg
        #global InternalEnergy=parameter.IEttrg
        #append!(ParameterTest,[parameter])
        #test = (-2*parameter.IEttrg.-internalenergyexact)./internalenergyexact
        #append!(IEtest,[abs.(test)])
        #test = (parameter.FEttrg.-feexact)./feexact
        #append!(FEtest,abs.(test))
        g["parameter_single$temperature"] = parameter
        #h5write("./calculation.h5","parameter"*"$temperature",parameter)
end
end

#=
#-----------------------  Below is calculation of internal energy vs sites
#chimax = 70
TemperatureExact = 2/log(1+sqrt(2))
#temperature = 0.9994TemperatureExact
temperature = TemperatureExact
T,Tsx = ConstructBaseTensorDouble(1/temperature)
InternalEnergyExact = []
#internalenergyexact = ComputeInternalEnergy(1/temperature)
#append!(InternalEnergyExact,internalenergyexact)
parameter = load("./calculation_$chimax.jld","Parameter/parameter_single$temperature")
RGTensor = parameter.RGTensor
sx = [0.0 1; 1 0]
@tensor Hamiltonian[-1,-2,-3,-4] := sx[-1,-3]*sx[-2,-4]

InternalEnergyTree1 = []
InternalEnergyTree2 = []
InternalEnergyTree3 = []
InternalEnergyTree4 = []
InternalEnergyMPS = []

=#




#=
PvMatrix = parameter.Isometryv
eigvalup,eigvecsup = eigsolve(y->ApplyT(RGTensor[4],y),rand(size(RGTensor[4],1)),10)
eigvaldn,eigvecsdn = eigsolve(y->ApplyT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(size(RGTensor[4],1)),10)
eigenup1 = real(eigvecsup[1])
eigendn1 = real(eigvecsdn[1])
eigenup2 = real(eigvecsup[2])
eigendn2 = real(eigvecsdn[2])
#rhoUp11v = ConstructRhoUp(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn1)
#rhoDn11v = ConstructRhoDn(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn1)
#------ use eigenup & eigendn have equivalent effect as eigenup &eigenup
#rhoUp12v = ConstructRhoUp(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn1,direction="VERTICAL")
#rhoDn12v = ConstructRhoDn(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn1,direction="VERTICAL")


#
PhMatrix = parameter.Isometryh
eigvalup,eigvecsup = eigsolve(y->ApplyT(permutedims(RGTensor[5],[4,1,2,3]),y),rand(size(RGTensor[5],2)))
eigvaldn,eigvecsdn = eigsolve(y->ApplyT(permutedims(RGTensor[5],[2,3,4,1]),y),rand(size(RGTensor[5],2)))
eigenup1 = real(eigvecsup[1])
eigendn1 = real(eigvecsdn[1])
eigenup2 = real(eigvecsup[2])
eigendn2 = real(eigvecsdn[2])
rhoUp11h = ConstructRhoUp(PhMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn1,direction="VERTICAL")
rhoDn11h = ConstructRhoDn(PhMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn1,direction="VERTICAL")
rhoUp12h = ConstructRhoUp(PhMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn2,direction="VERTICAL")
rhoDn12h = ConstructRhoDn(PhMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn2,direction="VERTICAL")
#
sz = [1.0 0.0; 0.0 -1.0]
TDnMatrix = []
#
for j in 2:253
    println("This is loop $j")
    #append!(InternalEnergyTree,[ComputeIE(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup,eigendn,j)])
    #append!(InternalEnergyTree1,[ComputeInternalEnergyTree(rhoUp11v,rhoDn11v,PvMatrix,T,Tsx,j)])
    #append!(InternalEnergyTree2,[ComputeInternalEnergyTree(rhoUp12v,rhoDn12v,PvMatrix,T,Tsx,j,direction="VERTICAL")])
    #append!(InternalEnergyTree3,[ComputeInternalEnergyTreeDouble(PvMatrix,eigenup1,eigendn1,j,sz)])
    #append!(InternalEnergyTree2,[ComputeInternalEnergyTree(rhoUp12v,rhoDn12v,PvMatrix,T,Tsx,j)])
    #append!(InternalEnergyTree3,[ComputeInternalEnergyTree(rhoUp12h,rhoDn12h,PhMatrix,T,Tsx,j,direction="VERTICAL")])
    append!(InternalEnergyMPS,[ComputeInternalEnergyMPS(PvMatrix,eigenup1,eigendn1,j,sz)])
end
=#

println("This is test")
