include("../../2DClassical/partition.jl")
include("./all.jl")



# TODO :: Mixed Gauge or Mixed Isometry for Center Projector
# TODO :: MPS, try to make it better for periodic and non periodic cases!!!

function ConstructBaseTensor(Beta) :: Tuple{Array{Float64},Array{Float64}}
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
    #@tensor Tsx[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[5,-3]*sqrt(Q)[-4,4]*g[1,2,3,4]*sz[3,5]
    return T,Tsx
end


function ConstructBaseTensor1(Beta::Float64)
    sx = [0.0 1.0;1.0 0.0]
    sz = [1.0 0.0; 0.0 -1.0]
    E0 = 0.0
    T = zeros(2,2,2,2)
    T = T .+ exp(-(0-E0)*Beta)
    T[1,1,1,1]=T[2,2,2,2]= exp(-(4-E0)*Beta)
    T[1,2,1,2]=T[2,1,2,1]= exp(-(-4-E0)*Beta)
    @tensor Tsx[-1,-2,-3,-4] := T[-1,-2,1,-4]*sz[1,-3]

    return T,Tsx
end




# ?
# * Important
# !
# @param myParam for this

mutable struct Parameter
    chi::Int64                           # largest bond dimension
    Beta::Float64                        # Beta parameter
    step::Int64                          # RG steps
    T::Array{Float64}                    # Base tensors
    TL::Array{Float64}
    TR::Array{Float64}
    RGTensor::Array{Array{Float64}}
    RGnorm::Array{Array{Float64}}
    FEttrg::Array{Float64}
    IEttrg::Array{Float64}
    Isometry::Array{Array{Float64}}
    function  Parameter(chi::Int64,Beta::Float64,step::Int64)
        # Construct Base Tensor According to Beta value
        t,tsx = ConstructBaseTensor(Beta)
        tl,tr = svdT(t)
        @tensor tt[-1,-2,-3,-4,-5,-6,-7,-8] := t[-1,-3,2,1]*t[-2,1,4,-7]*t[2,-4,-5,3]*t[4,3,-6,-8]
        tt = reshape(tt,4,4,4,4)
        @tensor tth[-1,-2,-3,-4,-5,-6] := t[-1,-2,1,-5]*t[1,-3,-4,-6]
        @tensor ttv[-1,-2,-3,-4,-5,-6] := t[-1,-3,-4,1]*t[-2,1,-5,-6]
        @tensor ttvsx[-1,-2,-3,-4,-5,-6] := tsx[-1,-3,-4,1]*tsx[-2,1,-5,-6]
        tth = reshape(tth,2,4,2,4)
        ttv = reshape(ttv,4,2,4,2)
        ttvsx = reshape(ttvsx,4,2,4,2)


        ttl,ttr = svdT(tt)

        # Initial RGTensor;
        # * RGTensor[1] = T
        # * RGTensor[2] = cl; RGTensor[3] = cr
        # * RGTensor[4] = vertical tensor; RGTensor[5] = horizontal tensor
        # * RGTensor[5] = Tsx
        #=
        RGTensor = Array{Array{Float64}}(undef,6)
        RGTensor[1] = t
        RGTensor[2] = tl;RGTensor[3] = tr
        RGTensor[4] = RGTensor[5] = t
        RGTensor[6] = tsx
        =#
        #
        RGTensor = Array{Array{Float64}}(undef,6)
        RGTensor[1] = t
        RGTensor[2] = ttl;RGTensor[3] = ttr
        RGTensor[4] = ttv
        RGTensor[5] = tth
        RGTensor[6] = ttvsx
        #
        # * RGnorm stors the norms of each steps
        RGnorm = [Array{Float64}(undef,0) for j in 1:6]
        FEttrg = Array{Float64}(undef,0)
        IEttrg = Array{Float64}(undef,0)
        Isometryv = Array{Array{Float64}}(undef,0)
        new(chi,Beta,step,t,tl,tr,RGTensor,RGnorm,FEttrg,IEttrg,Isometryv)
    end

end




function LinearRG(Param::Parameter)
    T,Tsx = ConstructBaseTensor(Param.Beta)
    FEttrg = Array{Float64}(undef,0)
    for j in 1:Param.step
        println("------------------Step $j---------------------")
        @time begin
        #TODO  first just use RGTensor = Param.RGTensor if it's not working , then we need to modify
        #RGTensor = Param.RGTensor; RGnorm = Param.RGnorm
        #TL = Param.TL
        #TR = Param.TR
        #T = RGTensor[1]
        #Tsx = RGTensor[6]
        PcL,PcR,Ph,Pv = Mover(Param.RGTensor,Param.TL,Param.TR,Param.chi,direction="LD")
        append!(Param.Isometry,[Pv])
        @tensor Param.RGTensor[2][-1,-2,-3] := Param.RGTensor[2][4,6,8]*PcL[7,8,-3]*Ph[1,2,-2]*Pv[4,5,-1]*
                                        Param.RGTensor[5][5,1,3,6]*Param.TL[3,2,7]
        @tensor Param.RGTensor[3][-1,-2,-3] := Param.RGTensor[3][5,4,8]*PcR[7,8,-3]*Ph[4,6,-2]*Pv[1,2,-1]*
                                        Param.RGTensor[4][5,3,1,6]*Param.TR[2,3,7]
        @tensor Param.RGTensor[5][-1,-2,-3,-4] := Param.RGTensor[5][-1,4,1,2]*Param.RGTensor[1][1,5,-3,3]*Ph[4,5,-2]*Ph[2,3,-4]
        @tensor Param.RGTensor[4][-1,-2,-3,-4] := Param.RGTensor[4][3,1,5,-4]*Param.RGTensor[1][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        #if j == 1
        #    @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][3,1,5,-4]*Param.RGTensor[6][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        #else
        @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][3,1,5,-4]*Param.RGTensor[1][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        #end
        NormalizeTensor(Param.RGTensor,Param.RGnorm)

        PcL,PcR,Ph,Pv = Mover(Param.RGTensor,Param.TL,Param.TR,Param.chi,direction="RU")
        append!(Param.Isometry,[Pv])
        @tensor Param.RGTensor[2][-1,-2,-3] := Param.RGTensor[2][6,4,7]*Param.TL[3,2,8]*Param.RGTensor[4][1,5,6,2]*
                                    Ph[4,5,-2]*Pv[1,3,-1]*PcL[8,7,-3]
        @tensor Param.RGTensor[3][-1,-2,-3] := Param.RGTensor[3][4,5,7]*Param.TR[2,3,8]*Param.RGTensor[5][2,5,6,1]*
                                    Ph[1,3,-2]*Pv[4,6,-1]*PcR[8,7,-3]
        @tensor Param.RGTensor[5][-1,-2,-3,-4] := Param.RGTensor[5][1,5,-3,3]*Param.RGTensor[1][-1,4,1,2]*Ph[5,4,-2]*Ph[3,2,-4]
        @tensor Param.RGTensor[4][-1,-2,-3,-4] := Param.RGTensor[4][2,-2,4,1]*Param.RGTensor[1][3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][2,-2,4,1]*Param.RGTensor[1][3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        NormalizeTensor(Param.RGTensor,Param.RGnorm)
        end
        #
        #Ns = (2*(j+1))^2
        #=
        Ns = 1+8*sum(1:1:j)
        Ns = (2*(j+1))^2
        @tensor Z[] := Param.RGTensor[2][1,2,3]*Param.RGTensor[3][1,2,3]
        println(Z[1])
        FE = (log(Z[1]) + sum(log.(Param.RGnorm[2]))+ sum(log.(Param.RGnorm[3]))+
                sum([j-1:-1:0...].*log.(Param.RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(Param.RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(Param.RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[4][2:2:end]))+
                sum([j:-1:1...].*log.(Param.RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[5][2:2:end])))/Ns
        append!(Param.FEttrg,[FE])
        =#
        #=
        @tensor Numerator[] := Tsx[17,3,5,1]*T[19,2,7,3]*Tsx[15,11,17,9]*T[16,10,19,11]*Param.RGTensor[4][18,1,4,2]*
                        Param.RGTensor[4][12,9,18,10]*Param.RGTensor[5][5,8,15,6]*Param.RGTensor[5][7,14,16,8]*
                        Param.RGTensor[2][4,6,13]*Param.RGTensor[3][12,14,13]
        @tensor Denominator[] := T[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*Param.RGTensor[4][18,1,4,2]*
                        Param.RGTensor[4][12,9,18,10]*Param.RGTensor[5][5,8,15,6]*Param.RGTensor[5][7,14,16,8]*
                        Param.RGTensor[2][4,6,13]*Param.RGTensor[3][12,14,13]
        internalenergytree = Numerator[1]/Denominator[1]
        append!(Param.IEttrg,[internalenergytree])
        =#
        #=
        @tensor Numerator[] := Tsx[18,3,4,1]*Tsx[4,6,10,7]*T[21,2,5,3]*T[5,9,13,6]*T[17,20,18,23]*T[19,24,21,20]*
                        Param.RGTensor[4][25,1,8,2]*Param.RGTensor[4][8,7,12,9]*Param.RGTensor[4][22,23,25,24]*
                        Param.RGTensor[5][10,14,17,11]*Param.RGTensor[5][13,16,19,14]*Param.RGTensor[2][12,11,15]*
                        Param.RGTensor[3][22,16,15]
        @tensor Denominator[] := T[18,3,4,1]*T[4,6,10,7]*T[21,2,5,3]*T[5,9,13,6]*T[17,20,18,23]*T[19,24,21,20]*
                        Param.RGTensor[4][25,1,8,2]*Param.RGTensor[4][8,7,12,9]*Param.RGTensor[4][22,23,25,24]*
                        Param.RGTensor[5][10,14,17,11]*Param.RGTensor[5][13,16,19,14]*Param.RGTensor[2][12,11,15]*
                        Param.RGTensor[3][22,16,15]
        internalenergytree = Numerator[1]/Denominator[1]
        append!(Param.IEttrg,[internalenergytree])
        =#
        #=
        @tensor Numerator[] := Tsx[24,3,4,1]*Tsx[4,6,10,7]*T[27,2,5,3]*T[17,19,23,20]*T[23,26,24,29]*T[18,22,25,19]*T[25,31,27,26]*
                                T[5,9,13,6]*Param.RGTensor[4][30,1,8,2]*Param.RGTensor[4][8,7,12,9]*Param.RGTensor[4][21,20,28,22]*
                                Param.RGTensor[4][28,29,30,31]*Param.RGTensor[5][10,14,17,11]*Param.RGTensor[5][13,16,18,14]*
                                Param.RGTensor[2][12,11,15]*Param.RGTensor[3][21,16,15]
        @tensor Denominator[] := T[24,3,4,1]*T[4,6,10,7]*T[27,2,5,3]*T[17,19,23,20]*T[23,26,24,29]*T[18,22,25,19]*T[25,31,27,26]*
                                T[5,9,13,6]*Param.RGTensor[4][30,1,8,2]*Param.RGTensor[4][8,7,12,9]*Param.RGTensor[4][21,20,28,22]*
                                Param.RGTensor[4][28,29,30,31]*Param.RGTensor[5][10,14,17,11]*Param.RGTensor[5][13,16,18,14]*
                                Param.RGTensor[2][12,11,15]*Param.RGTensor[3][21,16,15]
        internalenergytree = Numerator[1]/Denominator[1]
        append!(Param.IEttrg,[internalenergytree])
        =#
        #append!(NumSite,[Ns])
        #=
        Ns = 2*(2*j+1)
        Lambda = FreeEnergyDensity(RGTensor[4],RGTensor[4])
        FE = (log(real(Lambda))+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*j+1))
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        =#
    end
end


#
FreeEnergyExact = Array{Float64}(undef,0)
FreeEnergyTree = Array{Float64}(undef,0)
FreeEnergy = Array{Array{Float64}}(undef,0)
InternalEnergy = Array{Array{Float64}}(undef,0)
InternalEnergyExact = Array{Float64}(undef,0)
NumSite = Array{Float64}(undef,0)
PvMatrix = Array{Array{Float64}}(undef,0)

ParameterTest = Array{Parameter}(undef,0)
IEtest = []
FEtest = []
PcLtest = []
PcRtest = []


chimax = 10
jldopen("./calculation_$chimax.jld", "w") do file
g = g_create(file,"Parameter")
#for temperature in 2.28:0.01:2.28
#for chimax in 10:10:120
    TemperatureExact = 2/log(1+sqrt(2))
    #temperature = 0.9994TemperatureExact
    temperature = TemperatureExact
    parameter = Parameter(
            chimax,
            1/temperature,
            255)
    feexact = ComputeFreeEnergy(parameter.Beta)
    append!(FreeEnergyExact,[feexact])
    #internalenergyexact = ComputeInternalEnergy(1/temperature)
    #append!(InternalEnergyExact,[internalenergyexact])
    @time LinearRG(parameter)

    #Ns = 1+8*sum(1:1:parameter.step)
    #
    NumLayer = parameter.step
    RGTensor = parameter.RGTensor;
    RGnorm = parameter.RGnorm
    #=
    Ns = 2*(2*NumLayer+2)
    FE = FreeEnergyDensitySingle(parameter.RGTensor,parameter.RGnorm,NumLayer)
    append!(FreeEnergyTree,[FE])
    append!(NumSite,[Ns])
    =#
    #=
    Ns = (2*(parameter.step+1))^2
    @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
    FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
            sum([ NumLayer-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
    append!(FreeEnergyTree,[FE])
    #
    append!(FreeEnergy,[parameter.FEttrg])
    append!(InternalEnergy,[parameter.IEttrg])
    append!(ParameterTest,[parameter])
    #test = (-2*parameter.IEttrg.-internalenergyexact)./internalenergyexact
    #append!(IEtest,[abs.(test)])
    test = (parameter.FEttrg.-feexact)./feexact
    append!(FEtest,[abs.(test)])
    =#

    g["parameter_single$temperature"] = parameter
    #h5write("./calculation.h5","parameter"*"$temperature",parameter)
end

#chi = 200
TemperatureExact = 2/log(1+sqrt(2))
temperature = TemperatureExact
T,Tsx = ConstructBaseTensor1(1/temperature)
InternalEnergyExact = []
parameter = load("./calculation_$chimax.jld","Parameter/parameter_single$temperature")
freeenergy = FreeEnergyDensitySingle(parameter.RGTensor,parameter.RGnorm,255)

accuracy = (freeenergy- FreeEnergy200_512)/FreeEnergy200_512




#save("./freeenergy_$chimax.jld","FreeEnergyExact",FreeEnergyExact)
#save("./freeenergy_$chimax.jld","FreeEnergyTree",FreeEnergyTree)
#end



#--------------------------------- Compute Internal Energy for different temperature

#=
chimax = 40
InternalEnergyTree = Array{Float64}(undef,0)
InternalEnergyExact = Array{Float64}(undef,0)
for temperature in 2.28:0.01:2.28
    parameter = load("./calculation_$chimax.jld","Parameter/parameter_single$temperature")

    T,Tsx = ConstructBaseTensor(1/temperature)
    RGTensor = parameter.RGTensor
    @tensor Numerator[] := Tsx[17,3,5,1]*T[19,2,7,3]*Tsx[15,11,17,9]*T[16,10,19,11]*RGTensor[4][18,1,4,2]*
                    RGTensor[4][12,9,18,10]*RGTensor[5][5,8,15,6]*RGTensor[5][7,14,16,8]*
                    RGTensor[2][4,6,13]*RGTensor[3][12,14,13]
    @tensor Denominator[] := T[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*RGTensor[4][18,1,4,2]*
                    RGTensor[4][12,9,18,10]*RGTensor[5][5,8,15,6]*RGTensor[5][7,14,16,8]*
                    RGTensor[2][4,6,13]*RGTensor[3][12,14,13]
    internalenergytree = Numerator[1]/Denominator[1]
    internalenergyexact = ComputeInternalEnergy(1/temperature)
    append!(InternalEnergyTree,[internalenergytree])
    append!(InternalEnergyExact,[internalenergyexact])
end
=#


#
#-----------------------  Below is calculation of internal energy vs sites

function ApplyT(T::Array{Float64},v::Array{Float64})
    @tensor v[-1] := T[1,2,-1,2]*v[1]
    return v
end


function ConstructRhoUp(Proj::Array{Array{Float64}},RGTensor::Array{Array{Float64}},Tsx::Array{Float64},
                         Hamiltonian::Array{Float64},eigenup::Array{Float64},eigendn::Array{Float64})
    rhoUp = Array{Array{Float64}}(undef,0)
    sitetot = size(Proj,1) + 1
    T = RGTensor[1]
    Proj = reverse(Proj)
    println(size(Proj[1]))
    @tensor Denominator[-1,-2,-3,-4] := eigenup[1]*eigendn[6]*Proj[1][2,3,1]*Proj[1][7,9,6]*
                        Proj[2][-1,4,2]*Proj[2][-3,8,7]*T[4,5,8,-2]*T[3,-4,9,5]
    Denominator = Denominator/maximum(Denominator)
    append!(rhoUp,[Denominator])
    for j in 3:size(Proj,1)-1
        if j%2 == 1
            @tensor Denominator[-1,-2,-3,-4] := Denominator[1,-2,4,2]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,-4,5,2]
        else
            @tensor Denominator[-1,-2,-3,-4] := Denominator[1,2,4,-4]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,2,5,-2]
        end
        Denominator = Denominator/maximum(Denominator)
        append!(rhoUp,[Denominator])
    end
    return rhoUp
end

function  ConstructRhoDn(PvMatrix::Array{Array{Float64}},RGTensor::Array{Array{Float64}},Tsx::Array{Float64},
                            Hamiltonian::Array{Float64},eigenup::Array{Float64},eigendn::Array{Float64})
    rhoDn = Array{Array{Float64}}(undef,0)
    sitetot = size(PvMatrix,1) + 1
    T = RGTensor[1]

    @tensor TT[-1,-2,-3,-4,-5,-6] := T[-1,-2,1,-5]*T[1,-3,-4,-6]
    TT = reshape(TT,4,2,4,2)
#    TT = T
    @tensor Denominator[-1,-2,-3,-4] := PvMatrix[1][3,1,-1]*PvMatrix[1][5,4,-3]*T[1,-2,4,2]*TT[3,2,5,-4]
    Denominator= Denominator/maximum(Denominator)
    append!(rhoDn,[Denominator])
    for j in 2:size(PvMatrix,1)-1
        if j %2 ==1
            @tensor Denominator[-1,-2,-3,-4] := Denominator[2,3,5,-4]*T[1,-2,4,3]*PvMatrix[j][2,1,-1]*PvMatrix[j][5,4,-3]
        elseif j%2 == 0
            @tensor Denominator[-1,-2,-3,-4] := Denominator[2,-2,4,3]*T[1,3,5,-4]*PvMatrix[j][2,1,-1]*PvMatrix[j][4,5,-3]
        end
        Denominator = Denominator/maximum(Denominator)
        append!(rhoDn,[Denominator])
    end
    return rhoDn
end


function ComputeInternalEnergyTree(rhoUp::Array{Array{Float64}},rhoDn::Array{Array{Float64}},PvMatrix::Array{Array{Float64}},
                            T::Array{Float64},Tsx::Array{Float64},site::Int64)
    #
    NumLayer = div(size(PvMatrix,1),2)
    PvMatrix = reverse(PvMatrix)
    if site < NumLayer -1
        @tensor Numerator[] := rhoUp[2*site-2][1,2,4,7]*PvMatrix[2*site][6,3,1]*PvMatrix[2*site][9,5,4]*PvMatrix[2*site+1][11,8,6]*
                        PvMatrix[2*site+1][15,10,9]*PvMatrix[2*site+2][16,13,11]*PvMatrix[2*site+2][18,14,15]*rhoDn[2*NumLayer-2*site-2][16,17,18,19]*
                        Tsx[3,2,5,12]*Tsx[13,12,14,17]*T[8,19,10,7]
        @tensor Denominator[] :=  rhoUp[2*site-2][1,2,4,7]*PvMatrix[2*site][6,3,1]*PvMatrix[2*site][9,5,4]*PvMatrix[2*site+1][11,8,6]*
                        PvMatrix[2*site+1][15,10,9]*PvMatrix[2*site+2][16,13,11]*PvMatrix[2*site+2][18,14,15]*rhoDn[2*NumLayer-2*site-2][16,17,18,19]*
                        T[3,2,5,12]*T[13,12,14,17]*T[8,19,10,7]
    elseif site == NumLayer-1
        @tensor TT[-1,-2,-3,-4,-5,-6] := T[-1,-2,1,-5]*T[1,-3,-4,-6]
        TT = reshape(TT,4,2,4,2)
        #TT = T
        @tensor Numerator[] := rhoUp[end-2][1,2,4,8]*Tsx[3,2,5,12]*Tsx[13,12,15,17]*TT[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
        @tensor Denominator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*T[13,12,15,17]*TT[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
    elseif site == NumLayer
        #
        @tensor TTsx[-1,-2,-3,-4,-5,-6] := Tsx[-1,-2,1,-5]*T[1,-3,-4,-6]
        TTsx = reshape(TTsx,4,2,4,2)
        @tensor TT[-1,-2,-3,-4,-5,-6] := T[-1,-2,1,-5]*T[1,-3,-4,-6]
        TT = reshape(TT,4,2,4,2)
        #
        #TTsx = Tsx;TT = T
        @tensor Numerator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*Tsx[13,12,15,17]*TTsx[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
        @tensor Denominator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*T[13,12,15,17]*TT[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
    elseif site == NumLayer+1
        #
        @tensor TTsx[-1,-2,-3,-4,-5,-6] := T[-1,-2,1,-5]*Tsx[1,-3,-4,-6]
        TTsx = reshape(TTsx,4,2,4,2)
        @tensor TT[-1,-2,-3,-4,-5,-6] := T[-1,-2,1,-5]*T[1,-3,-4,-6]
        TT = reshape(TT,4,2,4,2)
        #
        #TTsx = Tsx;TT = T
        @tensor Numerator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*T[13,12,15,17]*TTsx[16,17,18,19]*Tsx[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
        @tensor Denominator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*T[13,12,15,17]*TT[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
    elseif site > NumLayer+1
        @tensor Numerator[] := rhoUp[4*NumLayer-2*site-1][1,7,4,3]*rhoDn[2*(site-NumLayer-1)-1][16,17,18,19]*PvMatrix[4*NumLayer-2*site+1][6,2,1]*
                        PvMatrix[4*NumLayer-2*site+1][9,5,4]*PvMatrix[4*NumLayer-2*site+2][11,8,6]*PvMatrix[4*NumLayer-2*site+2][14,10,9]*
                        PvMatrix[4*NumLayer-2*site+3][16,12,11]*PvMatrix[4*NumLayer-2*site+3][18,15,14]*Tsx[12,19,15,13]*Tsx[2,13,5,3]*T[8,7,10,17]
        @tensor Denominator[] := rhoUp[4*NumLayer-2*site-1][1,7,4,3]*rhoDn[2*(site-NumLayer-1)-1][16,17,18,19]*PvMatrix[4*NumLayer-2*site+1][6,2,1]*
                        PvMatrix[4*NumLayer-2*site+1][9,5,4]*PvMatrix[4*NumLayer-2*site+2][11,8,6]*PvMatrix[4*NumLayer-2*site+2][14,10,9]*
                        PvMatrix[4*NumLayer-2*site+3][16,12,11]*PvMatrix[4*NumLayer-2*site+3][18,15,14]*T[12,19,15,13]*T[2,13,5,3]*T[8,7,10,17]

    end
    internalenergytree = Numerator[1]/Denominator[1]
    return internalenergytree
    #
end




chimax = 200
TemperatureExact = 2/log(1+sqrt(2))
#temperature = 0.9994TemperatureExact
temperature = TemperatureExact
T,Tsx = ConstructBaseTensor(1/temperature)
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

eigvalup,eigvecsup = eigsolve(y->ApplyT(RGTensor[4],y),rand(size(RGTensor[4],1)),2)
eigvaldn,eigvecsdn = eigsolve(y->ApplyT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(size(RGTensor[4],1)))
eigenup1 = real(eigvecsup[1])
eigendn1 = real(eigvecsdn[1])
eigenup2 = real(eigvecsup[2])
eigendn2 = real(eigvecsdn[2])

eigendn1 = eigenup1
eigendn2 = eigenup2

PvMatrix = parameter.Isometry
rhoUp11 = ConstructRhoUp(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn1)
rhoDn11 = ConstructRhoDn(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn1)

rhoUp12 = ConstructRhoUp(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn2)
rhoDn12 = ConstructRhoDn(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup1,eigendn2)





for j in 2:509
    println("This is loop $j")
    #append!(InternalEnergyTree,[ComputeIE(PvMatrix,RGTensor,Tsx,Hamiltonian,eigenup,eigendn,j)])
    append!(InternalEnergyTree1,[ComputeInternalEnergyTree(rhoUp11,rhoDn11,PvMatrix,T,Tsx,j)])
    append!(InternalEnergyTree2,[ComputeInternalEnergyTree(rhoUp12,rhoDn12,PvMatrix,T,Tsx,j)])
    #append!(InternalEnergyTree3,[ComputeInternalEnergyTree(rhoUp22,rhoDn22,PvMatrix,T,Tsx,j)])
    #append!(InternalEnergyTree4,[ComputeInternalEnergyTree(rhoUp21,rhoDn21,PvMatrix,T,Tsx,j)])

end
InternalEnergyExact = ComputeInternalEnergy(0.9999999*1/temperature)
#test = (2*InternalEnergyTree.-InternalEnergyExact)./InternalEnergyExact
#test = (-2*InternalEnergyTree.-InternalEnergy250)./InternalEnergy250

x = 1
#
println("This is test")
#

#=
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
for temperature in 2.25:0.01:2.31
    chimax = 42
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
    gp =  zeros(Dlink,Dlink,Dlink,Dlink)
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
    NumLayer = 600
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

        #=
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
        =#
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
    #
    Ns = 1+8*sum(1:1:NumLayer)
    @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
    FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
    append!(FreeEnergyTree,[FE])
    append!(NumSite,[Ns])
    #


end




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

=#
=#
