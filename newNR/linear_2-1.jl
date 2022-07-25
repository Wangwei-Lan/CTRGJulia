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
    function  Parameter(chi::Int64,Beta::Float64,step::Int64)
        # Construct Base Tensor According to Beta value
        t,tsx = ConstructBaseTensor(Beta)
        tl,tr = svdT(t)

        # Initial RGTensor;
        # * RGTensor[1] = T
        # * RGTensor[2] = cl; RGTensor[3] = cr
        # * RGTensor[4] = vertical tensor; RGTensor[5] = horizontal tensor
        # * RGTensor[5] = Tsx
        RGTensor = Array{Array{Float64}}(undef,6)
        RGTensor[1] = t
        RGTensor[2] = tl;RGTensor[3] = tr
        RGTensor[4] = RGTensor[5] = t
        RGTensor[6] = tsx

        # * RGnorm stors the norms of each steps
        RGnorm = [Array{Float64}(undef,0) for j in 1:6]
        FEttrg = Array{Float64}(undef,0)
        new(chi,Beta,step,t,tl,tr,RGTensor,RGnorm,FEttrg)
    end

end




function LinearRG(Param::Parameter)

    FEttrg = Array{Float64}(undef,0)
    for j in 1:Param.step
        println("------------------Step $j---------------------")

        #TODO  first just use RGTensor = Param.RGTensor if it's not working , then we need to modify
        #RGTensor = Param.RGTensor; RGnorm = Param.RGnorm
        #TL = Param.TL
        #TR = Param.TR
        #T = RGTensor[1]
        #Tsx = RGTensor[6]
        PcL,PcR,Ph,Pv = Mover(Param.RGTensor,Param.TL,Param.TR,Param.chi,direction="LD")
        @tensor Param.RGTensor[2][-1,-2,-3] := Param.RGTensor[2][4,6,8]*PcL[7,8,-3]*Ph[1,2,-2]*Pv[4,5,-1]*
                                        Param.RGTensor[5][5,1,3,6]*Param.TL[3,2,7]
        @tensor Param.RGTensor[3][-1,-2,-3] := Param.RGTensor[3][5,4,8]*PcR[7,8,-3]*Ph[4,6,-2]*Pv[1,2,-1]*
                                        Param.RGTensor[4][5,3,1,6]*Param.TR[2,3,7]
        @tensor Param.RGTensor[5][-1,-2,-3,-4] := Param.RGTensor[5][-1,4,1,2]*Param.RGTensor[1][1,5,-3,3]*Ph[4,5,-2]*Ph[2,3,-4]
        @tensor Param.RGTensor[4][-1,-2,-3,-4] := Param.RGTensor[4][3,1,5,-4]*Param.RGTensor[1][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        if j == 1
            @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][3,1,5,-4]*Param.RGTensor[6][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        else
            @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][3,1,5,-4]*Param.RGTensor[1][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        end
        NormalizeTensor(Param.RGTensor,Param.RGnorm)

        PcL,PcR,Ph,Pv = Mover(Param.RGTensor,Param.TL,Param.TR,Param.chi,direction="RU")
        @tensor Param.RGTensor[2][-1,-2,-3] := Param.RGTensor[2][6,4,7]*Param.TL[3,2,8]*Param.RGTensor[4][1,5,6,2]*
                                    Ph[4,5,-2]*Pv[1,3,-1]*PcL[8,7,-3]
        @tensor Param.RGTensor[3][-1,-2,-3] := Param.RGTensor[3][4,5,7]*Param.TR[2,3,8]*Param.RGTensor[5][2,5,6,1]*
                                    Ph[1,3,-2]*Pv[4,6,-1]*PcR[8,7,-3]
        @tensor Param.RGTensor[5][-1,-2,-3,-4] := Param.RGTensor[5][1,5,-3,3]*Param.RGTensor[1][-1,4,1,2]*Ph[5,4,-2]*Ph[3,2,-4]
        @tensor Param.RGTensor[4][-1,-2,-3,-4] := Param.RGTensor[4][2,-2,4,1]*Param.RGTensor[1][3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][2,-2,4,1]*Param.RGTensor[1][3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        NormalizeTensor(Param.RGTensor,Param.RGnorm)

        #
        #Ns = (2*(j+1))^2
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := Param.RGTensor[2][1,2,3]*Param.RGTensor[3][1,2,3]
        println(Z[1])
        FE = (log(Z[1]) + sum(log.(Param.RGnorm[2]))+ sum(log.(Param.RGnorm[3]))+
                sum([j-1:-1:0...].*log.(Param.RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(Param.RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(Param.RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[4][2:2:end]))+
                sum([j:-1:1...].*log.(Param.RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[5][2:2:end])))/Ns
        append!(Param.FEttrg,[FE])
        #append!(NumSite,[Ns])
        #
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
ParameterTest = Array{Parameter}(undef,0)
#for temperature in 2.24:0.01:2.32
    temperature = 2.24
    parameter = Parameter(
            10,
            1/temperature,
            2500)
    feexact = ComputeFreeEnergy(parameter.Beta)
    append!(FreeEnergyExact,[feexact])
    LinearRG(parameter)

    Ns = 1+8*sum(1:1:parameter.step)
    NumLayer = parameter.step
    RGTensor = parameter.RGTensor;
    RGnorm = parameter.RGnorm
    @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
    FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
    append!(FreeEnergyTree,[FE])
    append!(FreeEnergy,[parameter.FEttrg])
    append!(ParameterTest,[parameter])
    #h5write("./calculation.h5","parameter"*"$temperature",parameter)
#end
