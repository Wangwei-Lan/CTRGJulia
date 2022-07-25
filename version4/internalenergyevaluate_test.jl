include("../../2DClassical/partition.jl")
using TensorOperations
using JLD
using QuadGK

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
        RGTensor = Array{Array{Float64}}(undef,6)
        RGTensor[1] = t
        RGTensor[2] = ttl;RGTensor[3] = ttr
        RGTensor[4] = ttv
        RGTensor[5] = tth
        RGTensor[6] = ttvsx

        # * RGnorm stors the norms of each steps
        RGnorm = [Array{Float64}(undef,0) for j in 1:6]
        FEttrg = Array{Float64}(undef,0)
        IEttrg = Array{Float64}(undef,0)
        new(chi,Beta,step,t,tl,tr,RGTensor,RGnorm,FEttrg)
    end
end


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


chimax = 10
for temperature in 2.24:0.01:2.32
    parameter = load("./Calculation1/calculation_$chimax.jld","Parameter/parameter$temperature")

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
