
mutable struct Parameter1
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
    Isometryv::Array{Array{Float64}}
    Isometryh::Array{Array{Float64}}
    function  Parameter1(chi::Int64,Beta::Float64,step::Int64)
        # Construct Base Tensor According to Beta value
        t,tsx = ConstructBaseTensorDouble(Beta)
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
        Isometryh = Array{Array{Float64}}(undef,0)
        new(chi,Beta,step,t,tl,tr,RGTensor,RGnorm,FEttrg,IEttrg,Isometryv,Isometryh)
    end

end
