using TensorOperations


function ConstructBaseTensor(Beta)
    #----- Base Tensor Test from 2D
    Q = Float64[exp(Beta) exp(-Beta);
                exp(-Beta) exp(Beta)]
    g = zeros(2,2,2,2)
    g[1,1,1,1] = g[2,2,2,2] = 1
    gp = zeros(Dlink,Dlink,Dlink,Dlink)
    gp[1,1,1,1] = -1;gp[2,2,2,2] = 1
    @tensor T[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*g[1,2,3,4]
    return T
end




BetaExact =1/2*log(1+sqrt(2))
Beta = 0.9994*BetaExact


T = ConstructBaseTensor(Beta)


global HT = T
HTnorm = Array{Float64}(undef,0)
for j in 1:20
    chi = size(HT,1);d = size(T,1)
    global @tensor HT[-1,-2,-3,-4,-5,-6] := HT[-1,-3,-4,1]*T[-2,1,-5,-6]
    global HT = reshape(HT,chi*d,d,chi*d,d)
    append!(HTnorm,[maximum(HT)])
    global HT = HT/maximum(HT)
end
