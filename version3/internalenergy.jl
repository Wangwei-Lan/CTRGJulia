
function InternalEnergyTreeRG(RGTensor::Array{Array{Float64}})
    chi = size(RGTensor[4],1)
    cl = RGTensor[2] ; cr = RGTensor[3]; t = RGTensor[1]
    t1 = RGTensor[4];t2 = RGTensor[5] ; t1e = RGTensor[8]
    #
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1,:LM,maxiter=300)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1,:LM,maxiter=300)
    @tensor Numerator1[] := real(eigvecsup)[1][1,3]*real(eigvecsdn)[1][6,5]*RGTensor[4][1,4,6,2]*RGTensor[8][3,2,5,4]
    @tensor Denominator1[] := real(eigvecsup[1])[1,3]*real(eigvecsdn)[1][6,5]*RGTensor[4][1,4,6,2]*RGTensor[4][3,2,5,4]
    #
    #
    @tensor Numerator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1e[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]
    @tensor Denominator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]
    #
    return Numerator[1]/Denominator[1],Numerator1[1]/Denominator1[1]
end
