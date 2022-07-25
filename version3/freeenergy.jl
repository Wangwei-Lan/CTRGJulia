function FreeEnergyDensity(RGTensor::Array{Array{Float64}},RGnorm::Array{Array{Float64}},NumLayer::Int64)
    chi = size(RGTensor[4],1)
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1)
    eigup = reshape(eigvecsup[1],chi^2)
    eigdn = reshape(eigvecsdn[1],chi^2)
    @tensor Z1[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*RGTensor[4][1,4,5,2]*RGTensor[4][3,2,6,4]
    Lambda = Z1[1]/LinearAlgebra.dot(eigup,eigdn)
    FE = (log(real(Lambda))+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*NumLayer+2))

    #=
    eigvalup,eigvecsup = eigsolve(y->ApplyVT1(RGTensor[4],y),rand(chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT1(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi),1)
    eigup = reshape(eigvecsup[1],chi)
    eigdn = reshape(eigvecsdn[1],chi)
    Z = LinearAlgebra.dot(eigup,eigdn)
    @tensor Z1[] := eigvecsup[1][1]*eigvecsdn[1][3]*RGTensor[4][1,2,3,2]
    Lambda = Z1[1]/LinearAlgebra.dot(eigup,eigdn)
    FE = (log(real(Lambda))+sum(log.(RGnorm[4])))/((2*NumLayer+2))
    =#

    return FE
end

function FreeEnergyDensitySingle(RGTensor::Array{Array{Float64}},RGnorm::Array{Array{Float64}},NumLayer::Int64)
    chi = size(RGTensor[4],1)
    #=
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1)
    eigup = reshape(eigvecsup[1],chi^2)
    eigdn = reshape(eigvecsdn[1],chi^2)
    @tensor Z1[] := eigvecsup[1][1,3]*eigvecsdn[1][5,6]*RGTensor[4][1,4,5,2]*RGTensor[4][3,2,6,4]
    Lambda = Z1[1]/LinearAlgebra.dot(eigup,eigdn)
    FE = (log(real(Lambda))+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*NumLayer+2))
    =#
    #
    eigvalup,eigvecsup = eigsolve(y->ApplyVT1(RGTensor[4],y),rand(chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT1(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi),1)
    eigup = reshape(eigvecsup[1],chi)
    eigdn = reshape(eigvecsdn[1],chi)
    Z = LinearAlgebra.dot(eigup,eigdn)
    @tensor Z1[] := eigvecsup[1][1]*eigvecsdn[1][3]*RGTensor[4][1,2,3,2]
    Lambda = Z1[1]/LinearAlgebra.dot(eigup,eigdn)
    FE = (log(real(Lambda))+sum(log.(RGnorm[4])))/((2*NumLayer+2))
    #
    return FE
end


#
function FreeEnergyDensityCenter(RGTensor::Array{Array{Float64}},RGnorm::Array{Array{Float64}},NumLayer::Int64)
    chi = size(RGTensor[2],1)
    eigvalup,eigvecsup = eigsolve(y->ApplyCenter(RGTensor[2],RGTensor[3],y,direction="up"),rand(chi),1)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyCenter(RGTensor[2],RGTensor[3],y,direction="dn"),rand(chi),1)

    eigup = reshape(eigvecsup[1],chi)
    eigdn = reshape(eigvecsdn[1],chi)
    Z = LinearAlgebra.dot(eigup,eigdn)
    @tensor Z1[] := eigvecsup[1][1]*eigvecsdn[1][4]*RGTensor[2][1,2,3]*RGTensor[3][4,2,3]
    Lambda = Z1[1]/LinearAlgebra.dot(eigup,eigdn)


    Ns = (2*NumLayer+2)^2
    FE = (log(real(Lambda))+sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
            sum([NumLayer:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([NumLayer-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
    return FE
end
#
